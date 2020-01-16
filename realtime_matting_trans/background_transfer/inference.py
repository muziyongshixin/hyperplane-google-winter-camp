import os
import re
import torch
import argparse
import torchvision
from . import autoencoder
from .log_utils import get_logger

from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms.functional as transforms
import numpy as np
import PIL
import cv2
log = get_logger()

import torch
import math
irange = range


def make_grid(tensor, nrow=8, padding=2,
              normalize=False, range=None, scale_each=False, pad_value=0):
    """Make a grid of images.
    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.
    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            grid.narrow(1, y * height + padding, height - padding)\
                .narrow(2, x * width + padding, width - padding)\
                .copy_(tensor[k])
            k = k + 1
    return grid


def parse_args():
    parser = argparse.ArgumentParser(
        description='Pytorch implementation of arbitrary style transfer via CNN features WCT trasform',
        epilog='Supported image file formats are: jpg, jpeg, png')

    parser.add_argument('--content', default='background_transfer/inputs/contents/1.jpg',
                        help='Path of the content image (or a directory containing images) to be trasformed')
    parser.add_argument('--style', default='background_transfer/inputs/styles/1.jpg',  help='Path of the style image (or a directory containing images) to use')
    parser.add_argument('--synthesis', default=False, action='store_true',
                        help='Flag to syntesize a new texture. Must provide a texture style image')
    parser.add_argument('--stylePair', help='Path of two style images (separated by ",") to use in combination')
    parser.add_argument('--mask',
                        help='Path of the binary mask image (white on black) to trasfer the style pair in the corrisponding areas')

    parser.add_argument('--contentSize', type=int, default=256,
                        help='Reshape content image to have the new specified maximum size (keeping aspect ratio)')  # default=768 in the paper
    parser.add_argument('--styleSize', type=int, default=128,
                        help='Reshape style image to have the new specified maximum size (keeping aspect ratio)')

    parser.add_argument('--outDir', default='outputs',
                        help='Path of the directory where stylized results will be saved')
    parser.add_argument('--outPrefix', help='Name prefixed in the saved stylized images')

    parser.add_argument('--alpha', type=float, default=0.2,
                        help='Hyperparameter balancing the blending between original content features and WCT-transformed features')
    parser.add_argument('--beta', type=float, default=0.5,
                        help='Hyperparameter balancing the interpolation between the two images in the stylePair')
    parser.add_argument('--no-cuda', default=False, action='store_true',
                        help='Flag to enables GPU (CUDA) accelerated computations')
    parser.add_argument('--single-level', default=False, action='store_true',
                        help='Flag to switch to single level stylization')

    return parser.parse_args("")



def load_img(path, new_size):
    img = Image.open(path).convert(mode='RGB')
    if new_size:
        # for fixed-size squared resizing, leave only the following line uncommented in this if statement
        # img = transforms.resize(img, (new_size, new_size), PIL.Image.BICUBIC)
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(new_size * (height / width)), new_size)
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
        else:
            new_shape = (new_size, int(new_size * (width / height)))
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
    return transforms.to_tensor(img)


def validate_args(args):
    supported_img_formats = ('.png', '.jpg', '.jpeg')

    # assert that we have a combinations of cli args meaningful to perform some task
    assert ((args.content and args.style) or (args.content and args.stylePair) or (args.style and args.synthesis) or (
            args.stylePair and args.synthesis) or (args.mask and args.content and args.stylePair))

    if args.content:
        if os.path.isfile(args.content) and os.path.splitext(args.content)[-1].lower().endswith(supported_img_formats):
            pass
        elif os.path.isdir(args.content) and any(
                [os.path.splitext(file)[-1].lower().endswith(supported_img_formats) for file in
                 os.listdir(args.content)]):
            pass
        else:
            raise ValueError(
                "--content '" + args.content + "' must be an existing image file or a directory containing at least one supported image")

    if args.style:
        if os.path.isfile(args.style) and os.path.splitext(args.style)[-1].lower().endswith(supported_img_formats):
            pass
        elif os.path.isdir(args.style) and any(
                [os.path.splitext(file)[-1].lower().endswith(supported_img_formats) for file in
                 os.listdir(args.style)]):
            pass
        else:
            raise ValueError(
                "--style '" + args.style + "' must be an existing image file or a directory containing at least one supported image")

    if args.stylePair:
        if len(args.stylePair.split(',')) == 2:
            args.style0 = args.stylePair.split(',')[0]
            args.style1 = args.stylePair.split(',')[1]
            if os.path.isfile(args.style0) and os.path.splitext(args.style0)[-1].lower().endswith(
                    supported_img_formats) and \
                    os.path.isfile(args.style1) and os.path.splitext(args.style1)[-1].lower().endswith(
                supported_img_formats):
                pass
            else:
                raise ValueError(
                    "--stylePair '" + args.stylePair + "' must be an existing and supported image file paths pair")
            pass
        else:
            raise ValueError('--stylePair must be a comma separeted pair of image file paths')

    if args.mask:
        if os.path.isfile(args.mask) and os.path.splitext(args.mask)[-1].lower().endswith(supported_img_formats):
            pass
        else:
            raise ValueError("--mask '" + args.mask + "' must be an existing and supported image file path")

    if args.outDir != './outputs':
        args.outDir = os.path.normpath(args.outDir)
        if re.search(r'[^A-Za-z0-9- :_\\\/]', args.outDir):
            raise ValueError("--outDir '" + args.outDir + "' contains illegal characters")

    if args.outPrefix:
        args.outPrefix = os.path.normpath(args.outPrefix)
        if re.search(r'[^A-Za-z0-9-_\\\/]', args.outPrefix):
            raise ValueError("--outPrefix '" + args.outPrefix + "' contains illegal characters")

    if args.contentSize and (args.contentSize <= 0 or args.contentSize > 3840):
        raise ValueError("--contentSize '" + args.contentSize + "' have an invalid value (must be between 0 and 3840)")

    if args.styleSize and (args.styleSize <= 0 or args.styleSize > 3840):
        raise ValueError("--styleSize '" + args.styleSize + "' have an invalid value (must be between 0 and 3840)")

    if not 0. <= args.alpha <= 1.:
        raise ValueError("--alpha '" + args.alpha + "' have an invalid value (must be between 0 and 1)")

    if not 0. <= args.beta <= 1.:
        raise ValueError("--beta '" + args.beta + "' have an invalid value (must be between 0 and 1)")

    return args


def save_image(img, content_name, style_name, out_ext, args):
    save_path = os.path.join(args.outDir, (
        args.outPrefix + '_' if args.outPrefix else '') + content_name + '_stylized_by_' + style_name + '_alpha_' + str(
        int(args.alpha * 100)) + '.' + out_ext)
    log.info("save image to {}".format(save_path))
    torchvision.utils.save_image(img.cpu().detach().squeeze(0), save_path)
    return save_path

def save_result_img(img,save_path):
    log.info("save image to {}".format(save_path))
    torchvision.utils.save_image(img.cpu().detach().squeeze(0), save_path)




args = validate_args(parse_args())

if not args.no_cuda and torch.cuda.is_available():
    log.info('Utilizing the first CUDA gpu available')
    args.device = torch.device('cuda:0')
else:
    log.info('Utilizing the cpu for computations')
    args.device = torch.device('cpu')
model = autoencoder.SingleLevelWCT(args)
model.to(device=args.device)
model.eval()

def inference(content_image_path=None):
    if content_image_path is not None:
        args.content = content_image_path
        args.outDir = os.path.dirname(content_image_path)
    try:
        os.makedirs(args.outDir, exist_ok=True)
    except OSError:
        log.exception('Error encoutered while creating output directory ' + args.outDir)

    content_img=load_img(args.content,args.contentSize).to(device=args.device)
    style_img=load_img(args.style,args.styleSize).to(device=args.device)
    out = model(content_img, style_img)
    save_path=os.path.join(args.outDir, 'stylized_background.jpg')
    save_result_img(out,save_path=save_path)
    log.info('Stylization completed, exiting.')


def real_time_inference(content_image):
    style_img = load_img(args.style, args.styleSize).to(device=args.device)
    content_image=cv2Tensor(content_image,args.contentSize)
    content_image=content_image.unsqueeze(0).to(device=args.device)
    style_image=style_img.unsqueeze(0).to(device=args.device)
    out = model(content_image, style_image)
    grid = make_grid(out)
    # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
    out = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
    out=out.squeeze()
    out = Image.fromarray(out)

    out = cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)
    return out


def cv2Tensor(input_img,new_size):
    img = Image.fromarray(cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB))
    if new_size:
        # for fixed-size squared resizing, leave only the following line uncommented in this if statement
        # img = transforms.resize(img, (new_size, new_size), PIL.Image.BICUBIC)
        width, height = img.size
        max_dim_ix = np.argmax(img.size)
        if max_dim_ix == 0:
            new_shape = (int(new_size * (height / width)), new_size)
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
        else:
            new_shape = (new_size, int(new_size * (width / height)))
            img = transforms.resize(img, new_shape, PIL.Image.BICUBIC)
    return transforms.to_tensor(img)

if __name__ == "__main__":

    log.info('Using single-level stylization pipeline')
    content_path='inputs/contents/1.jpg'
    img=cv2.imread(content_path)
    print(img.shape)
    content_image = cv2Tensor(img,512)
    result=real_time_inference(content_image)
    result=result.permute(1,2,0).numpy()
    plt.imshow(result)
    plt.show()
