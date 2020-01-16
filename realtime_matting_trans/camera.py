'''
Author  : Zhengwei Li
Version : 1.0.0 
'''
import time
import cv2
import torch 
import pdb
import argparse
import numpy as np
import os 
from collections import OrderedDict
import torch.nn.functional as F

parser = argparse.ArgumentParser(description='human matting')
parser.add_argument('--model', default='./model/*.pt', help='preTrained model')
parser.add_argument('--without_gpu', action='store_true', default=False, help='use cpu')
parser.add_argument('--only_bg', action='store_true', default=True, help='use cpu')

if __name__ == "__main__":
    args = parser.parse_args()
else:
    args = parser.parse_args("")

torch.set_grad_enabled(False)
INPUT_SIZE = 256


#################################
#----------------
if args.without_gpu:
    print("use CPU !")
    device = torch.device('cpu')
else:
    if torch.cuda.is_available():
        n_gpu = torch.cuda.device_count()
        print("----------------------------------------------------------")
        print("|       use GPU !      ||   Available GPU number is {} !  |".format(n_gpu))
        print("----------------------------------------------------------")

        device = torch.device('cuda:0,1')

#################################
#---------------
def load_model(args):
    print('Loading model from {}...'.format(args.model))
    if args.without_gpu:
        myModel = torch.load(args.model, map_location=lambda storage, loc: storage)
    else:
        myModel = torch.load(args.model)
    myModel.avgpool = torch.nn.AvgPool2d(kernel_size=7, stride=1, padding=0)
    myModel.eval()
    myModel.to(device)
    
    return myModel


from background_transfer.inference import real_time_inference


def seg_process(args, image, net):

    # opencv
    origin_img = image.copy()
    origin_h, origin_w, c = image.shape
    image_resize = cv2.resize(image, (INPUT_SIZE,INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
    image_resize = (image_resize - (104., 112., 121.,)) / 255.0


    tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE)
    
    tensor_4D[0,:,:,:] = torch.FloatTensor(image_resize.transpose(2,0,1))
    inputs = tensor_4D.to(device)
    # -----------------------------------------------------------------

    t0 = time.time()

    seg, alpha = net(inputs)

    print((time.time() - t0))  

    if args.without_gpu:
        alpha_np = alpha[0,0,:,:].data.numpy()
    else:
        alpha_np = alpha[0,0,:,:].cpu().data.numpy()


    fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)


    # -----------------------------------------------------------------
    fg = np.multiply(fg_alpha[..., np.newaxis], image)


    # gray
    bg = image
    bg_alpha = 1 - fg_alpha[..., np.newaxis]
    bg_alpha[bg_alpha<0] = 0

    bg_gray = np.multiply(bg_alpha, image)
    #bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

    #bg[:,:,0] = bg_gray
    #bg[:,:,1] = bg_gray
    #bg[:,:,2] = bg_gray
    bg = bg_gray
    # -----------------------------------------------------------------

    src = cv2.imread('/Users/yifyang/Documents/workspace/mobile_phone_human_matting/background1.jpg')
    height, width = image.shape[:2]

    src = cv2.resize(src, (width, height), interpolation=cv2.INTER_AREA)
    fg = cv2.resize(fg, (width, height), interpolation=cv2.INTER_AREA)
    fg = fg.astype(np.uint8)


    fg_edge_mask = np.zeros(fg.shape, fg.dtype)
    fg_edge_mask[(fg_alpha < 0.996).__and__(fg_alpha > 0.005)] = 255

    fg_mask = np.zeros(fg.shape, fg.dtype)
    fg_ones_mask = np.ones(fg.shape, fg.dtype) * 255

    # 绘制多边形
    #cv2.fillPoly(fg_mask, [poly], (255, 255, 255))
    fg_mask[fg > 0] = 255
    # 飞机中心位置在dst的坐标
    center = (width // 2,  height // 2)
    # 泊松融合.

    # fg : olor, bg : gray
    #out = fg + bg
    # fg : color
    if args.only_bg == True:
        out = bg
        out_temp = out.copy()
        out = out.astype(np.uint8)
        out = real_time_inference(out)
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
        out[out_temp<1] = origin_img[out_temp<1]
        out[out_temp>255] = origin_img[out_temp>255]
        out = out.astype(np.uint8)
        out = cv2.seamlessClone(origin_img, out, fg_edge_mask, center, cv2.MIXED_CLONE)

    else:
        out = fg
        out[out<0.1] = src[out<0.1]
        out[out>255] = src[out>255]
        out = out.astype(np.uint8)
        out = real_time_inference(out)
        out = cv2.resize(out, (width, height), interpolation=cv2.INTER_AREA)
    #out = cv2.seamlessClone(fg, src, fg_mask, center, cv2.MIXED_CLONE)


    return out


def camera_seg(args, net):

    videoCapture = cv2.VideoCapture(0)


    while(1):
        # get a frame
        ret, frame = videoCapture.read()
        frame = cv2.flip(frame,1)
        origin_h, origin_w, c = frame.shape
        frame = cv2.resize(frame, (origin_w // 2, origin_h// 2), interpolation=cv2.INTER_CUBIC)

        frame_seg = seg_process(args, frame, net)


        # show a frame
        cv2.imshow("capture", frame_seg)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    videoCapture.release()
class Mutting():
    def __init__(self):
       self.net = load_model(args)

    def seg_process(self, args, image):

        # opencv
        origin_img = image.copy()
        origin_h, origin_w, c = image.shape
        image_resize = cv2.resize(image, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_CUBIC)
        image_resize = (image_resize - (104., 112., 121.,)) / 255.0

        tensor_4D = torch.FloatTensor(1, 3, INPUT_SIZE, INPUT_SIZE)

        tensor_4D[0, :, :, :] = torch.FloatTensor(image_resize.transpose(2, 0, 1))
        inputs = tensor_4D.to(device)
        # -----------------------------------------------------------------

        t0 = time.time()

        seg, alpha = self.net(inputs)

        print((time.time() - t0))

        if args.without_gpu:
            alpha_np = alpha[0, 0, :, :].data.numpy()
        else:
            alpha_np = alpha[0, 0, :, :].cpu().data.numpy()

        fg_alpha = cv2.resize(alpha_np, (origin_w, origin_h), interpolation=cv2.INTER_CUBIC)

        # -----------------------------------------------------------------
        fg = np.multiply(fg_alpha[..., np.newaxis], image)

        # gray
        bg = image
        bg_alpha = 1 - fg_alpha[..., np.newaxis]
        bg_alpha[bg_alpha < 0] = 0

        bg_gray = np.multiply(bg_alpha, image)
        #bg_gray = cv2.cvtColor(bg_gray, cv2.COLOR_BGR2GRAY)

        bg[:, :, 0] = bg_gray
        bg[:, :, 1] = bg_gray
        bg[:, :, 2] = bg_gray
        # -----------------------------------------------------------------
        height, width = image.shape[:2]

        fg = cv2.resize(fg, (width, height), interpolation=cv2.INTER_AREA)
        fg = fg.astype(np.uint8)

        fg_edge_mask = np.zeros(fg.shape, fg.dtype)
        fg_edge_mask[(fg_alpha < 0.996).__and__(fg_alpha > 0.005)] = 255

        fg_mask = np.zeros(fg.shape, fg.dtype)
        fg_ones_mask = np.ones(fg.shape, fg.dtype) * 255

        fg_mask[fg > 0] = 255
        center = (width // 2, height // 2)
        # 泊松融合.

        # fg : olor, bg : gray
        # out = fg + bg
        # fg : color
        return bg, fg

    def get_bg_fg(self, image):
        return seg_process(args, image)


def main(args):
    myModel = load_model(args)
    camera_seg(args, myModel)


if __name__ == "__main__":
    main(args)


