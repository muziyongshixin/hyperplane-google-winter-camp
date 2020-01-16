from .UGATIT import UGATIT
import argparse
from .utils import *
import os
import collections
from PIL import Image

"""parsing and configuration"""


def parse_args():
    desc = "Tensorflow implementation of U-GAT-IT"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--phase', type=str, default='test', help='[train / test]')
    parser.add_argument('--light', type=str2bool, default=False,
                        help='[U-GAT-IT full version / U-GAT-IT light version]')
    parser.add_argument('--dataset', type=str, default='selfie2anime', help='dataset_name')

    parser.add_argument('--epoch', type=int, default=100, help='The number of epochs to run')
    parser.add_argument('--iteration', type=int, default=10000, help='The number of training iterations')
    parser.add_argument('--batch_size', type=int, default=1, help='The size of batch size')
    parser.add_argument('--print_freq', type=int, default=1000, help='The number of image_print_freq')
    parser.add_argument('--save_freq', type=int, default=1000, help='The number of ckpt_save_freq')
    parser.add_argument('--decay_flag', type=str2bool, default=True, help='The decay_flag')
    parser.add_argument('--decay_epoch', type=int, default=50, help='decay epoch')

    parser.add_argument('--lr', type=float, default=0.0001, help='The learning rate')
    parser.add_argument('--GP_ld', type=int, default=10, help='The gradient penalty lambda')
    parser.add_argument('--adv_weight', type=int, default=1, help='Weight about GAN')
    parser.add_argument('--cycle_weight', type=int, default=10, help='Weight about Cycle')
    parser.add_argument('--identity_weight', type=int, default=10, help='Weight about Identity')
    parser.add_argument('--cam_weight', type=int, default=1000, help='Weight about CAM')
    parser.add_argument('--gan_type', type=str, default='lsgan',
                        help='[gan / lsgan / wgan-gp / wgan-lp / dragan / hinge]')

    parser.add_argument('--smoothing', type=str2bool, default=True, help='AdaLIN smoothing effect')

    parser.add_argument('--ch', type=int, default=64, help='base channel number per layer')
    parser.add_argument('--n_res', type=int, default=4, help='The number of resblock')
    parser.add_argument('--n_dis', type=int, default=6, help='The number of discriminator layer')
    parser.add_argument('--n_critic', type=int, default=1, help='The number of critic')
    parser.add_argument('--sn', type=str2bool, default=True, help='using spectral norm')

    parser.add_argument('--img_size', type=int, default=256, help='The size of image')
    parser.add_argument('--img_ch', type=int, default=3, help='The size of image channel')
    parser.add_argument('--augment_flag', type=str2bool, default=True, help='Image augmentation use or not')

    parser.add_argument('--checkpoint_dir', type=str,
                        default='D:/codes/python code/hyperplane-google-winter-camp/face_transfer/checkpoint/',
                        help='Directory name to save the checkpoints')
    parser.add_argument('--result_dir', type=str, default='results',
                        help='Directory name to save the generated images')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory name to save training logs')
    parser.add_argument('--sample_dir', type=str, default='samples',
                        help='Directory name to save the samples on training')
    parser.add_argument('--test_img_path', type=str,
                        default='D:/codes/python code/hyperplane-google-winter-camp/face_transfer/dataset/selfie2anime/testA/timg.jpg',
                        help='test_image_path')

    return check_args(parser.parse_args())


"""checking arguments"""


def check_args(args):
    # --checkpoint_dir
    check_folder(args.checkpoint_dir)

    # --result_dir
    check_folder(args.result_dir)

    # --result_dir
    check_folder(args.log_dir)

    # --sample_dir
    check_folder(args.sample_dir)

    # --epoch
    try:
        assert args.epoch >= 1
    except:
        print('number of epochs must be larger than or equal to one')

    # --batch_size
    try:
        assert args.batch_size >= 1
    except:
        print('batch size must be larger than or equal to one')
    return args


"""main"""
# parse arguments
args = parse_args()
if args is None:
    raise ValueError("args in the face inference model is None")


class Context(object):
    def __init__(self, sess=None, gan=None, initialized=False):
        self.sess = sess
        self.gan = gan
        self.initialized = initialized


global context
context = Context()

# print('Initialized')
if not context.initialized:
    context.initialized = True
    print('Initialized')
    graph = tf.get_default_graph()
    context.sess = tf.Session(
        graph=graph,
        config=tf.ConfigProto(allow_soft_placement=True))
    context.gan = gan = UGATIT(context.sess, args)

    # build graph
    gan.build_model()

    # init
    context.sess.run(tf.global_variables_initializer())
    gan.load_test_model()


# show network architecture
# show_all_variables()

def inference(input_img_path=None, **kwargs):
    # open session
    # with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

    test_img_path = args.test_img_path
    if input_img_path is not None:
        test_img_path = input_img_path

    save_dir = os.path.dirname(input_img_path)
    img = Image.open(test_img_path)
    print(img.size)
    img_size=img.size

    fake_img = context.gan.test(test_image_path=test_img_path)  # , sess=context.sess)

    image_path = os.path.join(save_dir, 'anime.jpg')

    save_images(fake_img, [1, 1], image_path,img_size)
    print('==============UGATIT save image to {}'.format(image_path))
    print(" [*] Face transfer finished!")
    return image_path


if __name__ == '__main__':
    # main()
    input_file_path = 'D:/codes/python code/hyperplane-google-winter-camp/face_transfer/dataset/selfie2anime/testA/bbbb.jpg'
    save_dir = 'D:/codes/python code/hyperplane-google-winter-camp/face_transfer/results/'
    inference(input_img_path=input_file_path, save_dir=save_dir)
