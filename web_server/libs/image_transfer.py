import os
import web_server.config as config
from PIL import Image
import numpy as np

from background_transfer.inference import inference as background_transfer
from face_transfer.inference import inference as face_transfer
from matting.camera import inference as face_matting


def matting_images(image_path):
    # cmd='D:/software/anaconda/envs/tf1.14/python "D:/codes/python code/hyperplane-google-winter-camp/matting/camera.py"'+'  --test_image_path={}'.format(image_path)
    # print( cmd)
    # os.system(cmd)
    # print("matting finished")
    front_img_path, back_img_path = face_matting(image_path)
    print("save front_image  and back_image to {}".format(front_img_path))
    return front_img_path, back_img_path


def transfer_human_face(image_path):
    result_image_path = face_transfer(image_path)
    print('get anime face image at {}'.format(result_image_path))
    return result_image_path


def merge_images(human_image_path, back_mask_path, background_image_path):
    result_image = None
    return result_image


def transfer_background(image_path):
    result_image_path = background_transfer(image_path)
    return result_image_path


def process_anime(mask_path, anime_img_path):
    mask_path = mask_path  # background mask
    res_path = anime_img_path  # anime image

    img = Image.open(mask_path)
    img = img.resize((256, 256))
    img = np.array(img)
    tmp = img
    img[tmp <= 5] = 1
    img[tmp > 5] = 0

    res_img = Image.open(res_path)
    res_img = res_img.resize((256, 256))
    res_img = np.array(res_img)
    res_img = res_img * img
    print(img)
    res_img = Image.fromarray(res_img)

    save_dir = os.path.dirname(anime_img_path)
    save_path = os.path.join(save_dir, 'front_anime.jpg')
    res_img.save(save_path)
    print("save front_anime image file to {}".format(save_path))
    return save_path


def fuse_result(front_anime_path, background_path):
    front_path = front_anime_path
    back_path = background_path
    # w_dim = 256
    # h_dim = 256
    back_w = 0.5
    back_h = 0.5
    front_w = 1
    front_h = 1

    back = Image.open(back_path)
    tmp = np.array(back)
    back = back.resize((int(tmp.shape[1] * back_w), int(tmp.shape[0] * back_h)))
    back = np.array(back)

    front = Image.open(front_path)
    tmp = np.array(front)
    w_dim, h_dim, _ = tmp.shape

    front = front.resize((int(h_dim * front_h), int(w_dim * front_w)))
    front = np.array(front)

    # back[-w_dim:, -h_dim:, :][front!=0] = 0
    back[int(-w_dim * front_w):, int(-h_dim * front_h):, :][front > 15] = front[front > 15]

    # back[-400:, -400:, :] *= front
    res_img = Image.fromarray(back)
    save_dir = os.path.dirname(front_anime_path)
    save_path = os.path.join(save_dir, 'result_image.jpg')
    res_img.save(save_path)

    print("fusion image,save result image to {}".format(save_path))
    return save_path


def image_transfer(image_path, **kwargs):
    '''
    :param image_path:  保存到instance里的用户上传的图片
    :return: 模型返回的图片路径
    '''
    result_background_path = transfer_background(kwargs['destination'])
    # result_background_path = r'D:\codes\python code\hyperplane-google-winter-camp\background_transfer\outputs\加拿大枫林_stylized_by_5_alpha_20.jpg'

    front_mask_path, back_mask_path = matting_images(image_path)
    # back_mask_path = r'D:\codes\python code\hyperplane-google-winter-camp\web_server\instance\results\search\f50baf5c-3845-11ea-875d-185680d0cf9a\back.jpg'
    anime_file_path = transfer_human_face(image_path)

    front_anime_path = process_anime(mask_path=back_mask_path, anime_img_path=anime_file_path)

    fuse_result_path = fuse_result(front_anime_path=front_anime_path, background_path=result_background_path)

    sample0 = {"url": fuse_result_path}
    # "url": 'D:/codes/python code/hyperplane-google-winter-camp/web_server/instance/results/search/efc996dc-3775-11ea-b462-185680d0cf9a/stylized_background.jpg'

    res_list = [sample0]

    return res_list


if __name__ == "__main__":
    # input_path = r"'D:\codes\python code\hyperplane-google-winter-camp\matting\input_img\selfie.jpg'"
    matting_main_path = "D:/codes/python code/hyperplane-google-winter-camp/matting/"
    image_path = "\'D:/codes/python code/hyperplane-google-winter-camp/web_server/instance/results/search/f565dcd2-3844-11ea-a3ac-185680d0cf9a/source_ccc.jpg\'"
    # input_path=os.path.relpath(image_path,matting_main_path)
    input_path = image_path
    matting_images(input_path)
