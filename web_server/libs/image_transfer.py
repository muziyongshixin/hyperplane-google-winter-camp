import os
import web_server.config as config
# from background_transfer.inference import inference as background_transfer
# from face_transfer.inference import inference as face_transfer
# from matting.camera import  inference as face_matting

def matting_images(image_path):
    os.system('python D:/codes/python code/hyperplane-google-winter-camp/matting/camera.py')
    # front_img_path,back_img_path=face_matting(image_path)
    # print("save front_image  and back_image to {}".format(front_img_path))
    # return front_img_path,back_img_path


def transfer_human_face(image_path):
    result_image_path = face_transfer(image_path )
    print('get anime face image at {}'.format(result_image_path))
    return result_image_path


def merge_images(human_image_path, back_mask_path, background_image_path):
    result_image = None
    return result_image


def transfer_background(image_path):
    result_image_path = background_transfer(image_path)
    return result_image_path


def image_transfer(image_path, **kwargs):
    '''
    :param image_path:  保存到instance里的用户上传的图片
    :return: 模型返回的图片路径
    '''

    cur_job_dir = kwargs['cur_job_dir']
    #
    # result_background_path=transfer_background(image_path)
    result_background_path=''
    anime_file_path=transfer_human_face(image_path)
    # front_file_path, back_file_path=matting_images(image_path)
    sample0 = {  "url":anime_file_path
        # "url": 'D:/codes/python code/hyperplane-google-winter-camp/web_server/instance/results/search/efc996dc-3775-11ea-b462-185680d0cf9a/stylized_background.jpg'
    }

    res_list = [sample0]

    return res_list


if __name__=="__main__":
