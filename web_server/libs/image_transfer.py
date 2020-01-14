import os
import web_server.config as config

def image_transfer(image_path,**kwargs):
    '''
    :param image_path:  保存到instance里的用户上传的图片
    :return: 模型返回的图片路径
    '''
    sample0 = {
        # "url": os.path.join(config.DATABASEDIR, "baggio_penalty_1994/6d1a89c83d554fc6a5e39fcadb172a79baf140fd.mp4"),
        "url": "results/transfer/d33b4e38-3683-11ea-91fa-185680d0cf9a/source_15789750324981206238653569297531.jpg",
        "start": 0,
        "end": 9,
    }

    res_list = [sample0]

    return res_list
