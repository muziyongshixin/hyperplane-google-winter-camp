import os
import sys
import uuid
import json
from datetime import datetime

from flask import (Flask, render_template, request, redirect,
                   send_from_directory, jsonify, Response)
import cv2
import numpy as np
# from .libs.boe_search import img_search
from web_server.libs.image_transfer import image_transfer

import zipfile
from flask import send_file

import random

def init_path():
    import importlib
    pass


init_path()

import web_server.config as config

# from facerec.libs.utils import save_image

app = Flask(__name__)
app.config['DEBUG'] = True
app.config['STATIC_FOLDER'] = 'static'
app.config['UPLOAD_FOLDER'] = 'uploads'


@app.route('/results/<path:path>')
@app.route('/<path:path>')
def results(path):
    return send_from_directory(
        os.path.join(app.instance_path, config.RESULT_BASEDIR),
        path)

@app.route('/videos/<path:path>')
def results_2(path):
    return send_from_directory(
        config.DATABASEDIR,
        path)

@app.route('/static/<path:path>')
def faceattr_static(path):
    return app.send_static_file(path)


@app.route('/', methods=['GET', 'POST'])
def search():
    if request.method == 'POST':
        format_ = request.form.get('format')
        attr_models = {}
        detection_options = None
        files = request.files.getlist('files')

        if files and not files[0].filename:
            return redirect(request.referrer)
        files.sort(key=lambda x: x.filename)

        rid = get_model_output(files)

        if format_ == 'json':
            path = os.path.join(app.instance_path, config.RESULT_BASEDIR, 'search', rid, 'result.json')
            with open(path) as f:
                return jsonify(json.load(f))
        return redirect('{}?r={}'.format(request.referrer.split('?')[0], rid))
    return render_template("faceattr.html", detect_models='', attr_models='')


def get_model_output(files ):
    rid = str(uuid.uuid1())

    result_dir = os.path.join(app.instance_path, config.RESULT_BASEDIR, 'search', rid)
    print(app.instance_path)
    if not os.path.exists(result_dir):
        os.makedirs(result_dir)
    results = []
    for i, file in enumerate(files):
        if not file.filename:
            continue

        query_file_path = os.path.join(result_dir, 'source_{}'.format(file.filename))
        if not file.closed:
            file.save(query_file_path)
            file.close()
        print('save query image to %s successfully' % query_file_path)
        cur_job_dir=result_dir
        print('cur_job dir is {}'.format(cur_job_dir))
        result_videos = image_transfer(query_file_path,cur_job_dir=cur_job_dir)  # 得到query video的图片以及 查询结果
        for ele in result_videos:
            ele['url']=os.path.relpath(ele['url'],app.instance_path)
        # # 将result video里的图片保存到instance目录下
        # for index, video_sample in enumerate(result_videos):
        #     video_sample['url'] = os.path.join("videos", os.path.relpath(video_sample['url'], config.DATABASEDIR))

        data = {'img_lists':result_videos}
        results.append(data)

    result = { 'records': results[0] }
    with open(os.path.join(result_dir, 'result.json'), 'w') as f:
        f.write(json.dumps(result))
    return rid


def init_models():
    pass

if __name__ == "__main__":
    init_models()
    app.run('0.0.0.0', 8181,use_reloader=False)
