import io
import json
import os
import random

import cv2
import numpy as np
from PIL import Image
from flask import Flask, send_file
from flask import request
from keras.applications.mobilenet import preprocess_input

import PILImgZIP
from algorithm.ImageAssessmentEvaluate.evaluate import load_x_data
from algorithm.PoseEstimate.pose_similarity import estimate_similarity_in_all_data

app = Flask(__name__)
root_path = '../File\\'
ip = '127.0.0.1/'


@app.route('/getPhoto', methods=["GET", "POST"])
def getPhoto():
    dirs = os.listdir(root_path)
    photos = [ip + dirs[random.randint(0, len(dirs))] for i in range(0, 15)]
    return json.dumps({
        "code": 200,
        "msg": "success",
        "photos": photos
    })


@app.route('/predict', methods=["GET", "POST"])
def get_predict():
    tensor = json.loads(request.form.to_dict()['img'])
    pic_name = estimate_similarity_in_all_data(tensor)[0]
    with open(root_path + pic_name, 'rb') as img_f:
        bytearray = io.BytesIO(img_f.read())
        bytearray = PILImgZIP.compress_img_PIL(bytearray, compress_rate=0.1)
        img_f.close()
        return send_file(
            bytearray,
            mimetype='image/png',
            as_attachment=True,
            attachment_filename=pic_name
        )


def file_2_bytes(file):
    byte_stream = io.BytesIO(file)
    im2 = Image.open(byte_stream).resize(size=(480, 640))
    img = np.asarray(im2, dtype=np.float32)
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x


@app.route("/getTips", methods=["GET", "POST"])
def getTips():
    upload_file = request.files['file'].read()
    x = file_2_bytes(upload_file)


@app.route('/markerImg', methods=['POST'])
def postImg():
    upload_file = request.files['file'].read()
    x = file_2_bytes(upload_file)
    mean, std = load_x_data(x)
    print(mean, std)
    return {
        "mean": {
            'data': mean,
            'description': "评分标准值"
        },
        "std": {
            'data': std,
            'description': "标准差"
        }
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
