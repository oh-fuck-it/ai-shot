import io
import json
import os
import random

import numpy as np
from PIL import Image
from flask import Flask, send_file
from flask import request
from keras.applications.mobilenet import preprocess_input

from Server import PILImgZIP
from algorithm.ImageAssessmentEvaluate.evaluate import load_x_data
from algorithm.PhotoTips.photo_tips import Tips
from algorithm.PoseEstimate.pose_similarity import estimate_similarity_in_all_data

app = Flask(__name__)
root_path = './File/'
ip = '0.0.0.0/'


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


temTips: Tips = None


@app.route("/setTips", methods=['POST'])
def setTips():
    global temTips
    fileName = request.form.to_dict()['img']
    temTips = Tips(fileName=fileName)
    return {
        "code": 200,
        "status": "OK"
    }


@app.route("/getTips", methods=['GET', 'POST'])
def getTips():
    global temTips
    if temTips is not None:
        pred_joints = json.loads(request.form.to_dict()['pred_joints'])
        pred_joints = np.array(pred_joints)
        return {
            "code": 200,
            "status": "OK",
            "data": temTips.get_tips(pred_joints=pred_joints)
        }
    return {
        "code": 403,
        "status": "failed,tips not found"
    }


if __name__ == '__main__':
    app.run(host="0.0.0.0", port="5000")
