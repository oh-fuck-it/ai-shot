import io
import json

import cv2
import numpy as np
from PIL import Image
from flask import Flask, send_file
from flask import request

import PILImgZIP
from algorithm.ImageAssessmentEvaluate.evaluate import load_x_data
from algorithm.PoseEstimate.pose_similarity import estimate_similarity_in_all_data

app = Flask(__name__)
root_path = '../File\\'


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


@app.route('/marker', methods=['POST'])
def marker():
    x = json.loads(request.form.to_dict()['tensor'])
    mean, std = load_x_data(x)
    return {
        "mean": mean,
        "std": std
    }


@app.route('/postImg', methods=['POST'])
def postImg():
    upload_file = request.files['file'].body
    byte_stream = io.BytesIO(upload_file)
    im2 = Image.open(byte_stream)
    mean, std = load_x_data(im2)
    return {
        "mean": mean,
        "std": std
    }


if __name__ == '__main__':
    app.run()
