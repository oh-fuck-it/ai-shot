import io
import json

from flask import Flask, send_file
from flask import request

import PILImgZIP
from algorithm.pose_similarity import estimate_similarity_in_all_data

app = Flask(__name__)
root_path = '../File\\'


@app.route('/predict', methods=["GET", "POST"])
def get_predict():
    tensor = json.loads(request.form.to_dict()['img'])
    pic_name = estimate_similarity_in_all_data(tensor)[0]
    with open(root_path + pic_name, 'rb') as img_f:
        bytearray = io.BytesIO(img_f.read())
        bytearray = PILImgZIP.Compress_img().compress_img_PIL(bytearray, compress_rate=0.1)
        img_f.close()
        return send_file(
            bytearray,
            mimetype='image/png',
            as_attachment=True,
            attachment_filename=pic_name
        )


if __name__ == '__main__':
    app.run()
