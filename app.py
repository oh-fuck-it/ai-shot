from flask import Flask

from algorithm.pose_estimate import predict_base64
from algorithm.pose_similarity import estimate_similarity_in_all_data

app = Flask(__name__)


@app.route('/predict', methods=["GET", "POST"])
def get_predict():
    pre = predict_base64()


if __name__ == '__main__':
    app.run()
