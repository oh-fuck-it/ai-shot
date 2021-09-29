import json
import os
import tensorflow as tf
import numpy as np
import tensorflow_hub as hub

model = hub.load("../PoseEstimate/movenet_singlepose_thunder")
joints = ['鼻子', '左眼', '右眼', '左耳', '右耳', '左肩', '右肩', '左肘', '右肘', '左腕', '右腕', '左腰', '右腰', '左膝', '右膝', '左脚', '右脚']


def estimate(image_path):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    net = model.signatures['serving_default']
    outputs = net(image)
    key_points = outputs['output_0'].numpy()[0][0]
    return key_points


class Tips:
    def __call__(self, reference_joints, pred_joints, threshold):
        self.threshold = threshold
        self.reference_score = reference_joints[:, 2]
        self.pred_score = pred_joints[:, 2]
        self.reference_coordinates = reference_joints[:, :2]
        self.pred_coordinates = pred_joints[:, :2]

    def __init__(self, reference_joints, pred_joints, threshold):
        self.threshold = threshold
        self.reference_score = reference_joints[:, 2]
        self.pred_score = pred_joints[:, 2]
        self.reference_coordinates = reference_joints[:, :2]
        self.pred_coordinates = pred_joints[:, :2]

    def diff_of_coordinate(self):
        rc = self.reference_coordinates
        pc = self.pred_coordinates
        res = []
        for i in range(17):
            t = pc[i] - rc[i]
            # if t[0]

    def tips_of_score(self):
        thd = self.threshold
        rs = self.reference_score
        ps = self.pred_score
        tmp = {}
        res = []
        for i in range(17):
            rs_s = True if rs[i] > thd else False
            ps_s = True if ps[i] > thd else False
            if rs_s != ps_s:
                diff = rs[i] - ps[i]
                tmp[diff] = f'请{"露出" if rs_s else "隐藏"}{joints[i]}'
        for k in sorted(tmp, reverse=True):
            res.append(tmp[k])
        return res


ref = estimate(
    'C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\File\\0a12508c-1550-11ec-8261-64bc580330d5.png')
pred = estimate(
    'C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\File\\0b3d89bb-1550-11ec-838d-64bc580330d5.png')

tips = Tips(ref, pred, 0.3)
print(tips.diff_of_coordinate())
print(tips.tips_of_score())
