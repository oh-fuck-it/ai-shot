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
        d_values = []
        res = []
        for i in range(17):
            d_values.append(pc[i] - rc[i])
            # res.append()

    def diff_of_score(self):
        thd = self.threshold
        rs = self.reference_score
        ps = self.pred_score
        res = []
        d_values = []
        d_list = []
        for i in range(17):
            rs_s = True if rs[i] > thd else False
            ps_s = True if ps[i] > thd else False
            d_list.append(rs_s and ps_s)
            d_values.append(rs[i] - ps[i])
            if not d_list[i]:
                res.append(f'请{"露出" if rs[i] == True else "隐藏"}{joints[i]}')
        return res


ref = estimate('C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\File\\0a256522-1550-11ec-ab0e'
               '-64bc580330d5.png')
pred = estimate('C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\File\\0a12508c-1550-11ec-8261'
                '-64bc580330d5.png')

tips = Tips(ref, pred, 0.4)
# print(tips.diff_of_coordinate())
print(tips.diff_of_score())
