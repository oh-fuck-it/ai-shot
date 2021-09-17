import tensorflow as tf
import numpy as np
import json
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

filename = 'result.json'
with open(filename) as file_obj:
    pics = json.load(file_obj)


# def handle_joints(pic_name, threshold=0.3):
#     coordinates = []
#     pic = pics[pic_name]
#     for i, joint in enumerate(pic):
#         if i == 0:
#             coordinates.append(joint[:])
#             continue
#         if joint[2] < threshold:
#             joint[:2] = [0, 0]
#         coordinates.append(joint[:2])
#     return coordinates


def cosine_similarity(q, a):
    q = tf.convert_to_tensor(q)
    a = tf.convert_to_tensor(a)
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.math.divide(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    return score


def estimate_similarity(pic1_name, pic2_name):
    if pic1_name not in pics and pic2_name not in pics:
        raise Exception('invalid pic name')
    coordinate = tf.convert_to_tensor(pics[pic1_name])[:, :2]
    score = tf.convert_to_tensor(pics[pic1_name])[:, 2:]
    print(tf.reshape(coordinate, [34, 1]))
    print(tf.reduce_mean(cosine_similarity(tf.reshape(coordinate, [34, 1]),tf.reshape(coordinate, [34, 1]))))

# print(handle_joints('0a2f7da2-1550-11ec-9b51-64bc580330d5.png', 0.3))
print(estimate_similarity('0a256522-1550-11ec-ab0e-64bc580330d5.png', '0a12508c-1550-11ec-8261-64bc580330d5.png'))
