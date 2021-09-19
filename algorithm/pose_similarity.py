import time

import tensorflow as tf
import numpy as np
import json

import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
filename = '../result.json'
with open(filename) as file_obj:
    print(file_obj)
    pics = json.load(file_obj)
for p in pics:
    pics[p] = tf.reshape(tf.convert_to_tensor(pics[p])[:, :2], [1, 34])


def cosine_similarity(q, a):
    q = tf.convert_to_tensor(q)
    a = tf.convert_to_tensor(a)
    pooled_len_1 = tf.sqrt(tf.reduce_sum(q * q, 1))
    pooled_len_2 = tf.sqrt(tf.reduce_sum(a * a, 1))
    pooled_mul_12 = tf.reduce_sum(q * a, 1)
    score = tf.math.divide(pooled_mul_12, pooled_len_1 * pooled_len_2 + 1e-8, name="scores")
    return score


def estimate_two_pic_similarity(pic1_name, pic2_name):
    if pic1_name not in pics and pic2_name not in pics:
        raise Exception('invalid pic name')
    coordinate1 = tf.convert_to_tensor(pics[pic1_name])[:, :2]
    coordinate2 = tf.convert_to_tensor(pics[pic2_name])[:, :2]
    score = tf.convert_to_tensor(pics[pic1_name])[:, 2]
    pic1_reshape = tf.reshape(coordinate1, [1, 34])
    pic2_reshape = tf.reshape(coordinate2, [1, 34])
    return [tf.reduce_mean(cosine_similarity(pic1_reshape, pic2_reshape)), score]


def estimate_similarity_in_all_data(coordinate, dataset=pics):
    res = [0, 0]
    pic_reshape = tf.reshape(tf.convert_to_tensor(coordinate)[:, :2], [1, 34])
    for pic in dataset:
        data_coordinate = dataset[pic]
        similarity = cosine_similarity(pic_reshape, data_coordinate)
        tf.reshape(similarity, [])
        if similarity > res[1]:
            res[0] = pic
            res[1] = similarity
    return res
