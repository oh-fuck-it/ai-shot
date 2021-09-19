import tensorflow as tf
import numpy as np
import json
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()
filename = 'result.json'
with open(filename) as file_obj:
    pics = json.load(file_obj)


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
    # print(pic2_reshape, pic1_reshape)
    return tf.reduce_mean(cosine_similarity(pic1_reshape, pic2_reshape))


def estimate_similarity_in_all_data(pic_name, data=pics):
    res = [0, 0]
    data = tf.convert_to_tensor(0)

    coordinate = tf.convert_to_tensor(pics[pic_name])[:, :2]
    score = tf.convert_to_tensor(pics[pic_name])[:, 2]
    pic_reshape = tf.reshape(coordinate, [1, 34])

    for pic in pics:
        if pic == pic_name:
            continue
        data_coordinate = tf.convert_to_tensor(pics[pic])[:, :2]
        data_reshape = tf.reshape(data_coordinate, [1, 34])
        # print(cosine_similarity(pic_reshape, data_reshape).numpy()[0])
        similarity = cosine_similarity(pic_reshape, data_reshape)
        tf.reshape(similarity, [])
        # print(data)
        if similarity > res[1]:
            res[0] = pic
            res[1] = similarity
            data = data_reshape
    return res


print(estimate_similarity_in_all_data('cuStP_i-xPg.png'))
# print(estimate_two_pic_similarity('02ada987-1550-11ec-8457-64bc580330d5.png','02ada987-1550-11ec-8457-64bc580330d5.png').)
