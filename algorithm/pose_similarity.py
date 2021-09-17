import tensorflow as tf
import numpy as np
import json
from tensorflow.python.ops.numpy_ops import np_config

np_config.enable_numpy_behavior()

filename = 'result.json'
with open(filename) as file_obj:
    pics = json.load(file_obj)


def handle_joints(pic_name, threshold=0.3):
    coordinates = []
    pic = pics[pic_name]
    for i, joint in enumerate(pic):
        if i == 0:
            coordinates.append(joint[:])
            continue
        if joint[2] < threshold:
            joint[:2] = [0, 0]
        coordinates.append(joint[:2])
    return coordinates


def highdim_pca(data, n_dim):
    '''
    when n_features(D) >> n_samples(N), highdim_pca is O(N^3)
    :param data: (n_samples, n_features)
    :param n_dim: target dimensions
    :return: (n_samples, n_dim)
    '''
    data = tf.convert_to_tensor(data)
    N = data.shape[0]
    data = data - np.mean(data, axis=0, keepdims=True)

    Ncov = np.dot(data, data.T)

    Neig_values, Neig_vector = np.linalg.eig(Ncov)
    indexs_ = np.argsort(-Neig_values)[:n_dim]
    Npicked_eig_values = Neig_values[indexs_]
    # print(Npicked_eig_values)
    Npicked_eig_vector = Neig_vector[:, indexs_]
    # print(Npicked_eig_vector.shape)

    picked_eig_vector = np.dot(data.T, Npicked_eig_vector)
    picked_eig_vector = picked_eig_vector / (N * Npicked_eig_values.reshape(-1, n_dim)) ** 0.5
    # print(picked_eig_vector.shape)

    data_ndim = np.dot(data, picked_eig_vector)
    return data_ndim


def convert_list_dim(list1):
    res = []
    for i in list1:
        res.append(i[0])
    return res


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
    pic1 = handle_joints(pic1_name)
    pic2 = handle_joints(pic2_name)
    # print(pic1, '\n', pic2)
    # c1 = convert_list_dim(highdim_pca(pic1, 1))
    # c2 = convert_list_dim(highdim_pca(pic2, 1))
    # print("c1", c1)
    # print("c2", c2)
    print(pic1)
    print(pic2)
    # print([c1[i] - c2[i] for i in range(len(c1))])


# print(handle_joints('0a2f7da2-1550-11ec-9b51-64bc580330d5.png', 0.3))
print(estimate_similarity('0a256522-1550-11ec-ab0e-64bc580330d5.png', '0a12508c-1550-11ec-8261-64bc580330d5.png'))
