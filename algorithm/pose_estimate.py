import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
root = 'D:\\TempDemo\\unsplash\\UnsplashDownloader\\File'
root_path = ['\\boy','\\girl','\\lady','\\man','\\Potrait']
# root_path = 'C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\file1'
threshold = 0.5
joints = ['nose', 'l-eye', 'r-eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow',
          'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee',
          'left ankle',
          'right ankle']
joints_dict = {}
filename = 'result.json'
model = hub.load("movenet_singlepose_thunder")


def estimate(image_path):
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    movenet = model.signatures['serving_default']
    outputs = movenet(image)
    key_points = outputs['output_0'].numpy()[0][0]
    base_name = os.path.basename(image_path)
    joints_dict[base_name] = []
    # joints_dict[base_name].append([])
    for i, keypoint in enumerate(key_points):
        joints_dict[base_name].append(keypoint.tolist())
    if tf.reduce_sum(tf.convert_to_tensor(joints_dict[base_name])[:, 2]) < threshold:
        del joints_dict[base_name]


for walk_path in root_path:
    walk_path = root + walk_path
    for filepath, dir_names, filenames in os.walk(walk_path):
        for  image_path in filenames:
            if image_path != '.DS_Store':
                path = os.path.join(filepath, image_path)
                print("now get %s", path)
                estimate(path)

with open(filename, 'w') as file_obj:
    json.dump(joints_dict, file_obj)
