import json
import os
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub

# root_path = 'C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\File'
root_path = 'C:\\Users\\holk\\Documents\\Tencent Files\\1599840925\\FileRecv\\file1'
image_path = root_path
joints = ['nose', 'l-eye', 'r-eye', 'left ear', 'right ear', 'left shoulder', 'right shoulder', 'left elbow',
          'right elbow', 'left wrist', 'right wrist', 'left hip', 'right hip', 'left knee', 'right knee',
          'left ankle',
          'right ankle']
joints_dict = {}
filename = 'result.json'
model = hub.load("movenet_singlepose_thunder")


def estimate(image_path, threshold=0.3):
    image = tf.io.read_file(image_path)
    image = tf.compat.v1.image.decode_jpeg(image)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    # plt.imshow(image)
    image = tf.expand_dims(image, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    movenet = model.signatures['serving_default']
    outputs = movenet(image)
    keypoints = outputs['output_0'].numpy()[0][0]
    base_name = os.path.basename(image_path)
    joints_dict[base_name] = []
    # joints_dict[base_name].append([])
    for i, keypoint in enumerate(keypoints):
        # if keypoint[2] > threshold:
        joints_dict[base_name].append(keypoint.tolist())
        # joints_dict[base_name][0].append(i)


for filepath, dirnames, filenames in os.walk(root_path):
    for image_path in filenames:
        if image_path != '.DS_Store':
            path = os.path.join(filepath, image_path)
            print("now get %s", path)
            estimate(path)
            print(joints_dict)

with open(filename, 'w') as file_obj:
    json.dump(joints_dict, file_obj)
# plt.show()
