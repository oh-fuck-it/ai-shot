import tensorflow as tf
import numpy as np
from utils.score_utils import mean_score, std_score
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet import preprocess_input

resize_image = True
img_path = 'C:\\Users\\holk\\Documents\\File\\0a256522-1550-11ec-ab0e-64bc580330d5.png'

target_size = (1080, 1080) if resize_image else None
img = load_img(img_path, target_size=target_size)
x = img_to_array(img)
x = np.expand_dims(x, axis=0)
x = preprocess_input(x)

interpreter = tf.lite.Interpreter(model_path="model/mobilenet_weights.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_shape = input_details[0]['shape']
interpreter.set_tensor(input_details[0]['index'], x)

interpreter.invoke()

scores = interpreter.get_tensor(output_details[0]['index'])
mean = mean_score(scores)
std = std_score(scores)
print(mean, std)