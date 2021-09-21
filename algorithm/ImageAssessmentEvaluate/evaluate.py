import tensorflow as tf
import numpy as np
from utils.score_utils import mean_score, std_score
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.mobilenet import preprocess_input


# resize_image = True
# img_path = 'D:\\TempDemo\\TempDemo\\ai-shot\\File\\0apwACX-W2Y.png'
#
# target_size = (1080, 1080) if resize_image else None
# img = load_img(img_path, target_size=target_size)
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)


def load_x_data(x: []):
    x = tf.convert_to_tensor(x)
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
    return mean,std
