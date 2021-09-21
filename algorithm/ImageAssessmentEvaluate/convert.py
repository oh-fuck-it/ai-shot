from keras.backend import clear_session
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout
from keras.applications.mobilenet import MobileNet

clear_session()
np.set_printoptions(suppress=True)
input_graph_name = "model/mobilenet_weights.h5"
output_graph_name = input_graph_name[:-3] + '.tflite'

base_model = MobileNet((640, 480, 3), alpha=1, include_top=False, pooling='avg', weights=None)
x = Dropout(0.75)(base_model.output)
x = Dense(10, activation='softmax')(x)

model = Model(base_model.input, x)
model.load_weights('model/mobilenet_weights.h5')
converter = tf.lite.TFLiteConverter.from_keras_model(model=model)
converter.post_training_quantize = True

tflite_model = converter.convert()
open(output_graph_name, "wb").write(tflite_model)
print("generate:", output_graph_name)
