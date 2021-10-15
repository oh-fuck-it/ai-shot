import tensorflow as tf

resize_image = True
img_path = '/File/0apwACX-W2Y.png'
#
target_size = (640, 480) if resize_image else None

from algorithm.ImageAssessmentEvaluate.utils.score_utils import mean_score, std_score

# img = load_img(img_path, target_size=target_size)
# x = img_to_array(img)
# x = np.expand_dims(x, axis=0)
# x = preprocess_input(x)
interpreter = tf.lite.Interpreter(model_path="./algorithm/ImageAssessmentEvaluate/model/mobilenet_weights.tflite")
interpreter.allocate_tensors()


# interpreter = tf.lite.Interpreter(model_path="./model/mobilenet_weights.tflite")
# interpreter.allocate_tensors()
# json.loads(x.tolist())
def load_x_data(x: []):
    x = tf.convert_to_tensor(x)
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    interpreter.set_tensor(input_details[0]['index'], x)

    interpreter.invoke()

    scores = interpreter.get_tensor(output_details[0]['index'])
    mean = mean_score(scores)
    std = std_score(scores)
    return mean, std
