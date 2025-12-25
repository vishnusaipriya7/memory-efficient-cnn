import tensorflow as tf
import numpy as np

MODEL_PATH = "tflite_models/cnn_int8.tflite"

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

def predict(image):
    """
    image: numpy array of shape (32, 32, 3), values [0,255]
    """
    image = image.astype(np.uint8)
    image = np.expand_dims(image, axis=0)

    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()

    output = interpreter.get_tensor(output_details[0]['index'])
    predicted_class = np.argmax(output)

    return predicted_class


# Test deployment
if __name__ == "__main__":
    dummy_image = np.random.randint(0, 255, (32, 32, 3))
    result = predict(dummy_image)
    print("Predicted class:", result)