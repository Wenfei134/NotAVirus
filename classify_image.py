import tensorflow.keras.models as models
import os
import tensorflow as tf
import cv2
import numpy as np

#Input Size
WIDTH_IN = 320 #432
HEIGHT_IN = 240 #288

WIDTH = int(WIDTH_IN/2)
HEIGHT = int(HEIGHT_IN/2)

POS_DIR = "./proc_positive_Covid-19"

# Uses the serialized model generated from main.py to classify mel spectrograms.
# Will return a label representing covid/not covid
def classify_image(image):
    # Load in the model with a test sample
    reconstructed_model = tf.keras.models.load_model("finalized_model")
    # test_input = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    test_input = np.array(test_input)

    test_input = test_input.reshape( 1, test_input.shape[0], test_input.shape[1], 1)
    print(test_input.shape)
    print(reconstructed_model.predict(test_input))


classify_image(os.path.join(POS_DIR, "positive_0_3.jpg")) # test
