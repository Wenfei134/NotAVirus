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

POS_DIR = "./proc_positive_Covid-19" #NOTE: this dataset has been reduced to end at positive_0_2094.jpg (336 of the original positives - to match the number of negatives)
NEG_DIR = "./proc_negative_Covid-19"

# Uses the serialized model generated from main.py to classify mel spectrograms.
# Will return a label representing covid/not covid
def classify_image(filepath):
    # Load in the model with a test sample
    reconstructed_model = tf.keras.models.load_model("finalized_model_c") #finalized_model_c is the best so far
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    test_input = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    test_input = np.array(test_input)

    test_input = test_input.reshape(1, test_input.shape[0], test_input.shape[1], 1)
    print(test_input.shape)
    print(reconstructed_model.predict(test_input))


# classify_image(os.path.join(POS_DIR, "positive_0_3.jpg")) # test
# classify_image(os.path.join(POS_DIR, "positive_0_23.jpg")) # test
# classify_image(os.path.join(POS_DIR, "positive_0_156.jpg")) # test
# classify_image(os.path.join(NEG_DIR, "negative_0_710.jpg")) # test
# classify_image(os.path.join(NEG_DIR, "negative_0_466.jpg")) # test
# classify_image(os.path.join(NEG_DIR, "negative_0_508.jpg")) # test
# classify_image(os.path.join(NEG_DIR, "negative_0_771.jpg")) # test


# Bulk test
# dirs = [ NEG_DIR] #POS_DIR,

# for dir in dirs:
#     # Rotate all images
#     for img_name in os.listdir(dir):
#         path = os.path.join(dir, img_name)
#         classify_image(path)
        
