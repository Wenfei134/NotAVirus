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
    final_pred = {}
    # Load in the model with a test sample
    reconstructed_model = tf.keras.models.load_model("finalized_model_c") #finalized_model_c is the best so far
    img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    test_input = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR)
    test_input = np.array(test_input)

    test_input = test_input.reshape(1, test_input.shape[0], test_input.shape[1], 1)
    # print(test_input.shape)
    prediction = reconstructed_model.predict(test_input)
    # print(prediction)

    if prediction[0, 0] > prediction[0, 1]:
        final_pred["result"] = 'negative'
        final_pred["confidence"] = prediction[0, 0] * 100
    else:
        final_pred["result"] = 'positive'
        final_pred["confidence"] = prediction[0, 1] * 100
    
    # for field in final_pred:
    #     print(final_pred[field])
    return final_pred


# # Bulk test
# dirs = [NEG_DIR] #POS_DIR,

# for dir in dirs:
#     # Rotate all images
#     for img_name in os.listdir(dir):
#         path = os.path.join(dir, img_name)
#         classify_image(path)
        
