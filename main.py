import tensorflow as tf
import numpy as np
import cv2
import os
import sys
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Change
EPOCHS = 10 # how many times to train

#Input Size
WIDTH = 320
HEIGHT = 240
POS_DIR = "./negative_Covid-19"
NEG_DIR = "./positive_Covid-19"

def main():
    
    # Split images into testing/data
    images, labels = load_images(POS_DIR, NEG_DIR)
    # labels = tf.keras.utils.to_categorical(labels) 
    # images = np.array(images)
    # labels = np.array(labels)
    # X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2)

    # print(X_train)

    # Build the CNN
    # model = build_model()

    # # Fit
    # model.fit(X_train, Y_train, epochs=EPOCHS)

    # Evaluate
    return

    
def load_images(pos_dir, neg_dir):
    '''
    Load images from directory "dir"
    Return images and labels 
    '''
    images = []
    labels = []

    # Load postive images with opencv, and label as 1
    for image in os.listdir(pos_dir):
        path = os.path.join(pos_dir, image)
        print(path)
        img = cv2.imread(path) #def: read in 
        try:
            # resize = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) #Can mess around with WIDTH, HEIGHT
            resize = cv2.resize(img, (10,10))
            images.append(resize)
            labels.append(1)
        except Exception as e:
            print(img.shape())
            print(str(e))
    
    # Load negative images, label = 0
    # for image in os.listdir(neg_dir):
    #     path = os.path.join(neg_dir, image)
    #     img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #def: read in
    #     resize = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) #Can mess around with WIDTH, HEIGHT
    #     images.append(resize)
    #     labels.append(0)
    
    return (images, labels) #numpy arrays



def build_model():
    """
    Create CNN
    """
    INPUTSHAPE = (HEIGHT, WIDTH)
    # model = models.Seq
    





    return model

if __name__ == "__main__":
    main()