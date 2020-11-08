import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.keras.models as models
import numpy as np
import cv2
import os
import sys
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import wave
import librosa 
import librosa.display
from pydub import AudioSegment

output_model_name = "finalized_model_c"

EPOCHS = 10 # how many times to train

#Input Size
WIDTH_IN = 320 #432
HEIGHT_IN = 240 #288

WIDTH = int(WIDTH_IN/2)
HEIGHT = int(HEIGHT_IN/2)
POS_DIR = "./proc_positive_Covid-19"
NEG_DIR = "./proc_negative_Covid-19"

def main():
    
    # Split images into testing/data
    images, labels = load_images(POS_DIR, NEG_DIR)
    labels = tf.keras.utils.to_categorical(labels) 
    images = np.array(images)
    labels = np.array(labels)

    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
    print(images.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(images, labels, test_size=0.2)
    # print(X_train.shape)
    # print(Y_train.shape)
    # sys.exit(0)
    # Build the CNN
    n_labels = labels.shape[1]
    model = build_model(n_labels)
    train_model(model, X_test, Y_test, X_train, Y_train)

    # # Fit
    # model.fit(X_train, Y_train, epochs=EPOCHS)

    # Evaluate

    
def load_images(pos_dir, neg_dir):
    '''
    Load images from directory "dir"
    Return images and labels 
    '''
    images = []
    labels = []
    # Load postive images with opencv, and label as 1
    for image in tqdm(os.listdir(pos_dir)):
        path = os.path.join(pos_dir, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #def: read in 
        try:
            resize = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) #Can mess around with WIDTH, HEIGHT
            images.append(resize)
            labels.append(1)
        except Exception as e:
            print(str(e))
    
    # Load negative images, label = 0
    for image in tqdm(os.listdir(neg_dir)):
        path = os.path.join(neg_dir, image)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE) #def: read in
        try:
            resize = cv2.resize(img, (WIDTH, HEIGHT), interpolation=cv2.INTER_LINEAR) #Can mess around with WIDTH, HEIGHT
            images.append(resize)
            labels.append(0)
        except Exception as e:
            print(str(e))    

    return (images, labels) #numpy arrays



def build_model(n_labels):
    """
    Create CNN
    """
    INPUTSHAPE = (HEIGHT, WIDTH, 1)
    model = models.Sequential([
        layers.MaxPool2D((2,2), input_shape=INPUTSHAPE),
        layers.Conv2D(16, (5,5), padding='same', activation='relu', input_shape=INPUTSHAPE),
        layers.Conv2D(16, (5,5), padding='same', activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.2),
        layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        layers.Conv2D(32, (5,5), padding='same', activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.2),
        layers.Flatten(),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu'),
        layers.Dense(n_labels, activation = 'softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model, x_test, y_test, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=EPOCHS, validation_data=(x_test, y_test))
    # plt.plot(history.history['accuracy'], label='accuracy')
    # plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    # plt.xlabel('Epoch')
    # plt.ylabel('Accuracy')
    # plt.ylim([0.5, 1])
    # plt.legend(loc='lower right')
    # plt.show()

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)

    # Serialize the model to be able to use quicker for single-sample identification
    model.save(output_model_name)

if __name__ == "__main__":
    main()