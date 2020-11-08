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


# Change
EPOCHS = 10 # how many times to train

#Input Size
WIDTH = 30
HEIGHT = 30
POS_DIR = "./negative_Covid-19"
NEG_DIR = "./positive_Covid-19"

def main():
    
    # Split images into testing/data
    images, labels = load_images(POS_DIR, NEG_DIR)
    labels = tf.keras.utils.to_categorical(labels) 
    images = np.array(images)
    labels = np.array(labels)

    images = images.reshape(images.shape[0], images.shape[1], images.shape[2], 1)
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
        layers.Conv2D(1, (5,5), padding='same', activation='relu'),
        layers.Conv2D(1, (5,5), padding='same', activation='relu'),
        layers.Conv2D(16, (5,5), padding='same', activation='relu'),
        layers.Conv2D(16, (5,5), padding='same', activation='relu'),
        layers.MaxPool2D((2,2)),
        layers.Dropout(0.15),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dense(n_labels, activation = 'softmax')
    ])

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

def train_model(model, x_test, y_test, x_train, y_train):
    history = model.fit(x_train, y_train, epochs=100, validation_data=(x_test, y_test))
    plt.plot(history.history['accuracy'], label='accuracy')
    plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.ylim([0.5, 1])
    plt.legend(loc='lower right')
    plt.show()

    test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)


def convertToMono(file): 
    sound = AudioSegment.from_wav( file )
    sound = sound.set_channels(1)
    sound.export(file, format="wav")    

def getMelSpectrogram(file):
    y, sr = librosa.load(file)
    n_fft = 2048

    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=32)
    mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect_dB)
    plt.plot(y)
    plt.savefig( "./audio.jpg", pil_kwargs={'progressive': True})
    return True

def getSpectrogram( file ):
    wav_file = wave.open( file )
    isMono = wav_file.getnchannels()
    wav_file.close()

    #can only use one channel 
    if isMono != 1:
        convertToMono( file )

    wav_file = wave.open( file )
    params = wav_file.getparams()
    sampleRate = params[2]
    totalFrames = params[3]
    #gets 3 seconds (or less) of the audio
    framesToRead = min( sampleRate * 3, totalFrames )
    data = wav_file.readframes( framesToRead )
    wav_file.close()

    #writes 3 seconds (or less) of the submitted audio
    wav_file = wave.open( "./audio.wav", 'wb')
    params = list( params )
    params[3] = framesToRead
    params = tuple( params )
    wav_file.setparams( params )
    wav_file.writeframes( data )
    return getMelSpectrogram("./audio.wav")

if __name__ == "__main__":
    main()