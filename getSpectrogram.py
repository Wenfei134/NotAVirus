from pydub import AudioSegment
import wave
import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display
import cv2
import image_preprocess as ipre

def convertToMono(file): 
    sound = AudioSegment.from_wav( file )
    sound = sound.set_channels(1)
    sound.export(file, format="wav")    

def getMelSpectrogram(file):
    y, sr = librosa.load(file)
    n_fft = 512
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=32)
    mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)
    librosa.display.specshow(mel_spect_dB)
    plt.plot(y)
    plt.savefig( "./audio.jpg", pil_kwargs={'progressive': True}, bbox_inches='tight', pad_inches=0)
    # path = "./audio.jpg"
    # img = cv2.imread(path)
    # pts1, pts2 = ipre.get_four_corners(img) #Scale
    # ipre.rotate_and_save_img(img, "audio.jpg", pts1, pts2)
    return True

def submitAudio( file ):
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

submitAudio("Afro Blue.wav")