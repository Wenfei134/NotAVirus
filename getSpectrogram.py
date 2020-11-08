from pydub import AudioSegment
import wave
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.cm as cm
matplotlib.use("agg")
import librosa
import librosa.display
import cv2
import io
import image_preprocess as ipre

def convertToMono(file): 
    sound = AudioSegment.from_wav( file )
    sound = sound.set_channels(1)
    sound.export(file, format="wav")   

def getMelSpectrogramOld(file):
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
    framesToRead = int( min( sampleRate * 1.5, totalFrames ) )
    data = wav_file.readframes( framesToRead )
    wav_file.close()

    #writes 3 seconds (or less) of the submitted audio
    wav_file = wave.open( "./audio.wav", 'wb' )
    params = list( params )
    params[3] = framesToRead
    params = tuple( params )
    wav_file.setparams( params )
    wav_file.writeframes( data )
    wav_file.close()

    y, sr = librosa.load("./audio.wav")
    n_fft = 512
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=32)
    mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)

    librosa.display.specshow(mel_spect_dB)
    plt.plot(y)
    plt.savefig( "./audio.jpg", pil_kwargs={'progressive': True}, bbox_inches='tight', pad_inches=0)
    return True 

# define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=10):
    buf = io.BytesIO()
    plt.savefig( "./audio.jpg", pil_kwargs={'progressive': True}, bbox_inches='tight', pad_inches=0)
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, cv2.IMREAD_GRAYSCALE)

    return img

def getMelSpectrogram(file):
    print(file)
    y, sr = librosa.load(file, mono=False)
    y = librosa.to_mono(y)
    n_fft = 512
    mel_spect = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=n_fft, hop_length=32)
    mel_spect_dB = librosa.power_to_db(mel_spect, ref=np.max)
    fig, ax = plt.subplots(figsize=(16, 12))
    im = ax.imshow(mel_spect_dB, cmap=cm.plasma)
    img = get_img_from_fig(fig)
    # showImage = cv2.imshow("grayscale_spec", img)
    return img
    # plt.savefig( "./audio.jpg", pil_kwargs={'progressive': True}, bbox_inches='tight', pad_inches=0)

def submitAudio( file ):
    # wav_file = wave.open( file )
    # isMono = wav_file.getnchannels()
    # wav_file.close()

    #can only use one channel 
    # if isMono != 1:
    #     convertToMono( file )

    # wav_file = wave.open( file )
    # params = wav_file.getparams()
    # sampleRate = params[2]
    # totalFrames = params[3]
    # #gets 3 seconds (or less) of the audio
    # framesToRead = min( sampleRate * 3, totalFrames )
    # data = wav_file.readframes( framesToRead )
    # wav_file.close()

    # #writes 3 seconds (or less) of the submitted audio
    # wav_file = wave.open( "./audio.wav", 'wb')
    # params = list( params )
    # params[3] = framesToRead
    # params = tuple( params )
    # wav_file.setparams( params )
    # wav_file.writeframes( data )
    return getMelSpectrogram(file)
