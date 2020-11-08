from flask import Flask, render_template, request
from getSpectrogram import submitAudio
from classify_image import classify_image

app = Flask(__name__, static_folder='./frontend/build/static/', template_folder='./frontend/build/')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():  
    if request.method == 'POST': 
        audio = request.files['audiofile']
        success = submitAudio( audio )
        if success:
            isCovid = classify_image( "./audio.jpg")
        return render_template('prediction.html', isCovid=isCovid )


if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)