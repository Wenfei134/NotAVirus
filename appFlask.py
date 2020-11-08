from flask import Flask, render_template, request
from getSpectrogram import submitAudio
import classify_image as ci
import os

app = Flask(__name__, static_folder='./frontend/build/static/', template_folder='./frontend/build/')
app.config['UPLOAD_FOLDER'] = './'
app.config['MAX_CONTENT_LENGTH'] = 32 * 1024 * 1024
i = 0

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():  
    if request.method == 'POST': 
        file = request.files['audiofile']
        print(file)
        filename = file.filename
        path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save( path )
        melSpec = submitAudio( path )
        results = ci.classify_image(melSpec)
        # if melSpec not None: 
        #     prediction = getPrediction( './audio.jpg')

        return results  
    else: 
        return "hello"

if __name__=="__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)