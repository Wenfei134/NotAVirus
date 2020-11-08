from flask import Flask, render_template, request
from main import getSpectrogram

app = Flask(__name__, static_folder='./frontend/build/static/', template_folder='./frontend/build/')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():  
    if request.method == 'POST': 
        audio = request.files['audiofile']
        success = getSpectrogram( audio )
        return render_template('prediction.html', output = success )


if __name__=="__main__":
    app.run(debug=True, port=5000)