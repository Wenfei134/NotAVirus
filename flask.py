from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/prediction', methods=['POST', 'GET'])
def prediction():  
    if request.method == 'POST': 
        audio = request.files['audiofile']
        return render_template('prediction.html')


