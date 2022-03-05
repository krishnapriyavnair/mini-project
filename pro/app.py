import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__, static_url_path="", static_folder="static")
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    height = float(request.form['height'])
    Cohesion = float(request.form['Cohesion'])
    angleoffriction = float(request.form['angleoffriction'])
    porewaterratio = float(request.form['porewaterratio'])

    prediction = model.predict( [[height,Cohesion,angleoffriction,porewaterratio]] )

    if prediction[0] == 1:
        pred = "Slope is not stable, take necessary actions"
    else:
        pred = "Slope is Stable."

    return render_template('index.html', prediction_text = pred)

if __name__ == "__main__":
    app.run(debug=True)
