import pickle
from flask import Flask, request, jsonify, render_template, app
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app=application

## import ridge regressor model and standard scaler pickle 
scaler= pickle.load(open('Model_Diabetic/LR_CV_Grid.pkl', 'rb'))
model= pickle.load(open('Model_Diabetic/standardScaler.pkl','rb'))

## route for home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata', methods=['Get', 'Post'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=float(request.form.get('Pregnancies'))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness=float(request.form.get('SkinThickness'))
        Insulin= float(request.form.get('Insulin'))
        BMI=float(request.form.get('BM!'))
        DiabetesPedigreeFunction=float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))
        
        new_data_scaled=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data_scaled)

        if predict[0]==1:
            result ='Diabetic'
        else:
            return render_template ('home.html')

        return render_template('single_prediction.html', result=result)

    else:
        return render_template('home.html')

if __name__=="__main__":
    app.run(host="0.0.0.0")
