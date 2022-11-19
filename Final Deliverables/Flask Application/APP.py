import requests
import numpy as np
import os
from PIL import Image
from flask import Flask, request, render_template, url_for
from werkzeug.utils import secure_filename, redirect
from gevent.pywsgi import WSGIServer
from flask import send_from_directory
from joblib import Parallel,delayed
import joblib
import pandas as pd
from scipy.sparse import issparse
from sklearn.preprocessing import MinMaxScaler



# UPLOAD_FOLDER = 'D:/sdhi/PROJECT DEVELOPMENT PHASE/SPRINT 3/UPLOADS'

app = Flask(__name__)
# app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
print("Flask application created")



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def upload():
    if request.method == "POST":
        data = [[request.form.get('age'),request.form.get('tb'),request.form.get('ap'),request.form.get('aa'),request.form.get('asa')
                ,request.form.get('tp'),request.form.get('a'),request.form.get('agr'),request.form.get('a1'),request.form.get('a2')]]
  
        df = pd.DataFrame(data, columns=['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor9',
                                        'sensor15', 'sensor17', 'sensor20', 'sensor21'])
        scaler = MinMaxScaler()
        df1 = scaler.fit_transform(df)
        
        # gh=joblib.load('SVC.pkl')
        # num=gh.predict(df1)
        # print(df1)

        # NOTE: you must manually set API_KEY below using information retrieved from your IBM Cloud account.
        API_KEY = "xU7wUCiLdUHS2Y9iDdLma2qNSXQyWBzohz-ZlQCOB1Ss"
        token_response = requests.post('https://iam.cloud.ibm.com/identity/token', data={"apikey":
        API_KEY, "grant_type": 'urn:ibm:params:oauth:grant-type:apikey'})
        mltoken = token_response.json()["access_token"]

        header = {'Content-Type': 'application/json', 'Authorization': 'Bearer ' + mltoken}

        # NOTE: manually define and pass the array(s) of values to be scored in the next line
        # payload_scoring = {"input_data": [{"field": [['sensor2','sensor3','sensor4','sensor7','sensor8','sensor9','sensor15','sensor17','sensor20','sensor21']], "values": [[0.310241, 0.304556, 0.386226, 0.618357, 0.257576, 0.208068, 0.495575, 0.416667, 0.651163, 0.442833]]}]}
        # payload_scoring = {"input_data": [{"field": [['sensor2','sensor3','sensor4','sensor7','sensor8','sensor9','sensor15','sensor17','sensor20','sensor21']], "values": [[0.183735,	0.406802,	0.309757,	0.726248,	0.242424,	0.109755,	0.363986,	0.333333	,0.713178,	0.724662]]}]}
        payload_scoring = {"input_data": [{"field": [['sensor2','sensor3','sensor4','sensor7','sensor8','sensor9','sensor15','sensor17','sensor20','sensor21']], "values": df1.tolist()}]}
        print("Sent API call to model stored on IBM Cloud")

        response_scoring = requests.post('https://us-south.ml.cloud.ibm.com/ml/v4/deployments/d8cc712a-d330-4983-98be-ba7c0a1b2dc9/predictions?version=2022-11-16', json=payload_scoring,
        headers={'Authorization': 'Bearer ' + mltoken})
        print("Scoring response")
        prediction = response_scoring.json()

        pred = prediction['predictions'][0]['values'][0][0]
        msg = "There is less chance of this engine failing in the immediate future" if pred == 0 else "There is a good chance that this engine will fail soon!"
        return render_template('predict.html', num=msg)


if __name__ == '__main__':
    app.run(debug=True, threaded=False)