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



UPLOAD_FOLDER = 'D:/sdhi/PROJECT DEVELOPMENT PHASE/SPRINT 3/UPLOADS'


app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER



@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == "POST":
        data = [[request.form.get('age'),request.form.get('tb'),request.form.get('ap'),request.form.get('aa'),request.form.get('asa')
                ,request.form.get('tp'),request.form.get('a'),request.form.get('agr'),request.form.get('a1'),request.form.get('a2')]]
  
        df = pd.DataFrame(data, columns=['sensor2', 'sensor3', 'sensor4', 'sensor7', 'sensor8', 'sensor9',
                                        'sensor15', 'sensor17', 'sensor20', 'sensor21'])
        scaler = MinMaxScaler()
        df1 = scaler.fit_transform(df)
        gh=joblib.load('SVC.pkl')
        num=gh.predict(df1)
        return render_template('predict.html', num=str(num[0]))


if __name__ == '__main__':
    app.run(debug=True, threaded=False)