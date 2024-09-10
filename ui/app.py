import pandas as pd
import numpy as np 
import ta 
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from ml_cross import base_accuracy, model_accuracy, ext_data_accuracy

# Testing all lib / module are correctly imported
print("All modules / libs imported successfuly")

# initialize the whole app as app
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            # Process the file
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            
            # Perform your calculations here
            # strategy_accuracy = 0.85  # Example value, replace with your actual calculation
            # model_accuracy = 0.92  # Example value, replace with your actual calculation
            # external_data_prediction_accuracy = 0.78  # Example value, replace with your actual calculation
            
            return redirect(url_for('results', 
                                    strategy_accuracy=base_accuracy,
                                    model_accuracy=model_accuracy,
                                    external_data_prediction_accuracy=ext_data_accuracy))
    return render_template('index.html')

@app.route('/results')
def results():
    strategy_accuracy = request.args.get('strategy_accuracy', 0)
    model_accuracy = request.args.get('model_accuracy', 0)
    external_data_prediction_accuracy = request.args.get('external_data_prediction_accuracy', 0)
    return render_template('results.html', 
                           strategy_accuracy=strategy_accuracy,
                           model_accuracy=model_accuracy,
                           external_data_prediction_accuracy=external_data_prediction_accuracy)

if __name__ == '__main__':
    app.run(debug=True)








    