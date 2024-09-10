
from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
from ml_cross import run_model_on_data, external_data_prediction

app = Flask(__name__)

# Global variable to store the initial results
initial_results = {
    'strategy_accuracy': None,
    'model_accuracy': None
}

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            # Process the file
            df = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            base_accuracy, model_accuracy = run_model_on_data(df)
            
            # Update the global results
            initial_results['strategy_accuracy'] = base_accuracy
            initial_results['model_accuracy'] = model_accuracy
            
            return redirect(url_for('results'))
    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    global initial_results
    
    external_data_accuracy = None
    
    if request.method == 'POST':
        file = request.files['file']
        if file and (file.filename.endswith('.csv') or file.filename.endswith('.xlsx')):
            # Process the external file
            test = pd.read_csv(file) if file.filename.endswith('.csv') else pd.read_excel(file)
            external_data_accuracy = external_data_prediction(test)
    
    return render_template('results.html',
                           strategy_accuracy=initial_results['strategy_accuracy'],
                           model_accuracy=initial_results['model_accuracy'],
                           external_data_prediction_accuracy=external_data_accuracy)

if __name__ == '__main__':
    app.run(debug=True)







    