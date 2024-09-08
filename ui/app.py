from flask import Flask, render_template, request, redirect, url_for, flash
import pandas as pd
import os
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = './uploads'
ALLOWED_EXTENSIONS = {'csv', 'xlsx'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Process the file
        df = pd.read_csv(file_path) if filename.endswith('.csv') else pd.read_excel(file_path)

        # Assuming the target variable is 'target', modify this based on your data.
        X = df.drop(columns=['target'])
        y = df['target']

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        clf = RandomForestClassifier()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        
        # Generate classification report
        report = classification_report(y_test, y_pred, output_dict=True)
        classification_report_json = json.dumps(report)
        
        # Prepare pie chart data (Example: Pie chart for class distribution in test set)
        pie_data = y_test.value_counts().to_dict()
        
        return render_template('results.html', report=classification_report_json, pie_data=pie_data)

if __name__ == '__main__':
    app.run(debug=True)
