import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
import shap
import joblib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from flask import Flask, request, jsonify, send_file
import os
import matplotlib
from flask_cors import CORS
matplotlib.use('Agg')


app = Flask(__name__)

# Initialize CORS
CORS(app)

app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def load_ml_model(file_name):
    try:
        if file_name.endswith('.sav') or file_name.endswith('.pkl'):
            try:
                with open(file_name, 'rb') as f:
                    model = pickle.load(f)
                print("Model loaded with pickle.")
            except pickle.UnpicklingError:
                model = joblib.load(file_name)
                print("Model loaded with joblib.")
        elif file_name.endswith('.joblib'):
            model = joblib.load(file_name)
            print("Model loaded with joblib.")
        elif file_name.endswith('.h5'):
            model = load_model(file_name)
            print("Model loaded with Keras.")
        else:
            raise ValueError("Unsupported file format.")
        return model
    except Exception as e:
        print(f"Failed to load model: {e}")
        return None
 
def evaluate_model(model, X_test, y_test):
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        cv_scores = cross_val_score(model, X_test, y_test, cv=5)
        cm = confusion_matrix(y_test, y_pred)
        if hasattr(model, "predict_proba"):
            y_pred_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_pred_prob = None
        return accuracy, cv_scores, f1, y_pred_prob, cm
    except Exception as e:
        print(f"Failed to evaluate model: {e}")
        return None, None, None, None, None
    
def plot_roc_curve(y_test, y_pred_prob, output_path):
    try:
        fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend(loc='lower right')
        plt.savefig(output_path)
        #plt.show()  
        plt.close()
    except Exception as e:
        print(f"Failed to plot ROC curve: {e}")
    
def model_interpretability(model, X, feature_names, output_path, background_data=None, n_background_samples=100):
    try:
        if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, XGBClassifier)):
            explainer = shap.TreeExplainer(model)
        elif isinstance(model, (LogisticRegression, SVC, KNeighborsClassifier, GaussianNB, DecisionTreeClassifier)):
            if background_data is None:
                background_data = X
            background_data = shap.sample(background_data, n_background_samples)
            if hasattr(model, "predict_proba"):
                explainer = shap.KernelExplainer(model.predict_proba, background_data)
            else:
                explainer = shap.KernelExplainer(model.predict, background_data)
        else:
            raise TypeError("Unsupported model type. Please provide a supported classifier.")

        shap_values = explainer.shap_values(X)

        shap.summary_plot(shap_values, X, feature_names=feature_names, plot_type='bar', show=False)
        plt.savefig(output_path, bbox_inches='tight')
        #plt.show()  
        plt.close()

    except Exception as e:
        print(f"Failed to generate SHAP summary plot: {e}")
        
@app.route('/evaluate', methods=['POST'])
def evaluate():
    if 'model_file' not in request.files or 'csv_file' not in request.files:
        return jsonify({"error": "Model file and CSV file are required."}), 400

    model_file = request.files['model_file']
    csv_file = request.files['csv_file']

    model_file_path = os.path.join(app.config['UPLOAD_FOLDER'], model_file.filename)
    csv_file_path = os.path.join(app.config['UPLOAD_FOLDER'], csv_file.filename)

    model_file.save(model_file_path)
    csv_file.save(csv_file_path)

    # Load the model and data
    model = load_ml_model(model_file_path)
    data = pd.read_csv(csv_file_path)  # only can read csv data file
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    if model is None:
        return jsonify({"error": "Failed to load model."}), 500

    # Evaluate the model
    accuracy, cv_scores, f1, y_pred_prob, cm = evaluate_model(model, X, y)

    # Plot ROC curve
    roc_curve_path = os.path.join(app.config['UPLOAD_FOLDER'], 'roc_curve.png')
    if y_pred_prob is not None:
        plot_roc_curve(y, y_pred_prob, roc_curve_path)

    # Generate SHAP summary plot
    shap_summary_plot_path = os.path.join(app.config['UPLOAD_FOLDER'], 'shap_summary_plot.png')
    model_interpretability(model, X, X.columns, shap_summary_plot_path)

    response = {
        "accuracy": accuracy,
        "cross_val_scores": cv_scores.tolist() if cv_scores is not None else None,
        "f1_score": f1,
        "roc_curve_url": f"/{roc_curve_path}",
        "shap_summary_plot_url": f"/{shap_summary_plot_path}"
    }

    return jsonify(response)
    
@app.route('/roc_curve.png')
def get_roc_curve():
    try:
        return send_file('roc_curve.png', mimetype='image/png')
    except Exception as e:
        print(f"Failed to send roc_curve.png: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/shap_summary_plot.png')
def get_shap_summary_plot():
    try:
        output_path = os.path.join(os.getcwd(), 'shap_summary_plot.png')
        return send_file(output_path, mimetype='image/png')
    except Exception as e:
        print(f"Failed to send shap_summary_plot.png: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/uploads/<filename>')
def get_uploaded_file(filename):
    try:
        return send_file(os.path.join(app.config['UPLOAD_FOLDER'], filename), mimetype='image/png')
    except Exception as e:
        print(f"Failed to send {filename}: {e}")
        return jsonify({"error": str(e)}), 500
    
if __name__ == '__main__':
    try:
        app.run(port=5001, debug=True) 
    except Exception as e:
        print(f"Failed to start Flask app: {e}")