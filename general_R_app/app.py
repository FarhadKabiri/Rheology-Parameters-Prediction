import os
import io
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, request, Response
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostRegressor

app = Flask(__name__)

FEATURES = ['C', 'W', 'S', 'G', 'SF', 'FA', 'LS', 'SP', 'TM']

# Load models
pkg_pv = joblib.load(os.path.join('models', 'stacking_ensemble_PV.pkl'))
pkg_dys = joblib.load(os.path.join('models', 'stacking_ensemble_DYS.pkl'))

def get_ensemble_prediction(package, df_input):
    scaler_X = package["scaler_X"]
    scaler_y = package["scaler_y"]
    base_models = package["base_models"]
    meta_learner = package["meta_learner"]
    
    X_scaled = scaler_X.transform(df_input.values)
    
    # Get individual base model predictions
    meta_features = []
    for model in base_models:
        pred_scaled = model.predict(X_scaled)
        pred = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        meta_features.append(pred)
    
    meta_features = np.array(meta_features)
    
    # Meta-learner prediction
    final_preds = meta_learner.predict(meta_features.T)
    
    return final_preds[0]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_data = {f: [float(request.form[f])] for f in FEATURES}
        input_df = pd.DataFrame(input_data)
        
        pv_val = get_ensemble_prediction(pkg_pv, input_df)
        dys_val = get_ensemble_prediction(pkg_dys, input_df)
        
        results = {
            'pv': round(float(pv_val), 3),
            'dys': round(float(dys_val), 3),
            'inputs': {f: request.form[f] for f in FEATURES}
        }
        return render_template('index.html', single_results=results)
    except Exception as e:
        return render_template('index.html', error=str(e))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)