from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from datetime import datetime
import json
import os
import io

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

DATA_FILE = 'data_records.json'
CSV_PATH = 'Health data.csv'

def load_dataset(path=CSV_PATH):
    with open(path, 'r', encoding='utf-8') as f:
        lines = [ln.rstrip('\n') for ln in f if ln.strip()]
    cleaned = []
    for ln in lines:
        if '|' in ln:
            parts = ln.split('|', 1)
            rest = parts[1]
        else:
            rest = ln
        cleaned.append(rest.strip())
    data = '\n'.join(cleaned)
    df = pd.read_csv(io.StringIO(data))
    cols = {c: c.strip() for c in df.columns}
    df.rename(columns=cols, inplace=True)
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if 'pulse' in lc:
            colmap[c] = 'Pulse'
        elif 'temp' in lc or 'temperature' in lc:
            colmap[c] = 'Temperature'
        elif 'spo2' in lc or 'spO2'.lower() in lc:
            colmap[c] = 'SPO2'
        elif 'status' in lc or 'label' in lc:
            colmap[c] = 'Label'
    df.rename(columns=colmap, inplace=True)
    if 'Label' not in df.columns:
        raise Exception('Dataset must contain a target column named Status or Label')
    features = []
    for f in ['Pulse','Temperature','SPO2']:
        if f in df.columns:
            features.append(f)
    if not features:
        raise Exception('No required feature columns found in dataset')
    df = df[features + ['Label']].dropna()
    df['Label'] = df['Label'].astype(int)
    return df, features

df_global, FEATURES = load_dataset()
X_global = df_global[FEATURES].values
y_global = df_global['Label'].values
scaler = StandardScaler().fit(X_global)
X_scaled = scaler.transform(X_global)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_global, test_size=0.2, random_state=42, stratify=y_global)
rf = RandomForestClassifier(n_estimators=50, max_depth=6, random_state=42)
svm = SVC(kernel='rbf', probability=True, C=1.0, random_state=42)
rf.fit(X_train, y_train)
svm.fit(X_train, y_train)

def metrics_model(model, X_t, y_t):
    y_pred = model.predict(X_t)
    acc = float(accuracy_score(y_t, y_pred))
    prec = float(precision_score(y_t, y_pred, average='weighted', zero_division=0))
    rec = float(recall_score(y_t, y_pred, average='weighted', zero_division=0))
    f1 = float(f1_score(y_t, y_pred, average='weighted', zero_division=0))
    cm = confusion_matrix(y_t, y_pred).tolist()
    return dict(accuracy=acc, precision=prec, recall=rec, f1=f1, confusion_matrix=cm)

rf_metrics = metrics_model(rf, X_test, y_test)
svm_metrics = metrics_model(svm, X_test, y_test)
rf_cv = float(np.mean(cross_val_score(rf, X_scaled, y_global, cv=3)))
svm_cv = float(np.mean(cross_val_score(svm, X_scaled, y_global, cv=3)))

def save_record(rec):
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, 'r') as f:
            data = json.load(f)
    else:
        data = []
    data.append(rec)
    with open(DATA_FILE, 'w') as f:
        json.dump(data, f)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/health')
def health():
    return jsonify(dict(status='ok'))

@app.route('/api/predict', methods=['POST'])
def predict():
    payload = request.json
    required = ['astronaut_id','astronaut_name','age','sex','pulse','temperature','spo2','timestamp']
    for k in required:
        if k not in payload:
            return jsonify(dict(error='missing '+k)),400
    try:
        pulse = float(payload['pulse'])
        temp = float(payload['temperature'])
        spo2 = float(payload['spo2'])
    except Exception:
        return jsonify(dict(error='invalid numeric values')),400
    features_vals = []
    for f in FEATURES:
        if f == 'Pulse':
            features_vals.append(pulse)
        elif f == 'Temperature':
            features_vals.append(temp)
        elif f == 'SPO2':
            features_vals.append(spo2)
    X_user = scaler.transform([features_vals])
    rf_pred = int(rf.predict(X_user)[0])
    svm_pred = int(svm.predict(X_user)[0])
    rf_prob = float(rf.predict_proba(X_user).max())
    svm_prob = float(svm.predict_proba(X_user).max())
    agreement = rf_pred == svm_pred
    overall = 'Unknown'
    next_step = 'Monitor'
    if rf_pred == 2 and svm_pred == 2:
        overall = 'High'
        next_step = 'Escalate'
    elif rf_pred == 1 or svm_pred == 1:
        overall = 'Moderate'
        next_step = 'Repeat measurement and log symptoms'
    else:
        overall = 'Low'
        next_step = 'Continue monitoring'
    record = dict(astronaut_id=payload['astronaut_id'],astronaut_name=payload['astronaut_name'],age=int(payload['age']),sex=payload['sex'],pulse=pulse,temperature=temp,spo2=spo2,timestamp=payload['timestamp'],rf_pred=rf_pred,svm_pred=svm_pred)
    save_record(record)
    out = dict(model_performance=dict(random_forest=dict(accuracy=rf_metrics['accuracy'],precision=rf_metrics['precision'],recall=rf_metrics['recall'],f1=rf_metrics['f1'],cv_score=rf_cv,confusion_matrix=rf_metrics['confusion_matrix']),svm=dict(accuracy=svm_metrics['accuracy'],precision=svm_metrics['precision'],recall=svm_metrics['recall'],f1=svm_metrics['f1'],cv_score=svm_cv,confusion_matrix=svm_metrics['confusion_matrix'])),connectivity_status=dict(browser_online=True,server='ok',local_storage_present=os.path.exists(DATA_FILE)),rule_based=dict(overall_risk=overall,recommended_next_step=next_step),ml_outputs=dict(random_forest_predicted_status=rf_pred,svm_predicted_status=svm_pred,model_agreement=agreement,random_forest_probability=rf_prob,svm_support_vectors=svm.support_vectors_.tolist()),timestamp=datetime.utcnow().isoformat())
    return jsonify(out)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
