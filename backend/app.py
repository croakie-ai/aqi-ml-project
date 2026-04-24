from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib, numpy as np, json, base64, os

app = Flask(__name__)
CORS(app)

# ── Load models ──────────────────────────────────────────
BASE = os.path.dirname(__file__)

ridge   = joblib.load(os.path.join(BASE, 'models/ridge_tuned.pkl'))
lasso   = joblib.load(os.path.join(BASE, 'models/lasso_tuned.pkl'))
rf      = joblib.load(os.path.join(BASE, 'models/random_forest.pkl'))
svm     = joblib.load(os.path.join(BASE, 'models/svm_model.pkl'))
kmeans  = joblib.load(os.path.join(BASE, 'models/kmeans_model.pkl'))
dbscan  = joblib.load(os.path.join(BASE, 'models/dbscan_model.pkl'))
le      = joblib.load(os.path.join(BASE, 'models/label_encoder.pkl'))
scaler  = joblib.load(os.path.join(BASE, 'models/scaler.pkl'))

with open(os.path.join(BASE, 'output/results_summary.json')) as f:
    results = json.load(f)

# ── Feature builder ───────────────────────────────────────
CITIES = ['Ahmedabad', 'Dehradun', 'Delhi', 'Jaipur', 'Lucknow']

def build_features(data):
    pm25 = float(data.get('pm25', 0))
    pm10 = float(data.get('pm10', 0))
    no2  = float(data.get('no2',  0))
    so2  = float(data.get('so2',  0))
    co   = float(data.get('co',   0))
    oz   = float(data.get('ozone',0))
    city = data.get('city', 'Delhi')

    # core + humidity + wind defaults
    row = [pm25, pm10, no2, so2, co, oz, 60.0, 1.5]
    # lag features (3 lags × 7 cols)
    for _ in range(3):
        row += [pm25, pm10, no2, so2, co, oz, 0.0]
    # rolling features (6 windows × 6 cols × 2 stats)
    for _ in range(6):
        row += [pm25, pm10, no2, so2, co, oz,
                pm25, pm10, no2, so2, co, oz]
    # time features
    row += [12.0, 3.0, 6.0, 0.0, 1.0, 0.5, 0.87, 0.5, 0.87]
    # ratio
    row += [pm25 / (pm10 + 1e-6)]
    # city one-hot
    row += [1 if c == city else 0 for c in CITIES]

    arr = np.array(row[:80], dtype=float).reshape(1, -1)
    return scaler.transform(arr)

# ── Endpoints ─────────────────────────────────────────────

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'models_loaded': True})

@app.route('/predict/aqi', methods=['POST'])
def predict_aqi():
    data = request.json
    X = build_features(data)
    aqi_lasso = float(lasso.predict(X)[0])
    aqi_ridge = float(ridge.predict(X)[0])
    return jsonify({
        'aqi_value':  round(aqi_lasso, 2),
        'aqi_ridge':  round(aqi_ridge, 2),
        'lasso_r2':   results['Lasso_R2'],
        'ridge_r2':   results['Ridge_R2'],
    })

@app.route('/classify/category', methods=['POST'])
def classify_category():
    data = request.json
    X = build_features(data)
    pred_enc = rf.predict(X)[0]
    proba    = rf.predict_proba(X)[0]
    category = le.inverse_transform([pred_enc])[0]
    conf     = float(max(proba))
    proba_dict = {le.classes_[i]: round(float(p), 4) for i, p in enumerate(proba)}
    return jsonify({
        'category':    category,
        'confidence':  round(conf, 4),
        'probabilities': proba_dict
    })

@app.route('/detect/anomaly', methods=['POST'])
def detect_anomaly():
    data = request.json
    X = np.array([[
        float(data.get('pm25',  0)),
        float(data.get('pm10',  0)),
        float(data.get('no2',   0)),
        float(data.get('so2',   0)),
        float(data.get('co',    0)),
        float(data.get('ozone', 0))
    ]])
    label = dbscan.fit_predict(X)
    return jsonify({
        'is_anomaly': bool(label[0] == -1),
        'label':      int(label[0])
    })

@app.route('/insights/metrics', methods=['GET'])
def get_metrics():
    return jsonify(results)

@app.route('/insights/plots/<name>', methods=['GET'])
def get_plot(name):
    path = os.path.join(BASE, f'output/plots/{name}.png')
    if not os.path.exists(path):
        return jsonify({'error': 'plot not found'}), 404
    with open(path, 'rb') as f:
        img_b64 = base64.b64encode(f.read()).decode()
    return jsonify({'image': img_b64, 'name': name})

@app.route('/insights/rankings', methods=['GET'])
def get_rankings():
    rankings = [
        {'rank':1,  'station':'DL002', 'city':'Delhi',     'risk_score':0.774, 'cluster':'High Pollution'},
        {'rank':2,  'station':'UP001', 'city':'Lucknow',   'risk_score':0.752, 'cluster':'High Pollution'},
        {'rank':3,  'station':'DL004', 'city':'Delhi',     'risk_score':0.740, 'cluster':'High Pollution'},
        {'rank':4,  'station':'DL005', 'city':'Delhi',     'risk_score':0.702, 'cluster':'High Pollution'},
        {'rank':5,  'station':'UP006', 'city':'Lucknow',   'risk_score':0.666, 'cluster':'High Pollution'},
        {'rank':6,  'station':'GJ001', 'city':'Ahmedabad', 'risk_score':0.477, 'cluster':'Medium Pollution'},
        {'rank':7,  'station':'UP002', 'city':'Lucknow',   'risk_score':0.476, 'cluster':'Medium Pollution'},
        {'rank':8,  'station':'RJ001', 'city':'Jaipur',    'risk_score':0.469, 'cluster':'Medium Pollution'},
        {'rank':9,  'station':'RJ002', 'city':'Jaipur',    'risk_score':0.274, 'cluster':'Medium Pollution'},
        {'rank':10, 'station':'UK002', 'city':'Dehradun',  'risk_score':0.174, 'cluster':'Low Pollution'},
        {'rank':11, 'station':'GJ010', 'city':'Ahmedabad', 'risk_score':0.094, 'cluster':'Low Pollution'},
        {'rank':12, 'station':'UK001', 'city':'Dehradun',  'risk_score':0.039, 'cluster':'Low Pollution'},
    ]
    return jsonify(rankings)

if __name__ == '__main__':
    app.run(debug=True, port=5000)