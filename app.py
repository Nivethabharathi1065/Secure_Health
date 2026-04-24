from flask import Flask, render_template, request, jsonify, redirect, url_for, session, flash, send_file, abort
import numpy as np
import pickle
from pathlib import Path
from concrete.ml.deployment import FHEModelClient, FHEModelServer
import time
import threading
import logging
from werkzeug.security import generate_password_hash, check_password_hash
import json
from datetime import datetime
import secrets

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)  # Secure session key

# Reduce noisy logs
log = logging.getLogger('werkzeug')
log.setLevel(logging.WARNING)

# ── Config ──────────────────────────────────────────────────────
SAVE_FOLDER = Path("saved_model")
CLIENT_ZIP = SAVE_FOLDER / "client.zip"
SERVER_ZIP = SAVE_FOLDER / "server.zip"
SYMPTOMS_PK = SAVE_FOLDER / "symptoms_list.pkl"
METRICS_JSON = SAVE_FOLDER / "metrics.json"
USERS_DB = Path("users.json")
PREDICTIONS_LOG = Path("predictions_log.json")

KEY_SYMPTOMS = [
    "Heavy / Extreme menstrual bleeding",
    "Menstrual pain (Dysmenorrhea)",
    "Pelvic pain",
    "Cramping",
    "Abdominal pain / pressure",
    "Back pain",
    "Lower back pain",
    "Bloating",
    "Fatigue / Chronic fatigue",
    "Nausea",
    "Headaches",
    "Irregular / Missed periods",
    "Depression",
    "Anxiety",
    "Painful bowel movements"
]

# ── Global State ────────────────────────────────────────────────
model_state = {
    'loading': True,
    'ready': False,
    'error': None,
    'progress': 0,
    'message': 'Initializing...',
    'client': None,
    'server': None,
    'all_symptoms': None,
    'symptom_to_idx': None,
    'load_time': 0,
    'metrics': None,
}

# ── User Management ─────────────────────────────────────────────
def load_users():
    if USERS_DB.exists():
        with open(USERS_DB, 'r') as f:
            return json.load(f)
    return {}

def save_users(users):
    with open(USERS_DB, 'w') as f:
        json.dump(users, f, indent=2)

def log_prediction(username, symptoms_count, risk, probability, encrypted_time):
    """Log prediction for analysis"""
    log_entry = {
        'timestamp': datetime.now().isoformat(),
        'username': username,
        'symptoms_count': symptoms_count,
        'risk_level': risk,
        'probability': probability,
        'encrypted_inference_time': encrypted_time
    }
    
    logs = []
    if PREDICTIONS_LOG.exists():
        with open(PREDICTIONS_LOG, 'r') as f:
            logs = json.load(f)
    
    logs.append(log_entry)
    
    with open(PREDICTIONS_LOG, 'w') as f:
        json.dump(logs, f, indent=2)

# ── Background Model Loading ────────────────────────────────────
def load_fhe_model_background():
    """Load FHE model in background with progress updates"""
    global model_state
    
    try:
        # Step 1: Check files
        model_state['progress'] = 10
        model_state['message'] = 'Checking model files...'
        time.sleep(0.1)
        
        if not all(f.exists() for f in [CLIENT_ZIP, SERVER_ZIP, SYMPTOMS_PK]):
            raise FileNotFoundError("Missing model files. Run train_once.py first!")
        
        # Step 2: Load client
        model_state['progress'] = 25
        model_state['message'] = 'Loading FHE client...'
        start = time.time()
        model_state['client'] = FHEModelClient(
            path_dir=str(SAVE_FOLDER), 
            key_dir=str(SAVE_FOLDER)
        )
        
        # Step 3: Load server
        model_state['progress'] = 45
        model_state['message'] = 'Loading FHE server...'
        model_state['server'] = FHEModelServer(path_dir=str(SAVE_FOLDER))
        
        # Step 4: Load symptoms
        model_state['progress'] = 60
        model_state['message'] = 'Loading symptom dictionary...'
        with open(SYMPTOMS_PK, "rb") as f:
            model_state['all_symptoms'] = pickle.load(f)
        
        model_state['symptom_to_idx'] = {
            sym: model_state['all_symptoms'].index(sym) 
            for sym in KEY_SYMPTOMS 
            if sym in model_state['all_symptoms']
        }

        # Optional: load precomputed training / accuracy metrics
        if METRICS_JSON.exists():
            with open(METRICS_JSON, "r") as f:
                model_state['metrics'] = json.load(f)
        
        # Step 5: Warm-up prediction
        model_state['progress'] = 75
        model_state['message'] = 'Running warm-up prediction (this takes time)...'
        
        dummy_vec = np.zeros((1, len(model_state['all_symptoms'])), dtype=np.float32)
        if len(model_state['symptom_to_idx']) >= 3:
            dummy_vec[0, list(model_state['symptom_to_idx'].values())[:3]] = 1.0
        
        ser_input = model_state['client'].quantize_encrypt_serialize(dummy_vec)
        ser_keys = model_state['client'].get_serialized_evaluation_keys()
        _ = model_state['server'].run(ser_input, ser_keys)
        
        # Done!
        model_state['progress'] = 100
        model_state['message'] = 'Model ready!'
        model_state['load_time'] = time.time() - start
        model_state['ready'] = True
        model_state['loading'] = False
        
        print(f"✓ FHE Model loaded successfully in {model_state['load_time']:.1f}s")
        
    except Exception as e:
        model_state['error'] = str(e)
        model_state['loading'] = False
        model_state['ready'] = False
        print(f"✗ Model loading failed: {e}")

# Start loading immediately
threading.Thread(target=load_fhe_model_background, daemon=True).start()

# ── Routes ──────────────────────────────────────────────────────

@app.route('/')
def home():
    """Landing page - always loads instantly"""
    if 'username' in session:
        return redirect(url_for('dashboard'))
    return render_template('landing.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    """User registration with bcrypt hashing"""
    if request.method == 'GET':
        return render_template('register.html')
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    if not username or not password:
        flash('Username and password required', 'error')
        return render_template('register.html')
    
    users = load_users()
    
    if username in users:
        flash('Username already exists', 'error')
        return render_template('register.html')
    
    # Hash password with bcrypt (via werkzeug)
    users[username] = {
        'password_hash': generate_password_hash(password, method='pbkdf2:sha256'),
        'created_at': datetime.now().isoformat()
    }
    
    save_users(users)
    flash('Registration successful! Please login.', 'success')
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User login with hash verification"""
    if request.method == 'GET':
        return render_template('login.html')
    
    username = request.form.get('username', '').strip()
    password = request.form.get('password', '')
    
    users = load_users()
    
    if username not in users:
        flash('Invalid credentials', 'error')
        return render_template('login.html')
    
    # Verify password hash
    if not check_password_hash(users[username]['password_hash'], password):
        flash('Invalid credentials', 'error')
        return render_template('login.html')
    
    session['username'] = username
    flash(f'Welcome back, {username}!', 'success')
    return redirect(url_for('dashboard'))

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('username', None)
    flash('Logged out successfully', 'info')
    return redirect(url_for('home'))

@app.route('/dashboard')
def dashboard():
    """Main dashboard - requires authentication"""
    if 'username' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    return render_template('dashboard.html', 
                          username=session['username'],
                          model_ready=model_state['ready'])

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    """Prediction page with FHE"""
    if 'username' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if not model_state['ready']:
        return render_template('model_loading.html', 
                              progress=model_state['progress'],
                              message=model_state['message'],
                              error=model_state['error'])
    
    if request.method == 'GET':
        return render_template('predict.html', 
                              symptoms=KEY_SYMPTOMS,
                              username=session['username'])
    
    # POST - Run prediction
    vector = np.zeros((1, len(model_state['all_symptoms'])), dtype=np.float32)
    selected_symptoms = []
    
    for sym in KEY_SYMPTOMS:
        if sym not in model_state['symptom_to_idx']:
            continue
        if request.form.get(sym) == 'on':
            vector[0, model_state['symptom_to_idx'][sym]] = 1.0
            selected_symptoms.append(sym)
    
    symptoms_count = len(selected_symptoms)
    
    if symptoms_count == 0:
        result = {
            'risk': 'LOW RISK',
            'percentage': 0.0,
            'raw_probability': 0.0,
            'encrypted_time': 0.0,
            'symptoms_count': 0
        }
    else:
        # Run encrypted prediction
        t_start = time.time()
        
        ser_input = model_state['client'].quantize_encrypt_serialize(vector)
        ser_keys = model_state['client'].get_serialized_evaluation_keys()
        enc_out = model_state['server'].run(ser_input, ser_keys)
        dec = model_state['client'].deserialize_decrypt_dequantize(enc_out)
        
        proba = float(dec[0, 1])
        encrypted_time = time.time() - t_start
        
        risk = "HIGH RISK" if proba >= 0.5 else "LOW RISK"
        
        result = {
            'risk': risk,
            'percentage': round(proba * 100, 1),
            'raw_probability': round(proba, 4),
            'encrypted_time': round(encrypted_time, 2),
            'symptoms_count': symptoms_count
        }
        
        # Log for analysis
        log_prediction(session['username'], symptoms_count, risk, proba, encrypted_time)
    
    model_plots_available = {
        'confusion_matrix': (SAVE_FOLDER / 'confusion_matrix.png').exists(),
        'roc_curve': (SAVE_FOLDER / 'roc_curve.png').exists(),
        'precision_recall_curve': (SAVE_FOLDER / 'precision_recall_curve.png').exists(),
    }
    return render_template(
        'result.html',
        result=result,
        selected_symptoms=selected_symptoms,
        username=session['username'],
        model_metrics=model_state.get('metrics'),
        model_plots_available=model_plots_available,
    )

@app.route('/analysis')
def analysis():
    """Analysis dashboard showing metrics"""
    if 'username' not in session:
        flash('Please login first', 'warning')
        return redirect(url_for('login'))
    
    if not PREDICTIONS_LOG.exists():
        return render_template('analysis.html', 
                              username=session['username'],
                              no_data=True)
    
    with open(PREDICTIONS_LOG, 'r') as f:
        logs = json.load(f)
    
    # Calculate metrics
    if not logs:
        return render_template('analysis.html', 
                              username=session['username'],
                              no_data=True)
    
    total_predictions = len(logs)
    avg_time = sum(l['encrypted_inference_time'] for l in logs) / total_predictions
    high_risk_count = sum(1 for l in logs if l['risk_level'] == 'HIGH RISK')
    avg_probability = sum(l['probability'] for l in logs) / total_predictions
    
    metrics = {
        'total_predictions': total_predictions,
        'avg_encrypted_time': round(avg_time, 2),
        'high_risk_percentage': round(high_risk_count / total_predictions * 100, 1),
        'avg_probability': round(avg_probability, 4),
        'model_load_time': round(model_state['load_time'], 2),
        'recent_logs': logs[-10:]  # Last 10 predictions
    }
    
    return render_template('analysis.html',
                          username=session['username'],
                          metrics=metrics,
                          no_data=False)

@app.route('/api/model_status')
def model_status():
    """API endpoint for checking model loading status"""
    return jsonify({
        'loading': model_state['loading'],
        'ready': model_state['ready'],
        'progress': model_state['progress'],
        'message': model_state['message'],
        'error': model_state['error']
    })

# Serve model plot images (confusion matrix, ROC, PR curve) - same UI source as result page
MODEL_PLOT_NAMES = ['confusion_matrix.png', 'roc_curve.png', 'precision_recall_curve.png']

@app.route('/model_plots/<filename>')
def model_plots(filename):
    if filename not in MODEL_PLOT_NAMES:
        abort(404)
    path = SAVE_FOLDER / filename
    if not path.exists():
        abort(404)
    return send_file(path, mimetype='image/png')

if __name__ == "__main__":
    print("\n" + "="*70)
    print("   🔐 Privacy-Preserving Health Assessment System")
    print("   Using Fully Homomorphic Encryption (FHE)")
    print("="*70)
    print("\n📦 FHE Model loading in background...")
    print("⚡ Web server starting immediately (no wait!)")
    print("\n🌐 Open: http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(debug=False, host="0.0.0.0", port=5000, threaded=True)