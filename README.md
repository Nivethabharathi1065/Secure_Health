# 🔐 Privacy-Preserving Endometriosis Risk Detection
### Using Fully Homomorphic Encryption (FHE) + XGBoost + Flask
t

This project is a **privacy-preserving web application** that predicts endometriosis risk from self-reported symptoms — without the server ever seeing the user's actual data.

It uses **Fully Homomorphic Encryption (FHE)** via the **Concrete ML** library to run machine learning inference directly on **encrypted symptom data**. The server computes the risk score on locked (encrypted) data, so no private health information is ever exposed.

**Key facts:**
- Endometriosis affects ~10% of reproductive-age women globally
- Average diagnosis delay: **7–10 years**
- This tool enables early, anonymous, privacy-safe screening from home
- Model accuracy: **91.57%** (identical in both plaintext and encrypted modes)
- Encryption method: **TFHE (Fast Fully Homomorphic Encryption over the Torus)**

---

## How It Works

```
User selects symptoms (checkboxes)
        ↓
Symptoms encoded as binary vector (56 features)
        ↓
Vector encrypted CLIENT-SIDE (never leaves device as plaintext)
        ↓
Encrypted data sent to Flask server
        ↓
Server runs XGBoost prediction ON ENCRYPTED DATA (sees nothing)
        ↓
Encrypted result returned to client
        ↓
Client decrypts → shows HIGH RISK / LOW RISK + probability %
```

---

## Project Structure

```
DPSA_CONCRETE_ML/
│
├── app.py                  ← Flask web application (main server)
├── train.py                ← Model training + FHE compilation script
├── fhe_predict.py          ← CLI-based prediction (terminal mode)
├── dataset.xlsx            ← Endometriosis symptom dataset
├── predictions_log.json    ← Auto-generated prediction history
├── users.json              ← Auto-generated user credentials (hashed)
│
├── saved_model/            ← Auto-generated after training
│   ├── client.zip          ← FHE client keys & functions
│   ├── server.zip          ← FHE server evaluation circuit
│   ├── symptoms_list.pkl   ← List of all 56 symptoms
│   ├── metrics.json        ← Model accuracy & performance metrics
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── precision_recall_curve.png
│
├── templates/              ← HTML pages (Jinja2)
│   ├── landing.html
│   ├── login.html
│   ├── register.html
│   ├── dashboard.html
│   ├── predict.html
│   ├── result.html
│   ├── analysis.html
│   └── model_loading.html
│
└── static/
    └── style.css           ← Dark-theme stylesheet
```

---

## System Requirements

| Requirement | Details |
|---|---|
| OS | Windows 10/11 (via WSL2) or Linux/macOS |
| WSL Distro | Ubuntu 22.04 LTS (recommended) |
| Python | 3.10 or 3.11 |
| RAM | Minimum 8 GB (16 GB recommended for FHE compilation) |
| Disk | ~3 GB free space |
| Internet | Required for package installation |

> ⚠️ **Concrete ML does NOT run natively on Windows.** You must use WSL (Windows Subsystem for Linux).

---

## Step 1 – Install WSL on Windows

### 1a. Enable WSL (Run as Administrator in PowerShell)

Open **PowerShell as Administrator** and run:

```powershell
wsl --install
```

This installs WSL2 + Ubuntu automatically. If that doesn't work on your version:

```powershell
dism.exe /online /enable-feature /featurename:Microsoft-Windows-Subsystem-Linux /all /norestart
dism.exe /online /enable-feature /featurename:VirtualMachinePlatform /all /norestart
```

Then **restart your PC**.

### 1b. Set WSL 2 as Default

```powershell
wsl --set-default-version 2
```

### 1c. Install Ubuntu from Microsoft Store

1. Open the **Microsoft Store**
2. Search for **"Ubuntu 22.04 LTS"**
3. Click **Install**
4. After install, open Ubuntu from Start Menu
5. Create a **username** and **password** when prompted

### 1d. Verify WSL is Working

In PowerShell:
```powershell
wsl --list --verbose
```
You should see Ubuntu listed with **Version: 2**

### 1e. Update Ubuntu Packages

Inside the Ubuntu terminal:
```bash
sudo apt update && sudo apt upgrade -y
```

---

## Step 2 – Set Up Python Environment

### 2a. Install Python and pip

```bash
sudo apt install python3 python3-pip python3-venv -y
```

Verify:
```bash
python3 --version
# Should show Python 3.10.x or 3.11.x
```

### 2b. Create a Virtual Environment

```bash
python3 -m venv concrete_env
```

### 2c. Activate the Virtual Environment

```bash
source concrete_env/bin/activate
```

> You should see `(concrete_env)` at the start of your terminal prompt.
> 
> ⚠️ **You must activate this every time you open a new terminal.**

---

## Step 3 – Install Dependencies

With the virtual environment **activated**, install all required packages:

```bash
pip install --upgrade pip
pip install concrete-ml
pip install xgboost
pip install pandas
pip install numpy
pip install scikit-learn
pip install flask
pip install openpyxl
pip install matplotlib
pip install werkzeug
```

Or install everything in one command:

```bash
pip install concrete-ml xgboost pandas numpy scikit-learn flask openpyxl matplotlib werkzeug
```

> ⏳ `concrete-ml` is a large package (~500 MB). This may take **5–15 minutes** depending on your internet speed.

Verify installation:
```bash
python3 -c "import concrete; print('Concrete ML OK')"
python3 -c "import flask; print('Flask OK')"
```

---

## Step 4 – Add Project Files

### 4a. Navigate to Your Home Directory in WSL

```bash
cd ~
mkdir DPSA_CONCRETE_ML
cd DPSA_CONCRETE_ML
```

### 4b. Copy Files from Windows to WSL

Your Windows drives are accessible inside WSL at `/mnt/c/`, `/mnt/d/`, etc.

For example, if your files are in `C:\Users\YourName\Downloads\project\`:

```bash
cp /mnt/c/Users/YourName/Downloads/project/app.py .
cp /mnt/c/Users/YourName/Downloads/project/train.py .
cp /mnt/c/Users/YourName/Downloads/project/fhe_predict.py .
cp /mnt/c/Users/YourName/Downloads/project/dataset.xlsx .
cp -r /mnt/c/Users/YourName/Downloads/project/templates ./templates
cp -r /mnt/c/Users/YourName/Downloads/project/static ./static
```

> Replace `YourName` with your actual Windows username.

### 4c. Verify All Files Are Present

```bash
ls -la
```

You should see: `app.py`, `train.py`, `fhe_predict.py`, `dataset.xlsx`, `templates/`, `static/`

---

## Step 5 – Train the Model

> ⚠️ This step only needs to be done **once**. It creates the `saved_model/` folder with all FHE artifacts.

Make sure you are in the project directory with your virtual environment activated:

```bash
cd ~/DPSA_CONCRETE_ML
source ~/concrete_env/bin/activate
python3 train.py
```

### What train.py does:

| Step | Action | Time |
|---|---|---|
| 1 | Loads dataset.xlsx (all symptom rows) | ~1s |
| 2 | Builds 56-symptom binary feature matrix | ~2s |
| 3 | Trains XGBoost classifier | ~5s |
| 4 | Evaluates accuracy, precision, recall, F1 | ~1s |
| 5 | **Compiles model for FHE** (slowest step) | ~30–120s |
| 6 | Saves client.zip, server.zip, symptoms_list.pkl | ~2s |
| 7 | Generates confusion matrix, ROC, PR curve plots | ~3s |

### Expected output:
```
1. Loading dataset...
   Loaded 300 rows
   → 56 unique symptoms
2. Splitting data...
3. Training model...
   Training done in 4.2 seconds
3b. Evaluating plaintext accuracy...
   Plaintext accuracy: 91.57%
   Precision: 0.9200, Recall: 0.9100, F1: 0.9150
4. Compiling for FHE (this is the slow part – be patient)...
   Compilation finished in 87.3 seconds
5. Saving model artifacts...
   Saved metrics and plots to saved_model/
======================================================================
   TRAINING & COMPILATION COMPLETE!
   Total time: 98.1 seconds
======================================================================
```

---

## Step 6 – Run the Web App

```bash
cd ~/DPSA_CONCRETE_ML
source ~/concrete_env/bin/activate
python3 app.py
```

### Expected output:
```
======================================================================
   🔐 Privacy-Preserving Health Assessment System
   Using Fully Homomorphic Encryption (FHE)
======================================================================

📦 FHE Model loading in background...
⚡ Web server starting immediately (no wait!)

🌐 Open: http://localhost:5000
======================================================================
```

> The FHE model loads in the **background**. The web app is immediately accessible. You can open the browser right away — a loading screen will show progress until the model is ready (~30–60 seconds).

### Open in Browser

On Windows, open your browser and go to:
```
http://localhost:5000
```

> WSL2 automatically forwards `localhost` to Windows, so this works directly.

---

## Step 7 – Use the App

### 7a. Register an Account

1. Click **"Get Started"** or go to `http://localhost:5000/register`
2. Enter a username and password
3. Click **Register**

> Passwords are hashed with **PBKDF2-SHA256 + 600,000 iterations + random salt** — never stored in plaintext.

### 7b. Login

1. Go to `http://localhost:5000/login`
2. Enter your credentials
3. You will be redirected to the **Dashboard**

### 7c. Run a Risk Assessment

1. Click **"Endometriosis Risk Assessment"** from the dashboard
2. Wait for the model status to show **"Model Ready ✓"**
3. Check all symptoms you are currently experiencing (out of 15 listed)
4. Click **"Compute Risk (Encrypted)"**
5. Wait **8–12 seconds** for the encrypted inference to complete
6. View your result: **HIGH RISK** or **LOW RISK** with probability percentage

### 7d. View Analytics

Go to `http://localhost:5000/analysis` to see:
- Total predictions made
- Average encrypted inference time
- High risk percentage across all users
- Recent prediction history table

### 7e. CLI Mode (Optional)

For terminal-only predictions without the web app:

```bash
python3 fhe_predict.py
```

Answer `y` or `n` for each symptom, and get the result in the terminal.

## Security Features

### 1. Password Hashing (PBKDF2-SHA256)
- Algorithm: `PBKDF2-HMAC-SHA256`
- Iterations: **600,000 rounds**
- Salt: **128-bit random**, unique per user
- Output: **256-bit hash**
- Storage: `pbkdf2:sha256:600000$[salt]$[hash]` in `users.json`

### 2. Fully Homomorphic Encryption (TFHE)
- Based on the **Learning With Errors (LWE)** hard problem
- Resistant to classical and quantum attacks
- **Circuit privacy**: server learns nothing about inputs or predictions
- **Key separation**: secret key stays only with the client

### 3. Session Security
- Flask encrypted session cookies
- Random 32-byte secret key generated at startup
- All routes protected by authentication middleware

### 4. Attack Resistance
| Attack | Defense |
|---|---|
| Ciphertext-Only | Cannot decrypt without secret key |
| Known-Plaintext | Observing pairs doesn't help attacker |
| Server Compromise | Server only sees ciphertext — no patient data |
| Rainbow Table | Per-user random salt prevents precomputed attacks |
| Brute Force | 600,000 hash iterations make cracking extremely costly |


## Quick Start Summary

```bash
# 1. Open Ubuntu (WSL) terminal

# 2. Activate environment
source ~/concrete_env/bin/activate

# 3. Go to project folder
cd ~/DPSA_CONCRETE_ML

# 4. Train model (first time only, takes ~2 minutes)
python3 train.py

# 5. Start the web app
python3 app.py

# 6. Open browser → http://localhost:5000
# 7. Register → Login → Run Assessment
```

---

## Tech Stack

| Component | Technology |
|---|---|
| ML Model | XGBoost (via Concrete ML) |
| Privacy Layer | Fully Homomorphic Encryption (TFHE) |
| FHE Library | Concrete ML (by Zama) |
| Web Framework | Flask (Python) |
| Password Security | PBKDF2-SHA256 (Werkzeug) |
| Dataset | ESPDataset (56 symptom features, binary labels) |
| Frontend | Jinja2 HTML + CSS (dark theme) |
| Platform | WSL2 + Ubuntu 22.04 |

