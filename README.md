# NeuroDetect AI - Early Parkinson's Detection System

A multi-modal AI system designed to detect early signs of Parkinson's disease through computerized analysis of motor skills (spiral drawings) and vocal biomarkers.

![NeuroDetect Dashboard](src/assets/dashboard_preview.png)

## 🧠 Machine Learning Architecture

The system utilizes two specialized deep learning models to analyze different modalities:

### 1. Motor Analysis Model (Spiral Test)
Analyzing hand-drawn spiral/wave patterns to detect micro-tremors and geometric irregularities.
- **Architecture**: **ResNet-50** (Residual Network with 50 layers).
- **Technique**: Transfer Learning (pretrained on ImageNet, fine-tuned on Parkinson's spiral dataset).
- **Input**: 224x224 grayscale images of hand-drawn spirals.
- **Output**: Binary classification (Healthy vs. Parkinson's) + Confidence Score.
- **Training Strategy**: Heavy data augmentation (rotation, flips, erasing), AdamW optimizer, Cosine Annealing scheduler.

### 2. Vocal Biomarker Model (Voice Test)
Analyzing voice recordings to detect frequency perturbations (jitter, shimmer) and non-linear dynamic features.
- **Architecture**: **Custom Deep Neural Network (MLP)**.
  - Layer 1: Linear(22 -> 256) + BatchNorm + LeakyReLU + Dropout
  - Layer 2: Linear(256 -> 128) + BatchNorm + LeakyReLU + Dropout
  - Layer 3: Linear(128 -> 64) + BatchNorm + LeakyReLU
  - Layer 4: Linear(64 -> 32) + BatchNorm + LeakyReLU
  - Output: Linear(32 -> 2)
- **Features**: Extracts 22 acoustic features using `librosa`, including:
  - **Fundamental Frequency**: Fo, Fhi, Flo
  - **Jitter**: Variations in pitch (MDVP:Jitter, RAP, PPQ)
  - **Shimmer**: Variations in loudness (MDVP:Shimmer, APQ3, APQ5)
  - **Harmonicity**: HNR (Harmonics-to-Noise Ratio)
  - **Non-linear Dynamics**: RPDE (Recurrence Period Density Entropy), DFA (Detrended Fluctuation Analysis), PPE.
- **Training Strategy**: 5-Fold Cross-Validation, Weighted Cross-Entropy Loss (handling class imbalance), Automatic Custom Dataset Building.

---

## 🔗 System Integration

The project follows a modern decoupled architecture:

### Frontend (React + Vite)
- **UI Framework**: React 18 with Tailwind CSS.
- **Design System**: Custom "Aurora" theme using glassmorphism components.
- **Role**: Handles user interaction, file uploads, and visualizing results.
- **Communication**: Sends `multipart/form-data` requests to the Backend via REST API.

### Backend (FastAPI + PyTorch)
- **Framework**: FastAPI (high-performance Python web framework).
- **Inference Engine**: PyTorch.
- **Workflow**:
  1. Receives image/audio file from Frontend.
  2. Preprocesses data (Image resizing / Audio feature extraction).
  3. Passes tensor to the loaded PyTorch model (`.pth` files).
  4. Returns prediction and confidence score as JSON.

---

## 🚀 Training on Custom Data

We implemented an **Automatic Self-Optimizing Pipeline** that allows you to train the models on your own raw data without writing code.

### Voice Model Training
1. **Drop your files**:
   - Healthy samples -> `backend/datasets/voice/raw/healthy/`
   - Parkinson samples -> `backend/datasets/voice/raw/parkinson/`
2. **Train**:
   ```bash
   cd backend
   python train_voice.py
   ```
   - The script automatically **detects raw .wav files**.
   - **Extracts acoustic features** (including complex ones like RPDE/DFA).
   - **Builds a dataset** (`custom_data.csv`).
   - **Trains a new model** using 5-fold cross-validation.
   - **Saves** the best model to `backend/tensors/voice_model_tabular.pth`.

---

## 🛠️ Setup & Installation

### Backend
```bash
cd backend
python -m venv myenv
.\myenv\Scripts\activate
pip install -r requirements.txt
python run_server.py
# Server runs at http://localhost:8000
```

### Frontend
```bash
npm install
npm run dev
# App runs at http://localhost:5173
```
