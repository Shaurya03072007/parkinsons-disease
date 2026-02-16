from fastapi import FastAPI, UploadFile, File, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
import torch
import torch.nn as nn
from .models import ParkingonsImageModel
from .utils import get_device
from .auth import User, get_current_active_user, fake_users_db, fake_hash_password
from .feature_extraction import extract_features
import torchvision.transforms as transforms
from PIL import Image
import io
import os
import joblib
import numpy as np

app = FastAPI(title="Parkinson's Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = get_device()
print(f"Server starting on device: {device}")

# Resolve paths relative to this file (not CWD, since run_server.bat does cd ..)
_BACKEND_DIR = os.path.dirname(os.path.abspath(__file__))
_TENSORS_DIR = os.path.join(_BACKEND_DIR, 'tensors')

# --- Load Image Model ---
image_model = ParkingonsImageModel().to(device)
IMAGE_MODEL_PATH = os.path.join(_TENSORS_DIR, "spiral_model.pth")

if os.path.exists(IMAGE_MODEL_PATH):
    try:
        image_model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
        image_model.eval()
        print("Loaded image model.")
    except Exception as e:
        print(f"Error loading image model: {e}")
else:
    print("WARNING: Image model not found.")

# --- Load Voice Model (Tabular) ---
class VoiceClassifier(nn.Module):
    def __init__(self, input_dim=22): # 22 features
        super(VoiceClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(32, 2)
        )
        
    def forward(self, x):
        return self.network(x)

voice_model = VoiceClassifier().to(device)
VOICE_MODEL_PATH = os.path.join(_TENSORS_DIR, "voice_model_tabular.pth")
VOICE_SCALER_PATH = os.path.join(_TENSORS_DIR, "voice_scaler.save")
voice_scaler = None

if os.path.exists(VOICE_MODEL_PATH):
    try:
        voice_model.load_state_dict(torch.load(VOICE_MODEL_PATH, map_location=device))
        voice_model.eval()
        print("Loaded voice model.")
    except Exception as e:
        print(f"Error loading voice model: {e}")

if os.path.exists(VOICE_SCALER_PATH):
    voice_scaler = joblib.load(VOICE_SCALER_PATH)
    print("Loaded voice scaler.")

# Transforms
image_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

from .auth import User, get_current_active_user, fake_users_db, fake_hash_password, verify_google_token
from pydantic import BaseModel

class Token(BaseModel):
    token: str

@app.post("/auth/google")
async def google_auth(token_data: Token):
    user_info = await verify_google_token(token_data.token)
    if not user_info:
        raise HTTPException(status_code=400, detail="Invalid Google Token")
    
    # In a real app, create user in DB if not exists
    # For now, we return user info which frontend acknowledges
    return {"status": "success", "user": user_info}

@app.get("/")
def read_root():
    return {
        "message": "Parkinson's Detection API is running",
        "models_loaded": {
            "image": os.path.exists(IMAGE_MODEL_PATH),
            "voice": os.path.exists(VOICE_MODEL_PATH)
        }
    }

@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        image_tensor = image_transforms(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = image_model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            # Map index to class. Assuming 'healthy' is 0, 'parkinson' is 1
            result = "Parkinson's Detected" if predicted.item() == 1 else "Healthy"
            
        return {
            "prediction": result,
            "confidence": float(confidence.item())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/voice")
async def predict_voice(file: UploadFile = File(...)):
    if not voice_scaler or not os.path.exists(VOICE_MODEL_PATH):
         raise HTTPException(status_code=500, detail="Voice model or scaler not loaded.")

    try:
        # Save temp file
        temp_filename = f"temp_{file.filename}"
        with open(temp_filename, "wb") as buffer:
            buffer.write(await file.read())
            
        try:
            # Extract 22 features
            features = extract_features(temp_filename)
            
            # Scale features
            features_scaled = voice_scaler.transform([features])
            
            # Convert to tensor
            features_tensor = torch.tensor(features_scaled, dtype=torch.float32).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = voice_model(features_tensor)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probabilities, 1)
                
                result = "Parkinson's Detected" if predicted.item() == 1 else "Healthy"
                
            return {
                "prediction": result,
                "confidence": float(confidence.item())
            }
        finally:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
                
    except Exception as e:
        print(f"Voice prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
