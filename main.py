from fastapi import FastAPI, HTTPException, Depends, File, UploadFile
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import librosa
import os
import logging
from pred3 import analyze_combined_audio_features, assess_health

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model and vectorizer
model_path = os.path.join(os.getcwd(), "models/emotion_model.pkl")
vectorizer_path = os.path.join(os.getcwd(), "models/tfidf_vectorizer.pkl")
with open(model_path, "rb") as model_file:
    model = pickle.load(model_file)
with open(vectorizer_path, "rb") as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# FastAPI instance
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Hello, Vercel!"}

# Vercel's entry point for the application
def handler(event, context):
    from mangum import Mangum
    asgi_handler = Mangum(app)
    return asgi_handler(event, context)

# API key validation
API_KEY = "mentalhealth_2024"
api_key_header = APIKeyHeader(name="api_key", auto_error=False)

def validate_api_key(api_key: str = Depends(api_key_header)):
    if api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return True

# Request model
class TextInput(BaseModel):
    text: str

# Routes
@app.get("/")
def read_root():
    return {"message": "Welcome to the Emotion Detection API"}

@app.post("/predict")
def predict_emotion(input: TextInput, valid: bool = Depends(validate_api_key)):
    try:
        text = input.text
        text_vectorized = vectorizer.transform([text])
        prediction = model.predict(text_vectorized)

        # Ensure the prediction is a regular Python int, not numpy.int64
        emotion = int(prediction[0]) if isinstance(prediction[0], np.int64) else prediction[0]

        return {"emotion": emotion}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")



        
# @app.post("/predict/audio")
# async def predict_audio(file: UploadFile = File(...), valid: bool = Depends(validate_api_key)):
#     try:
#         # Save and process audio file
#         temp_file = "temp_audio.wav"
#         with open(temp_file, "wb") as f:
#             f.write(await file.read())
        
#         y, sr = librosa.load(temp_file, sr=None)
#         pitch = librosa.feature.mfcc(y=y, sr=sr)
        
#         # Placeholder for feature-to-text conversion (you need to replace this with actual logic)
#         text = "extracted features from audio"  # Placeholder for feature extraction
        
#         # Vectorizing the text
#         text_vectorized = vectorizer.transform([text])
        
#         # Get prediction
#         prediction = model.predict(text_vectorized)

#         # Ensure the prediction is a regular Python int, not numpy.int64
#         emotion = int(prediction[0]) if isinstance(prediction[0], np.int64) else prediction[0]
        
#         return {"emotion": emotion}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
@app.post("/predict/audio")
async def predict_audio(file: UploadFile = File(...), valid: bool = Depends(validate_api_key)):
    try:
        # Save and process the audio file
        temp_file = "temp_audio.wav"
        with open(temp_file, "wb") as f:
            f.write(await file.read())
        
        # Analyze features from the audio file
        features = analyze_combined_audio_features(temp_file)
        
        # Assess the health based on the extracted features
        health_assessment = assess_health(features)
        
        return {"health_assessment": health_assessment, "features": features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
