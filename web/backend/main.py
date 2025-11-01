"""
FastAPI backend for NeuroVoice web interface.

Provides REST API for model inference, visualization, and analysis.
"""

import io
import base64
from pathlib import Path
from typing import Optional, List, Dict
import tempfile

import torch
import numpy as np
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import cv2
from PIL import Image

from src.config import MODELS_DIR
from src.models.fusion_model import FusionModel
from src.features.audio_features import extract_wav2vec_embeddings
from src.features.video_features import extract_facial_landmarks, extract_emotion_embedding

app = FastAPI(title="NeuroVoice API", version="1.0.0")

# CORS middleware for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model cache
MODEL_CACHE = {}


class PredictionRequest(BaseModel):
    disease: str = "alzheimer"
    model_path: Optional[str] = None


class PredictionResponse(BaseModel):
    prediction: float
    confidence: float
    disease: str
    status: str


def load_model(disease: str, model_path: Optional[str] = None) -> torch.nn.Module:
    """Load model from checkpoint."""
    cache_key = f"{disease}_{model_path}"
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]
    
    if model_path is None:
        model_path = MODELS_DIR / f"{disease}_best_model.pt"
    else:
        model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found at {model_path}")
    
    # Create model architecture
    model = FusionModel(
        audio_input_dim=768,
        video_input_dim=None,
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    MODEL_CACHE[cache_key] = model
    return model


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "NeuroVoice API",
        "version": "1.0.0",
        "endpoints": [
            "/predict",
            "/health",
            "/models",
            "/analyze",
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.get("/models")
async def list_models():
    """List available trained models."""
    models = {}
    for disease in ["alzheimer", "parkinson", "depression"]:
        model_path = MODELS_DIR / f"{disease}_best_model.pt"
        if model_path.exists():
            checkpoint = torch.load(model_path, map_location='cpu')
            models[disease] = {
                "path": str(model_path),
                "epoch": checkpoint.get('epoch', 'unknown'),
                "val_loss": checkpoint.get('val_loss', 'unknown'),
                "val_auc": checkpoint.get('val_auc', 'unknown'),
            }
    return {"models": models}


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    audio_file: UploadFile = File(...),
    video_file: Optional[UploadFile] = File(None),
    disease: str = "alzheimer",
    model_path: Optional[str] = None,
):
    """
    Predict disease from audio and optional video.
    
    Args:
        audio_file: Audio file (WAV format)
        video_file: Optional video file (MP4 format)
        disease: Disease type to predict
        model_path: Optional path to specific model
    
    Returns:
        Prediction result with confidence score
    """
    try:
        # Load model
        model = load_model(disease, model_path)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # Process audio
        audio_bytes = await audio_file.read()
        
        # Save to temporary file for processing
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name
        
        try:
            # Extract audio features
            audio_embeddings = extract_wav2vec_embeddings(tmp_audio_path)
            audio_tensor = torch.tensor(audio_embeddings).unsqueeze(0).to(device)
            
            # Process video if provided
            video_tensor = None
            if video_file:
                video_bytes = await video_file.read()
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                    tmp_video.write(video_bytes)
                    tmp_video_path = tmp_video.name
                
                try:
                    # Extract video features (landmarks)
                    landmarks = extract_facial_landmarks(tmp_video_path, max_faces=1)
                    if landmarks.size > 0:
                        # Average landmarks over frames
                        landmarks_avg = landmarks.mean(axis=0)
                        video_tensor = torch.tensor(landmarks_avg).unsqueeze(0).unsqueeze(0).to(device)
                finally:
                    Path(tmp_video_path).unlink()
            else:
                # Create dummy video tensor
                video_tensor = torch.zeros(1, 30, 468 * 3).to(device)
            
            # Get disease index
            disease_idx = torch.tensor([["alzheimer", "parkinson", "depression"].index(disease)]).to(device)
            
            # Make prediction
            with torch.no_grad():
                outputs = model(audio_tensor, video_tensor, disease_idx)
                
                if isinstance(outputs, dict):
                    logits = outputs['logits']
                else:
                    logits = outputs
                
                probs = torch.softmax(logits, dim=1)
                prediction_prob = probs[0, 1].item()  # Probability of disease
                prediction_class = int(prediction_prob > 0.5)
                confidence = abs(prediction_prob - 0.5) * 2  # Confidence in [0, 1]
            
            return PredictionResponse(
                prediction=prediction_class,
                confidence=float(confidence),
                disease=disease,
                status="success"
            )
        
        finally:
            Path(tmp_audio_path).unlink()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/analyze")
async def analyze_features(
    audio_file: UploadFile = File(...),
    video_file: Optional[UploadFile] = File(None),
):
    """
    Analyze audio and video features without making predictions.
    
    Returns feature statistics and visualizations.
    """
    try:
        results = {}
        
        # Process audio
        audio_bytes = await audio_file.read()
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as tmp_audio:
            tmp_audio.write(audio_bytes)
            tmp_audio_path = tmp_audio.name
        
        try:
            # Extract audio features
            audio_embeddings = extract_wav2vec_embeddings(tmp_audio_path)
            
            results['audio'] = {
                'embedding_dim': len(audio_embeddings),
                'mean': float(audio_embeddings.mean()),
                'std': float(audio_embeddings.std()),
                'min': float(audio_embeddings.min()),
                'max': float(audio_embeddings.max()),
            }
            
            # Process video if provided
            if video_file:
                video_bytes = await video_file.read()
                with tempfile.NamedTemporaryFile(suffix='.mp4', delete=False) as tmp_video:
                    tmp_video.write(video_bytes)
                    tmp_video_path = tmp_video.name
                
                try:
                    landmarks = extract_facial_landmarks(tmp_video_path, max_faces=1)
                    
                    results['video'] = {
                        'num_frames': landmarks.shape[0] if landmarks.size > 0 else 0,
                        'num_landmarks': landmarks.shape[1] // 3 if landmarks.size > 0 else 0,
                        'has_face': landmarks.size > 0,
                    }
                    
                    # Extract emotion if possible
                    try:
                        cap = cv2.VideoCapture(tmp_video_path)
                        ret, frame = cap.read()
                        cap.release()
                        
                        if ret:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            emotion_probs = extract_emotion_embedding(frame_rgb, return_logits=False)
                            
                            emotion_names = ["Angry", "Disgust", "Fear", "Happy", "Sad", "Surprise", "Neutral"]
                            dominant_emotion = emotion_names[np.argmax(emotion_probs)]
                            
                            results['video']['emotion'] = {
                                'dominant': dominant_emotion,
                                'probabilities': {name: float(prob) for name, prob in zip(emotion_names, emotion_probs)}
                            }
                    except Exception:
                        pass
                
                finally:
                    Path(tmp_video_path).unlink()
            
            return {"status": "success", "analysis": results}
        
        finally:
            Path(tmp_audio_path).unlink()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis error: {str(e)}")


@app.get("/stats/{disease}")
async def get_stats(disease: str):
    """Get statistics for a specific disease model."""
    model_path = MODELS_DIR / f"{disease}_best_model.pt"
    
    if not model_path.exists():
        raise HTTPException(status_code=404, detail=f"Model not found for {disease}")
    
    try:
        checkpoint = torch.load(model_path, map_location='cpu')
        return {
            "disease": disease,
            "epoch": checkpoint.get('epoch', 'unknown'),
            "metrics": {
                "val_loss": checkpoint.get('val_loss', 'unknown'),
                "val_auc": checkpoint.get('val_auc', 'unknown'),
                "train_metrics": checkpoint.get('train_metrics', {}),
                "val_metrics": checkpoint.get('val_metrics', {}),
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading stats: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

