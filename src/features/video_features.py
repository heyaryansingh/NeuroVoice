"""
Video feature extraction utilities.

Extracts facial landmarks, emotion embeddings, and other visual features.
"""

from pathlib import Path
from typing import List, Optional, Union

import cv2
import mediapipe as mp
import numpy as np
import torch


def extract_facial_landmarks(
    video_path: Union[str, Path],
    frame_indices: Optional[List[int]] = None,
    max_faces: int = 1,
) -> np.ndarray:
    """
    Extract facial landmarks from video using MediaPipe.
    
    Args:
        video_path: Path to video file
        frame_indices: Specific frame indices to extract (None = all frames)
        max_faces: Maximum number of faces to detect
    
    Returns:
        Landmarks array of shape (num_frames, num_landmarks * 3)
        Each landmark has x, y, z coordinates
    """
    # Initialize MediaPipe face mesh
    mp_face_mesh = mp.solutions.face_mesh
    face_mesh = mp_face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=max_faces,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    
    cap = cv2.VideoCapture(str(video_path))
    landmarks_list = []
    frame_idx = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Check if this frame should be processed
        if frame_indices is not None and frame_idx not in frame_indices:
            frame_idx += 1
            continue
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process frame
        results = face_mesh.process(frame_rgb)
        
        if results.multi_face_landmarks:
            # Get first face
            face_landmarks = results.multi_face_landmarks[0]
            
            # Extract x, y, z coordinates
            landmarks = []
            for landmark in face_landmarks.landmark:
                landmarks.extend([landmark.x, landmark.y, landmark.z])
            
            landmarks_list.append(np.array(landmarks))
        else:
            # No face detected, append zeros
            num_landmarks = 468  # MediaPipe face mesh has 468 landmarks
            landmarks_list.append(np.zeros(num_landmarks * 3))
        
        frame_idx += 1
    
    cap.release()
    
    if landmarks_list:
        return np.stack(landmarks_list)
    else:
        # Return zeros if no frames processed
        num_landmarks = 468
        return np.zeros((1, num_landmarks * 3))


def load_emotion_model(model_path: Optional[Path] = None, device: str = 'cpu') -> torch.nn.Module:
    """
    Load pre-trained emotion classification model.
    
    Args:
        model_path: Path to emotion model checkpoint (optional, uses default if None)
        device: Device to load model on
    
    Returns:
        Loaded emotion model
    """
    import torch
    import torch.nn as nn
    from torchvision import models
    from src.config import MODELS_DIR, NUM_EMOTIONS
    
    if model_path is None:
        model_path = MODELS_DIR / "emotion_model.pth"
    
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Emotion model not found at {model_path}. "
            "Please train the model first: python scripts/train_emotion_model.py"
        )
    
    # Create model architecture
    resnet = models.resnet18(pretrained=False)
    num_features = resnet.fc.in_features
    resnet.fc = nn.Linear(num_features, NUM_EMOTIONS)
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device)
    resnet.load_state_dict(checkpoint['model_state_dict'])
    resnet.to(device)
    resnet.eval()
    
    return resnet


def extract_emotion_embedding(
    frame: np.ndarray,
    model: Optional[torch.nn.Module] = None,
    model_path: Optional[Path] = None,
    device: str = 'cpu',
    return_logits: bool = True,
) -> np.ndarray:
    """
    Extract emotion embedding from a single frame.
    
    Args:
        frame: RGB image frame (numpy array, shape HxWx3, values 0-255)
        model: Pre-trained emotion classification model (optional, will load if None)
        model_path: Path to emotion model (used if model is None)
        device: Device to run model on
        return_logits: If True, return logits; if False, return probabilities
    
    Returns:
        Emotion embedding (logits or probabilities)
    """
    import torch
    from torchvision import transforms
    
    num_emotions = 7  # Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
    
    # Load model if not provided
    if model is None:
        try:
            model = load_emotion_model(model_path, device)
        except FileNotFoundError:
            # Return zeros if model not available
            return np.zeros(num_emotions)
    
    # Preprocess frame
    # Convert to PIL Image if needed
    if isinstance(frame, np.ndarray):
        # Normalize to 0-1 if in 0-255 range
        if frame.max() > 1.0:
            frame = frame.astype(np.float32) / 255.0
        
        # Convert RGB to tensor
        from PIL import Image
        if frame.shape[-1] == 3:
            # HWC format
            frame_pil = Image.fromarray((frame * 255).astype(np.uint8))
        else:
            frame_pil = Image.fromarray(frame)
    
    # Apply transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    frame_tensor = transform(frame_pil).unsqueeze(0).to(device)
    
    # Extract embedding
    with torch.no_grad():
        logits = model(frame_tensor)
        
        if return_logits:
            return logits.cpu().numpy().squeeze()
        else:
            # Return probabilities
            probs = torch.softmax(logits, dim=1)
            return probs.cpu().numpy().squeeze()


def extract_facial_features(
    video_path: Union[str, Path],
    extract_landmarks: bool = True,
    extract_emotion: bool = True,
    max_frames: int = 30,
    emotion_model: Optional[torch.nn.Module] = None,
) -> dict:
    """
    Extract comprehensive facial features from video.
    
    Args:
        video_path: Path to video file
        extract_landmarks: Whether to extract facial landmarks
        extract_emotion: Whether to extract emotion embeddings
        max_frames: Maximum number of frames to process
        emotion_model: Pre-trained emotion model (optional)
    
    Returns:
        Dictionary with extracted features:
            - 'landmarks': Facial landmarks array
            - 'emotions': Emotion embeddings array
    """
    cap = cv2.VideoCapture(str(video_path))
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, frame_count // max_frames)
    
    landmarks_list = []
    emotions_list = []
    frame_idx = 0
    
    # Initialize MediaPipe if needed
    if extract_landmarks:
        mp_face_mesh = mp.solutions.face_mesh
        face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.5,
        )
    
    while cap.isOpened() and len(landmarks_list) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            if extract_landmarks:
                results = face_mesh.process(frame_rgb)
                if results.multi_face_landmarks:
                    face_landmarks = results.multi_face_landmarks[0]
                    landmarks = []
                    for landmark in face_landmarks.landmark:
                        landmarks.extend([landmark.x, landmark.y, landmark.z])
                    landmarks_list.append(np.array(landmarks))
                else:
                    landmarks_list.append(np.zeros(468 * 3))
            
            if extract_emotion:
                emotion = extract_emotion_embedding(frame_rgb, emotion_model)
                emotions_list.append(emotion)
        
        frame_idx += 1
    
    cap.release()
    
    result = {}
    if extract_landmarks and landmarks_list:
        result['landmarks'] = np.stack(landmarks_list)
    if extract_emotion and emotions_list:
        result['emotions'] = np.stack(emotions_list)
    
    return result

