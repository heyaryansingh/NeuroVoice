"""
Video preprocessing script.

Extracts facial landmarks and emotion embeddings from video files.
"""

import argparse
import pickle
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    FEATURE_CONFIG,
)
from src.features.video_features import (
    extract_facial_landmarks,
    extract_facial_features,
)


def preprocess_video_file(
    video_path: Path,
    feature_types: List[str],
    output_dir: Path,
    max_frames: int = 30,
) -> dict:
    """
    Preprocess a single video file.
    
    Args:
        video_path: Path to video file
        feature_types: List of feature types to extract
        output_dir: Directory to save processed features
        max_frames: Maximum number of frames to process
    
    Returns:
        Dictionary with feature paths and metadata
    """
    features = {}
    
    try:
        if "landmarks" in feature_types or "both" in feature_types:
            landmarks = extract_facial_landmarks(video_path, max_faces=1)
            feature_path = output_dir / f"{video_path.stem}_landmarks.npy"
            np.save(feature_path, landmarks)
            features['landmarks_path'] = str(feature_path)
        
        if "emotion" in feature_types or "both" in feature_types:
            # Extract comprehensive features (includes landmarks + emotion)
            facial_features = extract_facial_features(
                video_path,
                extract_landmarks=True,
                extract_emotion=True,
                max_frames=max_frames,
            )
            
            if 'landmarks' in facial_features:
                landmarks_path = output_dir / f"{video_path.stem}_landmarks.npy"
                np.save(landmarks_path, facial_features['landmarks'])
                features['landmarks_path'] = str(landmarks_path)
            
            if 'emotions' in facial_features:
                emotion_path = output_dir / f"{video_path.stem}_emotions.npy"
                np.save(emotion_path, facial_features['emotions'])
                features['emotions_path'] = str(emotion_path)
    
    except Exception as e:
        print(f"Error processing {video_path}: {e}")
        return {}
    
    features['video_path'] = str(video_path)
    return features


def preprocess_dataset(
    dataset_name: str,
    feature_types: List[str] = None,
    max_frames: int = 30,
):
    """
    Preprocess video files for a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('alzheimer', 'parkinson', 'depression')
        feature_types: List of feature types to extract
        max_frames: Maximum frames per video
    """
    feature_types = feature_types or FEATURE_CONFIG["video_features"]
    
    # Dataset-specific paths
    if dataset_name == "alzheimer":
        data_dir = DATA_DIR / "dementiabank"
    elif dataset_name == "parkinson":
        data_dir = DATA_DIR / "parkinson_tsi"
    elif dataset_name == "depression":
        data_dir = DATA_DIR / "daic_woz"
    else:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    # Output directory
    output_dir = PROCESSED_DATA_DIR / "video" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find video files
    video_files = list(data_dir.rglob("*.mp4")) + list(data_dir.rglob("*.avi"))
    
    if not video_files:
        print(f"No video files found in {data_dir}")
        print("Note: Some datasets may only have audio files")
        return
    
    print(f"Found {len(video_files)} video files")
    print(f"Extracting features: {feature_types}")
    
    # Process files
    processed_features = []
    for video_file in tqdm(video_files, desc="Processing video"):
        features = preprocess_video_file(video_file, feature_types, output_dir, max_frames)
        if features:
            processed_features.append(features)
    
    # Save metadata
    if processed_features:
        metadata_df = pd.DataFrame(processed_features)
        metadata_path = output_dir / "metadata.csv"
        metadata_df.to_csv(metadata_path, index=False)
        print(f"\n✓ Processed {len(processed_features)} files")
        print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess video data")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "alzheimer", "parkinson", "depression"],
        default="all",
        help="Dataset to preprocess",
    )
    parser.add_argument(
        "--features",
        type=str,
        nargs="+",
        choices=["landmarks", "emotion", "both"],
        default=None,
        help="Feature types to extract",
    )
    parser.add_argument(
        "--max_frames",
        type=int,
        default=30,
        help="Maximum frames per video",
    )
    
    args = parser.parse_args()
    
    datasets = ["alzheimer", "parkinson", "depression"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Preprocessing: {dataset.upper()}")
        print('='*60)
        preprocess_dataset(dataset, args.features, args.max_frames)


if __name__ == "__main__":
    main()

