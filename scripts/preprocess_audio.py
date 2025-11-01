"""
Audio preprocessing script.

Cleans, segments, and extracts features from audio files.
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
    AUDIO_CONFIG,
    FEATURE_CONFIG,
)
from src.features.audio_features import (
    extract_mfcc,
    extract_wav2vec_embeddings,
    extract_spectral_features,
)


def preprocess_audio_file(
    audio_path: Path,
    feature_types: List[str],
    output_dir: Path,
) -> dict:
    """
    Preprocess a single audio file.
    
    Args:
        audio_path: Path to audio file
        feature_types: List of feature types to extract
        output_dir: Directory to save processed features
    
    Returns:
        Dictionary with feature paths and metadata
    """
    features = {}
    
    # Extract features
    for feature_type in feature_types:
        try:
            if feature_type == "mfcc":
                mfcc = extract_mfcc(audio_path)
                feature_path = output_dir / f"{audio_path.stem}_mfcc.npy"
                np.save(feature_path, mfcc)
                features['mfcc_path'] = str(feature_path)
            
            elif feature_type == "wav2vec":
                wav2vec = extract_wav2vec_embeddings(audio_path)
                feature_path = output_dir / f"{audio_path.stem}_wav2vec.npy"
                np.save(feature_path, wav2vec)
                features['wav2vec_path'] = str(feature_path)
            
            elif feature_type == "spectral":
                spectral = extract_spectral_features(audio_path)
                feature_path = output_dir / f"{audio_path.stem}_spectral.pkl"
                with open(feature_path, 'wb') as f:
                    pickle.dump(spectral, f)
                features['spectral_path'] = str(feature_path)
        
        except Exception as e:
            print(f"Error processing {audio_path} for {feature_type}: {e}")
            continue
    
    features['audio_path'] = str(audio_path)
    return features


def preprocess_dataset(
    dataset_name: str,
    feature_types: List[str] = None,
):
    """
    Preprocess audio files for a specific dataset.
    
    Args:
        dataset_name: Name of dataset ('alzheimer', 'parkinson', 'depression')
        feature_types: List of feature types to extract
    """
    feature_types = feature_types or FEATURE_CONFIG["audio_features"]
    
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
    output_dir = PROCESSED_DATA_DIR / "audio" / dataset_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find audio files
    audio_files = list(data_dir.rglob("*.wav")) + list(data_dir.rglob("*.mp3"))
    
    if not audio_files:
        print(f"No audio files found in {data_dir}")
        print("Please download the dataset first using:")
        print("  python scripts/download_data.py --dataset <dataset_name>")
        return
    
    print(f"Found {len(audio_files)} audio files")
    print(f"Extracting features: {feature_types}")
    
    # Process files
    processed_features = []
    for audio_file in tqdm(audio_files, desc="Processing audio"):
        features = preprocess_audio_file(audio_file, feature_types, output_dir)
        if features:
            processed_features.append(features)
    
    # Save metadata
    metadata_df = pd.DataFrame(processed_features)
    metadata_path = output_dir / "metadata.csv"
    metadata_df.to_csv(metadata_path, index=False)
    print(f"\n✓ Processed {len(processed_features)} files")
    print(f"✓ Metadata saved to {metadata_path}")


def main():
    """Main preprocessing function."""
    parser = argparse.ArgumentParser(description="Preprocess audio data")
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
        choices=["mfcc", "wav2vec", "spectral"],
        default=None,
        help="Feature types to extract",
    )
    
    args = parser.parse_args()
    
    datasets = ["alzheimer", "parkinson", "depression"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Preprocessing: {dataset.upper()}")
        print('='*60)
        preprocess_dataset(dataset, args.features)


if __name__ == "__main__":
    main()

