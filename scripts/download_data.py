"""
Automated data download script for NeuroVoice datasets.

Downloads datasets from various sources including Kaggle, UCI, and provides
instructions for datasets requiring manual registration.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path
import urllib.request

from src.config import DATA_DIR, DATASET_URLS


def download_parkinson_dataset(output_dir: Path):
    """
    Download Parkinson's dataset from UCI ML Repository.
    
    Args:
        output_dir: Directory to save dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    output_file = output_dir / "parkinsons.data"
    
    print(f"Downloading Parkinson's dataset from {url}...")
    try:
        urllib.request.urlretrieve(url, output_file)
        print(f"✓ Downloaded to {output_file}")
    except Exception as e:
        print(f"✗ Error downloading: {e}")
        sys.exit(1)


def download_fer2013_dataset(output_dir: Path):
    """
    Download FER2013 dataset from Kaggle.
    
    Requires Kaggle API credentials. Set up at: https://www.kaggle.com/docs/api
    
    Args:
        output_dir: Directory to save dataset
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if kaggle is installed
    try:
        import kaggle
    except ImportError:
        print("Kaggle API not installed. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "kaggle"])
    
    # Check for Kaggle credentials
    kaggle_dir = Path.home() / ".kaggle"
    credentials_file = kaggle_dir / "kaggle.json"
    
    if not credentials_file.exists():
        print("⚠ Kaggle credentials not found!")
        print("Please download your Kaggle API token from:")
        print("https://www.kaggle.com/settings -> API -> Create New Token")
        print(f"Then place kaggle.json in {kaggle_dir}")
        print("\nAlternatively, download manually from:")
        print("https://www.kaggle.com/datasets/msambare/fer2013")
        return
    
    print("Downloading FER2013 dataset from Kaggle...")
    try:
        subprocess.run([
            "kaggle", "datasets", "download",
            "-d", "msambare/fer2013",
            "-p", str(output_dir),
            "--unzip",
        ], check=True)
        print(f"✓ Downloaded to {output_dir}")
    except subprocess.CalledProcessError as e:
        print(f"✗ Error downloading: {e}")
        print("Make sure Kaggle API credentials are set up correctly.")


def print_manual_instructions(dataset_name: str):
    """Print instructions for manually downloading datasets."""
    dataset_info = DATASET_URLS.get(dataset_name)
    
    if not dataset_info:
        print(f"Unknown dataset: {dataset_name}")
        return
    
    print(f"\n=== Manual Download Instructions for {dataset_name.upper()} ===")
    print(f"URL: {dataset_info['url']}")
    print(f"Description: {dataset_info.get('description', 'N/A')}")
    print("\nSteps:")
    
    if dataset_name == "daic_woz":
        print("1. Visit the DAIC-WOZ website and register for an account")
        print("2. Accept the terms of use")
        print("3. Download the dataset files")
        print(f"4. Extract to: {DATA_DIR / 'daic_woz'}")
    
    elif dataset_name == "dementiabank":
        print("1. Visit the DementiaBank website and register")
        print("2. Agree to the terms of use")
        print("3. Download the Pitt corpus (English)")
        print(f"4. Extract to: {DATA_DIR / 'dementiabank'}")
    
    print("\nAfter downloading, run the preprocessing scripts:")
    print(f"  python scripts/preprocess_audio.py --dataset {dataset_name}")
    print(f"  python scripts/preprocess_video.py --dataset {dataset_name}")


def main():
    """Main download function."""
    parser = argparse.ArgumentParser(description="Download NeuroVoice datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "parkinson", "fer2013", "daic_woz", "dementiabank"],
        default="all",
        help="Dataset to download",
    )
    
    args = parser.parse_args()
    
    datasets_to_download = []
    
    if args.dataset == "all":
        datasets_to_download = ["parkinson", "fer2013", "daic_woz", "dementiabank"]
    else:
        datasets_to_download = [args.dataset]
    
    for dataset in datasets_to_download:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset.upper()}")
        print('='*60)
        
        if dataset == "parkinson":
            download_parkinson_dataset(DATA_DIR / "parkinson_tsi")
        
        elif dataset == "fer2013":
            download_fer2013_dataset(DATA_DIR / "faces" / "fer2013")
        
        elif dataset in ["daic_woz", "dementiabank"]:
            print_manual_instructions(dataset)
        
        else:
            print(f"Unknown dataset: {dataset}")
    
    print("\n✓ Download process completed!")
    print("\nNext steps:")
    print("1. Complete manual downloads if needed")
    print("2. Run preprocessing scripts:")
    print("   python scripts/preprocess_audio.py")
    print("   python scripts/preprocess_video.py")
    print("3. Split data:")
    print("   python scripts/split_data.py")


if __name__ == "__main__":
    main()

