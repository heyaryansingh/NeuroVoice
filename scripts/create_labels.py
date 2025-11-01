"""
Create label CSV files for datasets.

Parses dataset-specific formats and generates standardized label CSVs
with format: file_path,label,disease
"""

import argparse
import csv
from pathlib import Path
from typing import List, Tuple

import pandas as pd

from src.config import DATA_DIR, SPLITS_DIR


def parse_dementiabank_labels(data_dir: Path) -> List[Tuple[str, int, str]]:
    """
    Parse DementiaBank Pitt Corpus labels.
    
    Assumes structure:
    - Transcript files (.cha) with metadata
    - Audio files (.wav) corresponding to transcripts
    - Control vs. Dementia labels in transcript metadata
    
    Args:
        data_dir: Directory containing DementiaBank data
    
    Returns:
        List of tuples (file_path, label, disease)
        label: 0=control, 1=alzheimer
    """
    labels = []
    
    # Look for transcript files or metadata files
    # Format varies by dataset - this is a template implementation
    transcript_dir = data_dir / "transcripts"
    audio_dir = data_dir / "audio"
    
    if not transcript_dir.exists():
        transcript_dir = data_dir
    
    # Find all audio files
    audio_files = list(audio_dir.rglob("*.wav")) if audio_dir.exists() else []
    
    # Parse transcripts or metadata
    # This is dataset-specific - adjust based on actual DementiaBank format
    for audio_file in audio_files:
        # Try to find corresponding transcript or metadata
        transcript_file = transcript_dir / audio_file.with_suffix('.cha').name
        
        if transcript_file.exists():
            # Read transcript file to extract label
            # DementiaBank format: *PAR: ... (participant), *INV: ... (interviewer)
            # Control group typically has specific naming or metadata
            with open(transcript_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read().lower()
                
                # Check for dementia indicators (adjust based on actual format)
                if 'dementia' in content or 'ad' in content or 'dem' in content:
                    label = 1  # Alzheimer's
                elif 'control' in content or 'normal' in content:
                    label = 0  # Control
                else:
                    # Default: skip if unclear
                    continue
        else:
            # If no transcript, check filename patterns
            filename_lower = audio_file.stem.lower()
            if 'dem' in filename_lower or 'ad' in filename_lower:
                label = 1
            elif 'control' in filename_lower or 'norm' in filename_lower:
                label = 0
            else:
                continue  # Skip unclear cases
        
        labels.append((
            str(audio_file.relative_to(DATA_DIR)),
            label,
            'alzheimer'
        ))
    
    return labels


def parse_daic_woz_labels(data_dir: Path, threshold: int = 10) -> List[Tuple[str, int, str]]:
    """
    Parse DAIC-WOZ depression dataset labels.
    
    Uses PHQ-8 scores from metadata.
    Threshold >= 10 indicates depression.
    
    Args:
        data_dir: Directory containing DAIC-WOZ data
        threshold: PHQ-8 threshold for depression (default: 10)
    
    Returns:
        List of tuples (file_path, label, disease)
        label: 0=no depression, 1=depression
    """
    labels = []
    
    # Look for metadata file (PHQ-8 scores)
    metadata_file = data_dir / "metadata.csv"
    
    if not metadata_file.exists():
        # Try alternative locations
        metadata_file = data_dir / "train" / "metadata.csv"
        if not metadata_file.exists():
            metadata_file = data_dir / "dev" / "metadata.csv"
    
    if metadata_file.exists():
        df = pd.read_csv(metadata_file)
        
        # Expected columns: participant_id, PHQ_Binary, PHQ_Score, etc.
        for _, row in df.iterrows():
            participant_id = str(row.get('Participant_ID', row.get('participant_id', '')))
            
            # Get PHQ-8 score
            phq_score = row.get('PHQ_Score', row.get('PHQ8_Score', row.get('phq8_score', None)))
            phq_binary = row.get('PHQ_Binary', row.get('phq_binary', None))
            
            # Determine label
            if pd.notna(phq_binary):
                label = int(phq_binary)
            elif pd.notna(phq_score):
                label = 1 if phq_score >= threshold else 0
            else:
                continue  # Skip if no score available
            
            # Find corresponding audio/video file
            audio_file = data_dir / f"{participant_id}_AUDIO.wav"
            if not audio_file.exists():
                audio_file = data_dir / "audio" / f"{participant_id}.wav"
            if not audio_file.exists():
                # Try with different naming
                audio_file = data_dir / f"{participant_id}.wav"
            
            if audio_file.exists():
                labels.append((
                    str(audio_file.relative_to(DATA_DIR)),
                    label,
                    'depression'
                ))
    else:
        # Fallback: look for audio files and try to infer from directory structure
        audio_dir = data_dir / "audio"
        if audio_dir.exists():
            for audio_file in audio_dir.rglob("*.wav"):
                # Try to extract participant ID from filename
                # Format may vary: 300_AUDIO.wav, participant_300.wav, etc.
                filename = audio_file.stem
                
                # This is a fallback - adjust based on actual DAIC-WOZ structure
                labels.append((
                    str(audio_file.relative_to(DATA_DIR)),
                    0,  # Default to control - requires manual correction
                    'depression'
                ))
    
    return labels


def parse_parkinson_labels(data_dir: Path) -> List[Tuple[str, int, str]]:
    """
    Parse Parkinson's UCI dataset labels.
    
    Uses the parkinsons.data CSV file with status column.
    
    Args:
        data_dir: Directory containing Parkinson's data
    
    Returns:
        List of tuples (file_path, label, disease)
        label: 0=healthy, 1=parkinson
    """
    labels = []
    
    # Look for parkinsons.data file
    data_file = data_dir / "parkinsons.data"
    
    if data_file.exists():
        df = pd.read_csv(data_file)
        
        # Expected columns: name, status (0=healthy, 1=parkinson)
        for _, row in df.iterrows():
            name = row.get('name', '')
            status = row.get('status', row.get('Status', None))
            
            if pd.isna(status):
                continue
            
            label = int(status)
            
            # Find corresponding audio file
            # UCI dataset may have different naming conventions
            audio_file = data_dir / f"{name}.wav"
            if not audio_file.exists():
                audio_file = data_dir / "audio" / f"{name}.wav"
            if not audio_file.exists():
                # Try alternative naming
                audio_file = data_dir / f"{name}_audio.wav"
            
            if audio_file.exists():
                labels.append((
                    str(audio_file.relative_to(DATA_DIR)),
                    label,
                    'parkinson'
                ))
    else:
        # Fallback: if structured differently
        audio_dir = data_dir / "audio"
        if audio_dir.exists():
            # This is a placeholder - actual structure may vary
            for audio_file in audio_dir.rglob("*.wav"):
                labels.append((
                    str(audio_file.relative_to(DATA_DIR)),
                    0,  # Default - requires manual correction
                    'parkinson'
                ))
    
    return labels


def create_label_csv(
    dataset_name: str,
    labels: List[Tuple[str, int, str]],
    output_path: Path,
):
    """
    Create label CSV file with multimodal paths.
    
    Args:
        dataset_name: Name of dataset
        labels: List of (file_path, label, disease) tuples
        output_path: Path to save CSV
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame and create separate audio/video paths
    df = pd.DataFrame(labels, columns=['file_path', 'label', 'disease'])
    
    # For multimodal: create audio_path and video_path
    # If file_path is audio (.wav), try to find corresponding video
    # If file_path is video (.mp4), try to find corresponding audio
    audio_paths = []
    video_paths = []
    
    for file_path in df['file_path']:
        path = Path(file_path)
        
        # Determine if it's audio or video
        if path.suffix.lower() in ['.wav', '.mp3', '.flac']:
            audio_paths.append(file_path)
            # Try to find corresponding video
            video_path = str(path.with_suffix('.mp4'))
            if not Path(video_path).exists():
                video_path = str(path.with_suffix('.avi'))
            if not Path(video_path).exists():
                video_path = file_path  # Fallback to same file
            video_paths.append(video_path)
        elif path.suffix.lower() in ['.mp4', '.avi', '.mov']:
            video_paths.append(file_path)
            # Try to find corresponding audio
            audio_path = str(path.with_suffix('.wav'))
            if not Path(audio_path).exists():
                audio_path = file_path  # Fallback to same file
            audio_paths.append(audio_path)
        else:
            # Unknown format, use as both
            audio_paths.append(file_path)
            video_paths.append(file_path)
    
    df['audio_path'] = audio_paths
    df['video_path'] = video_paths
    
    # Reorder columns: file_path, audio_path, video_path, label, disease
    df = df[['file_path', 'audio_path', 'video_path', 'label', 'disease']]
    
    df.to_csv(output_path, index=False)
    
    print(f"✓ Created label CSV for {dataset_name}")
    print(f"  Total samples: {len(df)}")
    print(f"  Label distribution:")
    print(df['label'].value_counts().to_string())
    print(f"  Saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Create label CSV files for datasets")
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "alzheimer", "depression", "parkinson"],
        default="all",
        help="Dataset to process",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(SPLITS_DIR),
        help="Output directory for label CSVs",
    )
    parser.add_argument(
        "--phq_threshold",
        type=int,
        default=10,
        help="PHQ-8 threshold for depression (DAIC-WOZ)",
    )
    
    args = parser.parse_args()
    
    datasets_to_process = []
    if args.dataset == "all":
        datasets_to_process = ["alzheimer", "depression", "parkinson"]
    else:
        datasets_to_process = [args.dataset]
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for dataset in datasets_to_process:
        print(f"\n{'='*60}")
        print(f"Processing: {dataset.upper()}")
        print('='*60)
        
        if dataset == "alzheimer":
            data_dir = DATA_DIR / "dementiabank"
            if not data_dir.exists():
                print(f"⚠ Warning: {data_dir} not found. Skipping.")
                continue
            labels = parse_dementiabank_labels(data_dir)
            output_path = output_dir / "alzheimer_labels.csv"
        
        elif dataset == "depression":
            data_dir = DATA_DIR / "daic_woz"
            if not data_dir.exists():
                print(f"⚠ Warning: {data_dir} not found. Skipping.")
                continue
            labels = parse_daic_woz_labels(data_dir, threshold=args.phq_threshold)
            output_path = output_dir / "depression_labels.csv"
        
        elif dataset == "parkinson":
            data_dir = DATA_DIR / "parkinson_tsi"
            if not data_dir.exists():
                print(f"⚠ Warning: {data_dir} not found. Skipping.")
                continue
            labels = parse_parkinson_labels(data_dir)
            output_path = output_dir / "parkinson_labels.csv"
        
        else:
            continue
        
        if labels:
            create_label_csv(dataset, labels, output_path)
        else:
            print(f"⚠ No labels extracted for {dataset}. Check data directory structure.")


if __name__ == "__main__":
    main()

