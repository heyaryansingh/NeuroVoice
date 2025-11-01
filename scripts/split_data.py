"""
Data splitting script.

Splits datasets into train/validation/test sets.
"""

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (
    DATA_DIR,
    PROCESSED_DATA_DIR,
    SPLITS_DIR,
    SPLIT_CONFIG,
)


def split_dataset(
    metadata_path: Path,
    output_dir: Path,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    stratify: bool = True,
    random_seed: int = 42,
):
    """
    Split dataset into train/val/test sets.
    
    Args:
        metadata_path: Path to metadata CSV file
        output_dir: Directory to save split files
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        stratify: Whether to stratify by labels
        random_seed: Random seed for reproducibility
    """
    # Validate ratios
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"
    
    # Load metadata
    df = pd.read_csv(metadata_path)
    
    # Ensure we have labels
    if 'label' not in df.columns:
        print("⚠ Warning: No 'label' column found. Cannot stratify.")
        stratify = False
    
    # Split
    if stratify and 'label' in df.columns:
        # First split: train vs. (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            stratify=df['label'],
            random_state=random_seed,
        )
        
        # Second split: val vs. test
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            stratify=temp_df['label'],
            random_state=random_seed,
        )
    else:
        # Non-stratified split
        train_df, temp_df = train_test_split(
            df,
            test_size=(val_ratio + test_ratio),
            random_state=random_seed,
        )
        
        val_size = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_size),
            random_state=random_seed,
        )
    
    # Save splits
    output_dir.mkdir(parents=True, exist_ok=True)
    
    dataset_name = metadata_path.parent.name
    train_df.to_csv(output_dir / f"{dataset_name}_train.csv", index=False)
    val_df.to_csv(output_dir / f"{dataset_name}_val.csv", index=False)
    test_df.to_csv(output_dir / f"{dataset_name}_test.csv", index=False)
    
    print(f"✓ Split complete for {dataset_name}")
    print(f"  Train: {len(train_df)} samples ({len(train_df)/len(df)*100:.1f}%)")
    print(f"  Val:   {len(val_df)} samples ({len(val_df)/len(df)*100:.1f}%)")
    print(f"  Test:  {len(test_df)} samples ({len(test_df)/len(df)*100:.1f}%)")


def main():
    """Main splitting function."""
    parser = argparse.ArgumentParser(description="Split data into train/val/test")
    parser.add_argument(
        "--split",
        type=float,
        nargs=3,
        default=[SPLIT_CONFIG["train_ratio"], SPLIT_CONFIG["val_ratio"], SPLIT_CONFIG["test_ratio"]],
        help="Train/val/test ratios (should sum to 1.0)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["all", "alzheimer", "parkinson", "depression"],
        default="all",
        help="Dataset to split",
    )
    parser.add_argument(
        "--no_stratify",
        action="store_true",
        help="Disable stratification by labels",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SPLIT_CONFIG["random_seed"],
        help="Random seed",
    )
    
    args = parser.parse_args()
    
    train_ratio, val_ratio, test_ratio = args.split
    
    datasets = ["alzheimer", "parkinson", "depression"] if args.dataset == "all" else [args.dataset]
    
    for dataset in datasets:
        # Find metadata files
        audio_metadata = PROCESSED_DATA_DIR / "audio" / dataset / "metadata.csv"
        video_metadata = PROCESSED_DATA_DIR / "video" / dataset / "metadata.csv"
        
        if audio_metadata.exists():
            print(f"\n{'='*60}")
            print(f"Splitting audio data: {dataset.upper()}")
            print('='*60)
            split_dataset(
                audio_metadata,
                SPLITS_DIR,
                train_ratio,
                val_ratio,
                test_ratio,
                stratify=not args.no_stratify,
                random_seed=args.seed,
            )
        
        if video_metadata.exists():
            print(f"\n{'='*60}")
            print(f"Splitting video data: {dataset.upper()}")
            print('='*60)
            split_dataset(
                video_metadata,
                SPLITS_DIR,
                train_ratio,
                val_ratio,
                test_ratio,
                stratify=not args.no_stratify,
                random_seed=args.seed,
            )
    
    print("\n✓ Data splitting complete!")


if __name__ == "__main__":
    main()

