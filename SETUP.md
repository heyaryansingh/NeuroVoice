# Setup Guide

Quick setup guide for NeuroVoice project.

## Initial Setup

1. **Clone and Navigate**
   ```bash
   git clone <repository-url>
   cd NeuroVoice
   ```

2. **Create Virtual Environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Dataset Setup

1. **Download Datasets**
   ```bash
   # Automated downloads
   python scripts/download_data.py --dataset parkinson
   python scripts/download_data.py --dataset fer2013
   
   # Manual downloads (follow instructions)
   python scripts/download_data.py --dataset daic_woz
   python scripts/download_data.py --dataset dementiabank
   ```

2. **Preprocess Data**
   ```bash
   python scripts/preprocess_audio.py --dataset all
   python scripts/preprocess_video.py --dataset all
   ```

3. **Split Data**
   ```bash
   python scripts/split_data.py --dataset all
   ```

## Training

1. **Train Model**
   ```bash
   python src/training/train.py --disease alzheimer --epochs 50 --use_gpu
   ```

2. **Evaluate Model**
   ```bash
   python src/training/evaluate.py --model_path outputs/models/alzheimer_best_model.pt --disease alzheimer --test_data data/splits/
   ```

## Jupyter Notebooks

1. **Start Jupyter**
   ```bash
   jupyter notebook
   ```

2. **Run Analysis Notebooks**
   - `01_data_exploration.ipynb` - Explore datasets
   - `02_audio_feature_analysis.ipynb` - Analyze audio features
   - `03_video_feature_analysis.ipynb` - Analyze video features
   - `04_multimodal_training.ipynb` - Interactive training
   - `05_results_visualization.ipynb` - Visualize results
   - `06_explainability.ipynb` - Model interpretability

## Testing

Run tests:
```bash
pytest tests/
```

Run specific test:
```bash
pytest tests/test_fusion_model.py
```

## Notes

- Ensure you have sufficient disk space for datasets (several GB)
- Some datasets require manual registration and download
- GPU recommended for training but not required (CPU fallback available)
- Check `data/README.md` for detailed dataset information

