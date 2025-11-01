# NeuroVoice 🧠🔊

**A Multimodal AI System for Diagnosing Alzheimer's, Parkinson's, and Depression from Speech and Facial Expression Data**

NeuroVoice is a deep learning framework that leverages **multimodal fusion** (audio + facial expressions) to detect neurodegenerative and mental health conditions. The system uses state-of-the-art transformer embeddings and cross-modal attention mechanisms to identify biomarkers in speech patterns and facial microexpressions.

---

## 🎯 Features

- **Multimodal Fusion**: Combines audio and visual data for improved diagnosis accuracy
- **Transformer-Based Embeddings**: Uses wav2vec2 for speech and Vision Transformers for facial expressions
- **Cross-Modal Attention**: Quantifies correlation between facial microexpressions and vocal tremors
- **Explainability Layer**: Visualizes neurodegenerative biomarkers in speech and facial dynamics
- **Multi-Disease Classification**: Supports Alzheimer's, Parkinson's, and Depression detection

---

## 📦 Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (optional, CPU fallback available)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/NeuroVoice.git
cd NeuroVoice
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 📊 Datasets

NeuroVoice uses the following publicly available datasets:

### 1. **DementiaBank Pitt Corpus** (Alzheimer's)
- **Source**: [https://dementia.talkbank.org/access/English/Pitt.html](https://dementia.talkbank.org/access/English/Pitt.html)
- Contains speech recordings and transcripts from Alzheimer's patients and controls
- Requires registration and agreement to terms of use

### 2. **DAIC-WOZ** (Depression)
- **Source**: [https://dcapswoz.ict.usc.edu/](https://dcapswoz.ict.usc.edu/)
- Contains audio, video, and transcript data from clinical interviews
- Free for academic use (requires signup)

### 3. **Parkinson's Telemonitoring Dataset** (UCI ML Repository)
- **Source**: [https://archive.ics.uci.edu/ml/datasets/parkinsons](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- Contains 195 sustained phonations from Parkinson's patients and controls
- Publicly available

### 4. **FER2013** (Facial Expressions)
- **Source**: [https://www.kaggle.com/datasets/msambare/fer2013](https://www.kaggle.com/datasets/msambare/fer2013)
- Large-scale facial expression dataset with 7 emotion classes
- Available via Kaggle API

### Downloading Datasets

Run the automated download script:

```bash
python scripts/download_data.py
```

**Note**: Some datasets require manual registration. The script will provide instructions for datasets that cannot be automatically downloaded.

---

## 🚀 Quick Start

### 1. Data Preparation

Download and preprocess the datasets:

```bash
# Download datasets
python scripts/download_data.py

# Preprocess audio data
python scripts/preprocess_audio.py --dataset all

# Preprocess video data
python scripts/preprocess_video.py --dataset all

# Split data into train/val/test
python scripts/split_data.py --split 0.7 0.15 0.15
```

### 2. Training

Train the multimodal fusion model:

```bash
python src/training/train.py \
    --disease alzheimer \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_gpu
```

Train for multiple diseases:

```bash
# Alzheimer's
python src/training/train.py --disease alzheimer --epochs 50

# Parkinson's
python src/training/train.py --disease parkinson --epochs 50

# Depression
python src/training/train.py --disease depression --epochs 50
```

### 3. Evaluation

Evaluate a trained model:

```bash
python src/training/evaluate.py \
    --model_path outputs/models/best_model.pt \
    --disease alzheimer \
    --test_data data/processed/test/
```

### 4. Jupyter Notebooks

Explore the analysis notebooks:

```bash
jupyter notebook notebooks/
```

- `01_data_exploration.ipynb` - Dataset overview and statistics
- `02_audio_feature_analysis.ipynb` - Audio feature extraction and visualization
- `03_video_feature_analysis.ipynb` - Facial expression analysis
- `04_multimodal_training.ipynb` - Interactive training and experimentation
- `05_results_visualization.ipynb` - Model performance metrics and plots
- `06_explainability.ipynb` - Attention maps and saliency visualization

---

## 🏗️ Project Structure

```
NeuroVoice/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
│
├── data/                        # Dataset storage
│   ├── README.md                # Dataset documentation
│   ├── daic_woz/                # DAIC-WOZ depression dataset
│   ├── parkinson_tsi/           # Parkinson's dataset
│   ├── dementiabank/            # Alzheimer's speech dataset
│   └── faces/                   # Facial expression datasets
│
├── scripts/                     # Utility scripts
│   ├── download_data.py         # Automated data downloaders
│   ├── preprocess_audio.py      # Audio preprocessing
│   ├── preprocess_video.py      # Video preprocessing
│   └── split_data.py            # Data splitting logic
│
├── src/                         # Source code
│   ├── config.py                # Configuration and paths
│   ├── data_loaders/            # PyTorch data loaders
│   ├── features/                # Feature extraction modules
│   ├── models/                  # Deep learning models
│   ├── training/                # Training and evaluation
│   └── utils/                   # Utility functions
│
├── notebooks/                   # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_audio_feature_analysis.ipynb
│   ├── 03_video_feature_analysis.ipynb
│   ├── 04_multimodal_training.ipynb
│   ├── 05_results_visualization.ipynb
│   └── 06_explainability.ipynb
│
├── tests/                       # Unit tests
│   ├── test_audio_pipeline.py
│   ├── test_video_pipeline.py
│   ├── test_model_training.py
│   └── test_fusion_model.py
│
└── outputs/                     # Training outputs
    ├── models/                  # Saved model checkpoints
    ├── logs/                    # Training logs
    ├── metrics/                 # Evaluation metrics
    └── visualizations/          # Generated plots
```

---

## 🧠 Model Architecture

### Audio Processing
- **MFCC Features**: 13-dimensional MFCCs extracted from speech segments
- **Wav2Vec2 Embeddings**: Transformer-based speech embeddings from `facebook/wav2vec2-base-960h`
- **LSTM/CNN Encoder**: Temporal modeling of audio sequences

### Video Processing
- **MediaPipe Landmarks**: 468 facial landmarks for geometric features
- **Emotion Embeddings**: Pre-trained emotion classifier (trained on FER2013/AffectNet)
- **ResNet/ViT Encoder**: Deep feature extraction from facial frames

### Multimodal Fusion
- **Cross-Modal Attention**: Learns interactions between audio and visual modalities
- **Gated Fusion**: Adaptive weighting of modalities
- **Classification Head**: Multi-task learning for disease classification

---

## 📈 Results

### Example Metrics (Placeholder - Update after training)

| Disease | Accuracy | ROC-AUC | Sensitivity | Specificity |
|---------|----------|---------|-------------|-------------|
| Alzheimer's | - | - | - | - |
| Parkinson's | - | - | - | - |
| Depression | - | - | - | - |

---

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

Or run individual tests:

```bash
pytest tests/test_audio_pipeline.py
pytest tests/test_fusion_model.py
```

---

## 🔬 Explainability

Visualize model attention and saliency maps:

```bash
jupyter notebook notebooks/06_explainability.ipynb
```

The explainability module provides:
- **Attention Heatmaps**: Shows which audio segments and facial regions the model focuses on
- **Saliency Maps**: Highlights important features in waveforms and video frames
- **Gradient Visualization**: Grad-CAM style visualizations for interpretability

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📝 Citation

If you use NeuroVoice in your research, please cite:

```bibtex
@software{neurovoice2024,
  title={NeuroVoice: Multimodal AI for Neurodegenerative Disease Detection},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/NeuroVoice}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- **DementiaBank** for providing the Alzheimer's speech corpus
- **USC Institute for Creative Technologies** for the DAIC-WOZ dataset
- **UCI Machine Learning Repository** for the Parkinson's dataset
- **Kaggle** for hosting the FER2013 dataset
- **Hugging Face** for transformer models (wav2vec2)
- **MediaPipe** for facial landmark detection

---

## ⚠️ Disclaimer

**This tool is for research purposes only and should not be used for clinical diagnosis without proper validation and medical supervision.**

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your-email@example.com].

---

**Built with ❤️ for advancing healthcare AI**

