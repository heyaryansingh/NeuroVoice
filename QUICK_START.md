# NeuroVoice Quick Start Guide

Complete guide to get NeuroVoice running with advanced ML features and web interface.

## 🚀 Complete Setup & Training

### 1. Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt

# Install web backend dependencies
pip install -r web/backend/requirements.txt

# Install frontend dependencies
cd web/frontend
npm install
cd ../..
```

### 2. Prepare Datasets

```bash
# Download datasets
python scripts/download_data.py --dataset parkinson
python scripts/download_data.py --dataset fer2013

# Create label CSVs (now with multimodal format!)
python scripts/create_labels.py --dataset all

# Preprocess data
python scripts/preprocess_audio.py --dataset all
python scripts/preprocess_video.py --dataset all

# Split into train/val/test
python scripts/split_data.py --dataset all
```

### 3. Train Models

#### Basic Training
```bash
python src/training/train.py \
    --disease alzheimer \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --use_gpu
```

#### Advanced Training with All Features
```bash
python src/training/train.py \
    --disease alzheimer \
    --epochs 50 \
    --batch_size 16 \
    --lr 1e-4 \
    --fusion crossmodal \
    --use_sam \
    --sam_rho 0.05 \
    --use_lookahead \
    --gradient_accumulation 4 \
    --analyze_gradients \
    --use_tensorboard \
    --experiment_name alzheimer_crossmodal_sam \
    --use_gpu
```

#### Train All Diseases
```bash
for disease in alzheimer parkinson depression; do
    python src/training/train.py \
        --disease $disease \
        --epochs 50 \
        --fusion crossmodal \
        --use_tensorboard \
        --use_gpu
done
```

### 4. View Training Metrics

```bash
# Start TensorBoard
tensorboard --logdir outputs/logs/tensorboard

# Open browser to http://localhost:6006
```

### 5. Run Web Interface

#### Start Backend
```bash
cd web/backend
python main.py
# API runs on http://localhost:8000
```

#### Start Frontend
```bash
cd web/frontend
npm run dev
# Web UI runs on http://localhost:3000
```

### 6. Deploy to Vercel

#### Frontend (Next.js)
1. Push code to GitHub
2. Import project in Vercel dashboard
3. Set environment variable: `NEXT_PUBLIC_API_URL` (your backend URL)
4. Deploy!

#### Backend (FastAPI)
Deploy to:
- **Railway**: Connect GitHub repo, auto-deploys
- **Render**: Create Web Service, point to `web/backend/main.py`
- **Heroku**: Use Procfile with `web: uvicorn main:app --host 0.0.0.0 --port $PORT`

## 📊 Advanced Features Enabled

### ✅ Training Features
- **SAM Optimizer**: Sharpness-Aware Minimization for better generalization
- **Lookahead**: k steps forward, 1 step back optimization
- **Gradient Accumulation**: Train with larger effective batch sizes
- **Gradient Analysis**: Real-time gradient statistics logging
- **TensorBoard Integration**: Visualize training metrics
- **Experiment Tracking**: Version control for experiments

### ✅ Model Features
- **Multimodal Fusion**: Cross-modal attention between audio and video
- **Wav2Vec2 Embeddings**: State-of-the-art speech representations
- **Emotion Recognition**: FER2013-based emotion detection
- **Early/Late Fusion**: Multiple fusion strategies

### ✅ Web Interface Features
- **Real-time Prediction**: Upload audio/video for instant diagnosis
- **Confidence Scores**: Detailed confidence metrics
- **Model Statistics**: Visualize model performance
- **Feature Analysis**: Analyze extracted features
- **Responsive Design**: Works on desktop and mobile
- **Dark Mode**: Modern UI with dark theme support

## 🎯 Usage Examples

### Command-Line Training
```bash
# Alzheimer's with SAM optimizer
python src/training/train.py --disease alzheimer --use_sam --analyze_gradients --use_gpu

# Parkinson's with early fusion
python src/training/train.py --disease parkinson --fusion early --use_gpu

# Depression with all advanced features
python src/training/train.py \
    --disease depression \
    --fusion crossmodal \
    --use_sam \
    --use_lookahead \
    --analyze_gradients \
    --use_tensorboard \
    --experiment_name depression_full \
    --use_gpu
```

### Web Interface
1. Open http://localhost:3000
2. Select disease type
3. Upload audio file (required)
4. Upload video file (optional)
5. Click "Predict"
6. View results with confidence scores

### API Usage
```bash
# Make prediction via API
curl -X POST "http://localhost:8000/predict" \
  -F "audio_file=@test.wav" \
  -F "video_file=@test.mp4" \
  -F "disease=alzheimer"

# Get model statistics
curl "http://localhost:8000/models"

# Analyze features
curl -X POST "http://localhost:8000/analyze" \
  -F "audio_file=@test.wav" \
  -F "video_file=@test.mp4"
```

## 📈 Performance Tips

1. **Use GPU**: Training is 10-50x faster on GPU
2. **Gradient Accumulation**: Use with small GPUs to simulate larger batch sizes
3. **Mixed Precision**: Consider using `torch.cuda.amp` for faster training
4. **Data Loading**: Increase `num_workers` if you have many CPU cores
5. **Model Caching**: Web backend caches loaded models for faster inference

## 🔧 Troubleshooting

### Issue: "Model not found"
**Solution**: Train a model first or check model path in config

### Issue: "CSV format error"
**Solution**: Run `python scripts/create_labels.py --dataset all` again

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use gradient accumulation

### Issue: "Web interface can't connect to API"
**Solution**: Check `NEXT_PUBLIC_API_URL` environment variable

## 🎓 Next Steps

1. **Experiment with hyperparameters**: Try different learning rates, batch sizes
2. **Try different fusion strategies**: Compare early, late, and cross-modal fusion
3. **Visualize gradients**: Use TensorBoard to analyze training dynamics
4. **Evaluate models**: Run evaluation scripts to get detailed metrics
5. **Deploy**: Share your trained models via the web interface!

---

**Ready to start!** Follow the steps above to train your first NeuroVoice model. 🚀

