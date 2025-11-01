# NeuroVoice Web Interface

Full-stack web application for NeuroVoice disease detection system.

## Architecture

- **Backend**: FastAPI (Python) - REST API for model inference
- **Frontend**: Next.js (React/TypeScript) - Modern web UI, deployable on Vercel

## Backend Setup

```bash
cd web/backend
pip install -r requirements.txt
python main.py
```

The API will run on `http://localhost:8000`

## Frontend Setup

```bash
cd web/frontend
npm install
npm run dev
```

The frontend will run on `http://localhost:3000`

## Deployment

### Backend (FastAPI)
Deploy to any Python hosting service (Heroku, Railway, Render, etc.)

### Frontend (Next.js/Vercel)
1. Push frontend code to GitHub
2. Import project in Vercel
3. Set `NEXT_PUBLIC_API_URL` environment variable
4. Deploy!

## API Endpoints

- `GET /` - API info
- `GET /health` - Health check
- `GET /models` - List available models
- `POST /predict` - Make prediction (audio + optional video)
- `POST /analyze` - Analyze features without prediction
- `GET /stats/{disease}` - Get model statistics

## Features

- ✅ File upload (drag & drop)
- ✅ Real-time prediction
- ✅ Confidence scores
- ✅ Model statistics visualization
- ✅ Responsive design
- ✅ Dark mode support

