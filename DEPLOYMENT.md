# NeuroVoice Deployment Guide

Complete guide for deploying NeuroVoice web interface to Vercel and backend services.

## 🚀 Quick Deployment

### Frontend (Vercel) - Recommended

1. **Push to GitHub**
   ```bash
   git add .
   git commit -m "Deploy NeuroVoice"
   git push origin main
   ```

2. **Import to Vercel**
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Set root directory to `web/frontend`
   - Add environment variable: `NEXT_PUBLIC_API_URL` = your backend URL

3. **Deploy!**
   - Vercel auto-detects Next.js and deploys
   - Your site will be live at `your-project.vercel.app`

### Backend (FastAPI) - Multiple Options

#### Option 1: Railway (Easiest)
1. Go to [railway.app](https://railway.app)
2. New Project → Deploy from GitHub
3. Select your repo
4. Set root directory: `web/backend`
5. Railway auto-detects Python and deploys
6. Get your backend URL and add to frontend env var

#### Option 2: Render
1. Go to [render.com](https://render.com)
2. New → Web Service
3. Connect GitHub repo
4. Settings:
   - Root Directory: `web/backend`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `uvicorn main:app --host 0.0.0.0 --port $PORT`
5. Deploy!

#### Option 3: Heroku
```bash
cd web/backend
heroku create neurovoice-api
heroku config:set PORT=8000
git subtree push --prefix web/backend heroku main
```

## 📝 Environment Variables

### Frontend (Vercel)
- `NEXT_PUBLIC_API_URL`: Backend API URL (e.g., `https://your-api.railway.app`)

### Backend (Railway/Render)
- `PORT`: Server port (usually auto-set)
- Optional: `MODEL_DIR`: Path to models directory

## 🔧 Production Checklist

- [ ] Set `NEXT_PUBLIC_API_URL` in Vercel
- [ ] Backend deployed and accessible
- [ ] Models uploaded to backend
- [ ] CORS configured correctly
- [ ] Test API endpoints
- [ ] Test file uploads
- [ ] Monitor logs for errors

## 📊 Monitoring

### Vercel Analytics
- Built-in analytics dashboard
- Monitor performance and errors

### Backend Logs
- Railway: Dashboard → Logs
- Render: Dashboard → Logs
- Heroku: `heroku logs --tail`

## 🔒 Security Notes

1. **API Rate Limiting**: Consider adding rate limiting to backend
2. **CORS**: Configure CORS properly in production (not `allow_origins=["*"]`)
3. **File Size Limits**: Configure max file size for uploads
4. **API Keys**: Keep sensitive keys in environment variables

## 🐛 Troubleshooting

### Frontend can't connect to backend
- Check `NEXT_PUBLIC_API_URL` is set correctly
- Verify backend is running and accessible
- Check CORS settings

### File upload fails
- Check file size limits
- Verify backend can handle multipart/form-data
- Check backend logs for errors

### Models not found
- Upload models to backend storage
- Set correct `MODEL_DIR` path
- Check file permissions

