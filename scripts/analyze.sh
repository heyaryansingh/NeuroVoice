#!/bin/bash
# Analysis script for NeuroVoice

set -e

echo "Starting NeuroVoice Analysis Pipeline"

# Start backend API
echo "Starting backend API..."
cd web/backend
python main.py &
BACKEND_PID=$!
cd ../..

# Wait for backend to start
sleep 5

# Start frontend
echo "Starting frontend..."
cd web/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo "Backend running on http://localhost:8000"
echo "Frontend running on http://localhost:3000"
echo ""
echo "Press Ctrl+C to stop both services"

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID" EXIT
wait

