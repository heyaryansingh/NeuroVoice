#!/bin/bash
# Training script for NeuroVoice models

set -e

DISEASE=${1:-alzheimer}
EPOCHS=${2:-50}
BATCH_SIZE=${3:-16}
LR=${4:-1e-4}

echo "Training NeuroVoice model for $DISEASE"
echo "Epochs: $EPOCHS, Batch Size: $BATCH_SIZE, LR: $LR"

python src/training/train.py \
    --disease $DISEASE \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --lr $LR \
    --fusion crossmodal \
    --use_sam \
    --use_lookahead \
    --analyze_gradients \
    --use_tensorboard \
    --experiment_name ${DISEASE}_advanced \
    --use_gpu

echo "Training complete! Check outputs/models/ for saved models."

