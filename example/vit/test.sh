#!/bin/bash
# Test all trained models (both runs1 and runs2)

cd "$(dirname "$0")"

echo "=================================="
echo "Testing all trained models..."
echo "=================================="

# Test runs1
if [ -d "./runs1" ]; then
    echo ""
    echo "Testing models in runs1..."
    python -m src.test \
        --logdir ./runs1 \
        --model_config src/ViT-S-16.json \
        --data_root ./data \
        --batch_size 256 \
        --device cuda
fi

# Test runs2
if [ -d "./runs2" ]; then
    echo ""
    echo "Testing models in runs2..."
    python -m src.test \
        --logdir ./runs2 \
        --model_config src/ViT-S-16.json \
        --data_root ./data \
        --batch_size 256 \
        --device cuda
fi

echo ""
echo "=================================="
echo "All testing completed!"
echo "=================================="
