#!/bin/bash

# Evaluation script for fine-tuned Qwen2.5-1.5B model
echo "=========================================="
echo "Starting Model Evaluation"
echo "=========================================="

# Set HuggingFace mirror if needed
export HF_ENDPOINT=https://hf-mirror.com

# GPU memory optimization
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:128

# Suppress transformers warnings about unused generation flags
export TRANSFORMERS_VERBOSITY=error

# Run evaluation on validation set
echo ""
echo "Evaluating on validation dataset..."
python evaluate.py \
    --model_path ./outputs/qwen2.5-1.5b-lora \
    --data_path data/val.json \
    --output_path evaluation_results.json

# Check if evaluation succeeded
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Evaluation completed successfully!"
    echo "Results saved to: evaluation_results.json"
else
    echo ""
    echo "❌ Evaluation failed. Check the error messages above."
    exit 1
fi

# Run event-specific analysis if script exists
if [ -f "analyze_by_event.py" ]; then
    echo ""
    echo "Running event-specific analysis..."
    python analyze_by_event.py \
        --results_path evaluation_results.json \
        --data_path data/val.json
fi

# Compare with baseline if results exist
if [ -f "../phase2_llm_api/llm_results.json" ]; then
    echo ""
    echo "Comparing with GPT-3.5 baseline..."
    python compare_results.py \
        --finetuned evaluation_results.json \
        --baseline ../phase2_llm_api/llm_results.json
fi

echo ""
echo "=========================================="
echo "Evaluation Complete"
echo "=========================================="