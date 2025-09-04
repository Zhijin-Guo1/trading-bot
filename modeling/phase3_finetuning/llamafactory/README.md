# LlamaFactory Fine-tuning for 8-K Analysis

## Overview
Fine-tune Qwen2.5-7B-Instruct model to predict stock price movements from SEC 8-K filings using LlamaFactory.

## Quick Start

### 1. Install Dependencies
```bash
pip install llamafactory[torch]
pip install transformers peft datasets
```

### 2. Prepare Data & Train
```bash
chmod +x train.sh
./train.sh
```

### 3. Evaluate Model
```bash
python3 evaluate.py --model_path ./outputs/qwen2.5-7b-lora
```

### 4. Run Inference
```bash
# Interactive mode
python3 inference.py --interactive

# Batch mode
python3 inference.py --input_file test_filings.json
```

## Files Structure

- `prepare_data.py` - Converts JSONL data to LlamaFactory format
- `config.yaml` - Training configuration 
- `train.sh` - Main training script
- `evaluate.py` - Model evaluation
- `inference.py` - Run predictions

## Configuration

### Model Settings
- Base model: Qwen/Qwen2.5-7B-Instruct
- Method: LoRA (rank=64, alpha=128)
- Precision: BF16
- Batch size: 2 (effective 16 with gradient accumulation)

### Data Format
Input JSONL format:
```json
{
  "text": "8-K filing text content...",
  "event_type": "7.01",
  "label": "STRONG_POSITIVE"
}
```

### Output Labels
- `STRONG_POSITIVE`: >3% increase expected
- `POSITIVE`: 0-3% increase expected  
- `NEGATIVE`: 0-3% decrease expected
- `STRONG_NEGATIVE`: >3% decrease expected

## Performance Tips

### GPU Memory Issues
Reduce batch size in `config.yaml`:
```yaml
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
```

### Faster Training
Enable Flash Attention:
```yaml
use_flash_attention_2: true
```

### Use Smaller Model
Change to Qwen2.5-1.5B:
```yaml
model_name_or_path: Qwen/Qwen2.5-1.5B-Instruct
```

## Monitoring

View training metrics:
```bash
tensorboard --logdir ./logs
```

## Results

Expected performance:
- Accuracy: 60-70% on validation set
- Training time: ~2-4 hours on single GPU
- Inference speed: ~5-10 predictions/second