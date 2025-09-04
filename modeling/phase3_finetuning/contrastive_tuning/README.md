# Contrastive Learning Pipeline for 8-K Filing Analysis

## 🎯 Overview

This pipeline implements **comparative fine-tuning** for language models to compare and rank 8-K filings based on expected stock performance. The model learns to identify which filing in a pair will perform better - a task that's often easier and more reliable than absolute prediction.

**Important Note**: Despite the name "contrastive", this uses standard language model fine-tuning with cross-entropy loss for next-token prediction, NOT contrastive losses like InfoNCE. The "contrastive" aspect refers to the comparative nature of the training data (pairs of filings).

## 📊 Key Innovation

**Why Comparative Learning Works:**
- **Relative comparison is easier** than absolute prediction
- **Reduces noise** by focusing on differential signals
- **Better calibration** through pairwise comparisons
- **Natural for pair trading strategies**
- **Learns from standard LM objective** (next-token prediction)

## 🚀 Complete Pipeline

### Step 1: Prepare Data on Local Machine

```bash
cd modeling/phase3_finetuning/contrastive_tuning

# Generate contrastive pairs from filtered profitable data
python prepare_contrastive_data.py
```

This creates:
- `data/train.json` - Training data with contrastive pairs
- `data/val.json` - Validation data
- `data/test.json` - Test data for evaluation
- `data/test_metadata.json` - Metadata for calculating metrics
- `data/dataset_info.json` - LlamaFactory configuration

**Data Statistics:**
- ~3,000 total samples (mix of pairs and singles)
- 6 profitable event types (7.01, 9.01, 5.07, 5.02, 1.01, 2.02)
- Only high-signal movements (>3% absolute)
- 70/15/15 train/val/test split

### Step 2: Transfer to GPU Server

```bash
# Create archive
tar -czf contrastive_tuning.tar.gz contrastive_tuning/

# Transfer to GPU server
scp contrastive_tuning.tar.gz user@gpu-server:/path/to/workspace/

# On GPU server, extract
ssh user@gpu-server
cd /path/to/workspace
tar -xzf contrastive_tuning.tar.gz
cd contrastive_tuning
```

### Step 3: Setup Environment on GPU Server

```bash
# Create conda environment
conda create -n contrastive python=3.10
conda activate contrastive

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers datasets peft accelerate
pip install llamafactory[torch,metrics]
pip install pandas numpy tqdm
```

### Step 4: Train Model

```bash
# Make training script executable
chmod +x train_contrastive.sh

# Run training (adjust config if needed for your GPU)
./train_contrastive.sh
```

**Training Configuration (RTX 3090 Optimized):**
- Model: Qwen2.5-7B-Instruct (or Llama-3.2-3B for smaller option)
- LoRA rank: 32
- Batch size: 1 with gradient accumulation of 8
- Flash Attention 2 enabled
- Mixed precision (bf16)
- ~3 epochs
- Expected training time: 4-6 hours

### Step 5: Evaluate Model

See detailed evaluation instructions in the Evaluation Pipeline section below.

## 📁 File Structure

```
contrastive_tuning/
├── Data Preparation
│   ├── prepare_contrastive_data.py  # Generate training data from filtered 8-K
│   └── data/                         # Generated datasets
│       ├── train.json                # 2097 training samples
│       ├── val.json                  # 449 validation samples
│       ├── test.json                 # 451 test samples
│       └── test_metadata.json        # Ground truth for evaluation
│
├── Training Scripts
│   ├── config_contrastive.yaml      # LlamaFactory configuration
│   ├── train_contrastive.sh         # Main training script (with HF mirror)
│   ├── train_simple.sh              # Simplified training (minimal params)
│   └── train_direct.py              # Python alternative to shell script
│
├── Evaluation Scripts
│   ├── simple_test.py               # Quick verification (run first!)
│   ├── evaluate_full.py             # Complete evaluation with metrics
│   ├── test_custom.py               # Interactive testing with own examples
│   ├── inference_contrastive.py     # Original evaluation script
│   └── inference_single_filing.py   # Production-style single filing prediction
│
├── Model Output
│   └── output_contrastive/          # After training completes
│       ├── adapter_model.bin        # LoRA weights (~50-200MB)
│       ├── adapter_config.json      # Configuration
│       └── trainer_log.jsonl        # Training logs
│
└── Documentation
    ├── README.md                     # This file
    └── training_explanation.md       # Technical details on loss function
```

## 🎯 Task Formats

### Format 1: Contrastive Pairs
```json
{
  "instruction": "Compare these two Regulation FD Disclosure filings...\nFiling A: [content]\nFiling B: [content]\nWhich filing resulted in better performance?",
  "output": "Filing A resulted in better performance.\nAnalysis: Filing A shows stronger earnings beat..."
}
```

### Format 2: Single Filing Prediction
```json
{
  "instruction": "Analyze this Financial Statements filing and predict market impact...",
  "output": "Prediction: STRONG_POSITIVE\nReasoning: Revenue exceeded guidance..."
}
```

## 📋 Evaluation Pipeline

### After Training Completes

Once training is finished, you'll have an `output_contrastive/` folder with:
- `adapter_model.bin` - LoRA weights (50-200MB)
- `adapter_config.json` - LoRA configuration
- `trainer_log.jsonl` - Training logs
- `training_loss.png` - Loss plot (if enabled)

### Evaluation Scripts Overview

Three evaluation scripts are provided for different purposes:

#### 1. `simple_test.py` - Quick Verification (Run First!)
Tests if the model loaded and learned correctly with 2 simple examples.

```bash
python simple_test.py

# Expected output:
# ✅ Test 1 PASSED: Model correctly identified Filing A as better
# ✅ Test 2 PASSED: Model correctly predicted positive impact
# ✅ ALL TESTS PASSED! Model is working correctly.
```

#### 2. `evaluate_full.py` - Complete Evaluation
Evaluates the model on the entire test set with comprehensive metrics.

```bash
# Quick test with 20 samples (run this first)
python evaluate_full.py --quick

# Full evaluation with all 451 test samples
python evaluate_full.py

# Output files:
# - pairs_evaluation_results.csv (detailed pair comparisons)
# - singles_evaluation_results.csv (single filing predictions)
```

#### 3. `test_custom.py` - Interactive Testing
Test the model with your own examples.

```bash
# Interactive mode
python test_custom.py
# Commands available: example, single, compare, quit

# Command line mode
python test_custom.py --mode single --text "Apple beat earnings by 15%"
```

### Step-by-Step Evaluation Process

```bash
# 1. First, verify the model works
python simple_test.py

# 2. If tests pass, run quick evaluation (20 samples, ~2 min)
python evaluate_full.py --quick

# 3. If results look good, run full evaluation (451 samples, ~15 min)
python evaluate_full.py

# 4. Test your own examples interactively
python test_custom.py

# 5. Download results to local machine (from local terminal)
scp user@server:/path/contrastive_tuning/*_results.csv ./
```

### Understanding the Results

**Pairs Comparison Metrics:**
- **Overall accuracy**: % of correct "A vs B" predictions
- **High confidence accuracy**: Performance when model is confident
- **Trading metrics**: Sharpe ratio, win rate for pair trading

**Single Filing Metrics:**
- **Category accuracy**: How well it predicts POSITIVE/NEGATIVE
- **Confusion matrix**: Distribution of predictions

### Expected Performance Metrics

**Good Performance Indicators:**
- ✅ Pair comparison accuracy > 60% (vs 50% random)
- ✅ High confidence (>0.7) accuracy > 70%
- ✅ Sharpe ratio > 1.0
- ✅ Win rate > 55%
- ✅ Clear difference between high/low confidence predictions

**Needs Improvement:**
- ⚠️ Accuracy close to 50% (random guessing)
- ⚠️ All predictions have same confidence
- ⚠️ Negative Sharpe ratio
- ⚠️ No correlation between confidence and accuracy

### Troubleshooting Evaluation

**If you get connection/download errors:**
```bash
# Use HF mirror
export HF_ENDPOINT=https://hf-mirror.com

# Force offline mode (use cached model)
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
python simple_test.py
```

**If out of memory during evaluation:**
```bash
# Use CPU only (slower but works)
export CUDA_VISIBLE_DEVICES=""
python simple_test.py

# Or reduce batch size in the script
```

**If model not found:**
```bash
# Check training output exists
ls -la output_contrastive/
# Should see: adapter_model.bin, adapter_config.json
```

### Production Use

For real-world prediction of single filings:

```bash
# Use the single filing predictor
python inference_single_filing.py \
    --model_path ./output_contrastive \
    --filing_text "Your 8-K text here" \
    --event_type 7.01 \
    --ticker AAPL
```

This script implements three strategies:
1. Direct prediction from single filing
2. Comparison against neutral baseline
3. Historical pattern matching
4. Ensemble of all three for final prediction

## 📈 Expected Performance

Based on the contrastive learning approach:

**Classification Metrics:**
- Pairwise accuracy: 65-75% (better than random 50%)
- High-confidence accuracy (>0.7): 70-80%
- Large differential accuracy (>10% diff): 75-85%

**Trading Metrics (Pair Trading):**
- Expected Sharpe: 1.5-2.5
- Win rate: 55-65%
- Only trades high-confidence pairs

## 🔧 Customization Options

### Use Smaller Model (for testing)
Edit `config_contrastive.yaml`:
```yaml
model_name_or_path: meta-llama/Llama-3.2-3B-Instruct  # 3B instead of 7B
```

### Adjust for Different GPU Memory
```yaml
# For 16GB GPU (e.g., V100)
per_device_train_batch_size: 1
gradient_accumulation_steps: 16
lora_rank: 16

# For 40GB GPU (e.g., A100)
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
lora_rank: 64
```

### Filter for Specific Event Types
In `prepare_contrastive_data.py`, modify:
```python
PROFITABLE_EVENTS = {
    "7.01": {...},  # Keep only events you want
    "9.01": {...}
}
```

## 🚨 Common Issues & Solutions

### Out of Memory (OOM)
- Reduce `cutoff_len` in config (e.g., 1024 instead of 2048)
- Reduce `lora_rank` (e.g., 16 instead of 32)
- Enable DeepSpeed: set `deepspeed: ds_z3_config.json`

### Slow Training
- Ensure Flash Attention is working: `flash_attn: fa2`
- Check GPU utilization: `nvidia-smi`
- Reduce logging frequency: `logging_steps: 50`

### Poor Performance
- Increase training epochs: `num_train_epochs: 5`
- Adjust learning rate: try `5e-5` or `2e-4`
- Ensure data quality by checking generated JSON files

## 📊 Monitoring Training

```bash
# Watch GPU usage
watch -n 1 nvidia-smi

# Monitor training logs
tail -f output_contrastive/trainer_log.txt

# TensorBoard visualization
tensorboard --logdir output_contrastive
```

## 🎯 Next Steps After Training

1. **Evaluate on different thresholds** - Try confidence levels 0.6, 0.7, 0.8
2. **Backtest pair trading strategy** - Long winner, short loser
3. **Ensemble with Phase 1 ML** - Combine predictions
4. **Deploy for real-time analysis** - Set up inference API

## 📝 Citation

This approach is inspired by:
- SimCSE contrastive learning
- RankNet for learning to rank
- Pair trading strategies in quantitative finance

---

For questions or issues, check the training logs first, then verify data quality in the generated JSON files.