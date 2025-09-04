# Phase 1: Machine Learning Models

## Overview
This folder contains traditional ML approaches for predicting 5-day stock returns from 8-K filing text data.

## Folder Structure

```
phase1_ml/
├── prepare_data.py                 # Data preparation (creates train/val/test splits)
├── prepare_data_nextday.py         # Next-day entry data preparation
├── baseline_analysis.py            # Baseline performance analysis
├── data/                           # Primary dataset 
│   ├── train.csv                   # Training data
│   ├── val.csv                     # Validation data  
│   └── test.csv                    # Test data
├── next_day_data/                  # Next-day entry dataset
│   ├── train.csv
│   ├── val.csv
│   └── test.csv
├── binary_classification/          # Binary UP/DOWN prediction models
│   ├── tfidf_logreg/              # TF-IDF + Logistic Regression
│   │   ├── train_tfidf_logreg.py  # Main training script
│   │   ├── train_sector_models.py # Sector-specific models
│   │   └── run_experiments.py     # Batch experiments
│   ├── embeddings/                # Embedding-based approaches
│   │   ├── train_embedding_models.py     # Multiple embedding methods
│   │   ├── train_embedding_simple.py     # Simplified version
│   │   ├── finetune_transformer.py       # Transformer fine-tuning
│   │   └── test_embeddings_sample.py     # Testing script
│   ├── qwen_finetuning/           # Qwen LLM fine-tuning
│   │   ├── train_qwen.py          # Full Qwen training
│   │   ├── train_qwen_fast.py     # Fast training (smaller model)
│   │   └── inference.py           # Inference script
│   └── text_embeddings_gpu.py     # GPU-accelerated embeddings
└── regression/                     # Continuous return prediction
    ├── finbert_regression/         # FinBERT-based regression
    │   └── train_regression.py    # Return magnitude prediction
    └── text_features/              # Feature-based regression
        ├── text_only_basic.py      # Basic text features
        └── text_only_comprehensive.py # Comprehensive analysis

```

## Task Types

### Binary Classification
**Goal**: Predict if stock will go UP (>0%) or DOWN (<0%) after 5 days

**Best Performance**: 
- TF-IDF + LogReg: 53.14% accuracy (baseline: 52%)
- Sector-specific: 54-55% for Technology sector

### Three-Class Classification  
**Goal**: Predict UP (>1%), STAY (±1%), or DOWN (<-1%)

**Baseline**: ~50% (majority class)

### Regression
**Goal**: Predict exact `adjusted_return_pct` value

**Performance**:
- Return prediction R²: 0.02-0.05
- Volatility prediction R²: 0.08-0.12

## Data Sources

### Primary Dataset (`data/`)
- Created: September 1, 2024
- Contains GPT summaries merged with price data
- Features: summary, items_present, sector, momentum, returns

### Next-Day Dataset (`next_day_data/`)
- Created: August 31, 2024
- Adjusted for next-day entry timing
- Used by some binary classification models

## Key Findings

1. **Weak Text Signal**: Text alone provides limited predictive power for 5-day returns
2. **Volatility > Direction**: Text better predicts magnitude than direction
3. **Sector Effects**: Technology sector shows stronger signals
4. **Item-Specific Signals**: Different 8-K items have varying predictive power

## Running the Models

### Binary Classification
```bash
# TF-IDF + Logistic Regression
cd binary_classification/tfidf_logreg
python train_tfidf_logreg.py

# Sector-specific models
python train_sector_models.py

# Embeddings (requires GPU for practical speed)
cd ../embeddings
python train_embedding_models.py
```

### Regression
```bash
# FinBERT regression
cd regression/finbert_regression
python train_regression.py

# Text feature analysis
cd ../text_features
python text_only_comprehensive.py
```

## Dependencies
- pandas, numpy, scikit-learn
- transformers, torch (for deep learning models)
- sentence-transformers (for embeddings)
- peft (for Qwen fine-tuning)

## Notes
- Most models use `data/` folder (train.csv, val.csv, test.csv)
- Some older experiments use `next_day_data/`
- GPU recommended for embedding and transformer models
- All paths have been updated for the new folder structure