#!/usr/bin/env python3
"""
Embedding Model Comparison for Volatility Prediction
=====================================================
Compares various text embedding models (FinBERT, MiniLM, MPNet, etc.) 
combined with LLM features for volatility prediction and classification.

Both tasks use MLP neural networks for fair comparison:
- Volatility Regression: MLP Regressor
- High Volatility Classification: MLP Classifier
"""

import os
import pandas as pd
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
)
from scipy.stats import pearsonr, spearmanr
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# ================================================================================
# CONFIGURATION
# ================================================================================

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Data path - relative to script location  
DATA_PATH = os.path.join(SCRIPT_DIR, '../../../llm_features/filtered_data/')

# Output directory for results
OUTPUT_DIR = os.path.join(SCRIPT_DIR, 'results')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Model parameters
VOLATILITY_THRESHOLD = 75  # Percentile for high volatility
RANDOM_SEED = 42

# MLP parameters for regression
MLP_REGRESSION_PARAMS = {
    'hidden_layer_sizes': (256, 128),
    'activation': 'relu',
    'solver': 'adam',
    'alpha': 0.001,
    'batch_size': 128,
    'learning_rate': 'adaptive',
    'max_iter': 500,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'random_state': RANDOM_SEED,
    'verbose': False
}

# Logistic Regression parameters for classification
LOGREG_PARAMS = {
    'C': 0.1,  # Inverse regularization strength
    'penalty': 'l2',
    'solver': 'liblinear',
    'max_iter': 1000,
    'class_weight': 'balanced',
    'random_state': RANDOM_SEED,
    'verbose': 0
}

class EmbeddingComparison:
    """Main class for comparing different embedding models"""
    
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.label_encoders = {}
        self.feature_scaler = None
        self.results = []
        self.embedding_models = {}
        
        # Report GPU status
        if self.device == 'cuda':
            print(f"✅ GPU Available: {torch.cuda.get_device_name(0)}")
            print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        else:
            print("⚠️ No GPU detected, using CPU (will be slower)")
    
    def load_data(self):
        """Load filtered datasets with LLM features"""
        print("\n" + "="*70)
        print("LOADING DATA")
        print("="*70)
        
        # Use filtered data (non-routine filings) with LLM features
        train_df = pd.read_csv(os.path.join(DATA_PATH, 'filtered_train.csv'))
        val_df = pd.read_csv(os.path.join(DATA_PATH, 'filtered_val.csv'))
        test_df = pd.read_csv(os.path.join(DATA_PATH, 'filtered_test.csv'))
        
        print(f"Data loaded from: {DATA_PATH}")
        print(f"  Train: {len(train_df):,} samples")
        print(f"  Val:   {len(val_df):,} samples")
        print(f"  Test:  {len(test_df):,} samples")
        
        return train_df, val_df, test_df
    
    def prepare_targets(self, train_df, val_df, test_df):
        """Prepare regression and classification targets"""
        print("\n" + "="*70)
        print("PREPARING TARGETS")
        print("="*70)
        
        # Regression target: volatility (absolute returns)
        y_train_reg = np.abs(train_df['adjusted_return_pct'].values)
        y_val_reg = np.abs(val_df['adjusted_return_pct'].values)
        y_test_reg = np.abs(test_df['adjusted_return_pct'].values)
        
        # Classification target: high volatility (top 25%)
        threshold = np.percentile(np.abs(train_df['adjusted_return_pct'].values), VOLATILITY_THRESHOLD)
        y_train_clf = (np.abs(train_df['adjusted_return_pct'].values) > threshold).astype(int)
        y_val_clf = (np.abs(val_df['adjusted_return_pct'].values) > threshold).astype(int)
        y_test_clf = (np.abs(test_df['adjusted_return_pct'].values) > threshold).astype(int)
        
        print(f"Volatility statistics:")
        print(f"  Train - Mean: {y_train_reg.mean():.3f}%, Std: {y_train_reg.std():.3f}%")
        print(f"  Test  - Mean: {y_test_reg.mean():.3f}%, Std: {y_test_reg.std():.3f}%")
        print(f"\nHigh volatility threshold (top 25%): {threshold:.3f}%")
        print(f"  Train high vol: {y_train_clf.mean():.1%}")
        print(f"  Test high vol:  {y_test_clf.mean():.1%}")
        
        return {
            'regression': (y_train_reg, y_val_reg, y_test_reg),
            'classification': (y_train_clf, y_val_clf, y_test_clf),
            'threshold': threshold
        }
    
    def extract_llm_features(self, df, is_training=False):
        """Extract LLM features (same as train_combined_regression.py)"""
        
        # Define feature groups
        numeric_features = [
            'salience_score', 'volatility_score',
            'tone_score', 'tone_confidence',
            'novelty_score', 'sub_topic_confidence'
        ]
        
        categorical_features = [
            'sub_topic', 'impact_magnitude', 'time_horizon', 
            'uncertainty_level', 'business_impact', 'outcome_clarity',
            'event_type', 'expected_reaction', 'surprise_level',
            'net_risk_assessment', 'action_orientation', 'surprise_factor'
        ]
        
        boolean_features = ['is_material', 'quantitative_support']
        
        market_features = [
            'momentum_7d', 'momentum_30d', 'momentum_90d', 
            'momentum_365d', 'vix_level'
        ]
        
        features = []
        feature_names = []
        
        # Process numeric features
        numeric_data = []
        for feat in numeric_features:
            if feat in df.columns:
                numeric_data.append(df[feat].fillna(0).values.reshape(-1, 1))
                feature_names.append(f'llm_{feat}')
        
        if numeric_data:
            numeric_array = np.hstack(numeric_data)
            features.append(numeric_array)
        
        # Process categorical features
        for feat in categorical_features:
            if feat in df.columns:
                if is_training:
                    if feat not in self.label_encoders:
                        self.label_encoders[feat] = LabelEncoder()
                        self.label_encoders[feat].fit(df[feat].fillna('unknown').astype(str))
                
                if feat in self.label_encoders:
                    encoded = []
                    for val in df[feat].fillna('unknown').astype(str):
                        if val in self.label_encoders[feat].classes_:
                            encoded.append(self.label_encoders[feat].transform([val])[0])
                        else:
                            encoded.append(0)
                    features.append(np.array(encoded).reshape(-1, 1))
                    feature_names.append(f'llm_{feat}_encoded')
        
        # Process boolean features
        for feat in boolean_features:
            if feat in df.columns:
                features.append(df[feat].fillna(False).astype(int).values.reshape(-1, 1))
                feature_names.append(f'llm_{feat}')
        
        # Process market features
        market_data = []
        for feat in market_features:
            if feat in df.columns:
                market_data.append(df[feat].fillna(0).values.reshape(-1, 1))
                feature_names.append(f'market_{feat}')
        
        if market_data:
            market_array = np.hstack(market_data)
            features.append(market_array)
        
        if is_training:
            self.llm_feature_names = feature_names
        
        if features:
            combined_features = np.hstack(features)
            return combined_features
        else:
            return np.zeros((len(df), 1))
    
    def load_embedding_model(self, model_name):
        """Load and cache embedding model"""
        if model_name not in self.embedding_models:
            print(f"  Loading {model_name}...")
            
            # Check if it's a sentence-transformers model
            if 'sentence-transformers' in model_name or 'BAAI/bge' in model_name or 'intfloat/e5' in model_name:
                model = SentenceTransformer(model_name, device=self.device)
                self.embedding_models[model_name] = ('sentence_transformer', None, model)
            
            # Handle specific model architectures
            elif 'finbert' in model_name.lower():
                if 'ProsusAI' in model_name:
                    tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
                    model = AutoModel.from_pretrained('ProsusAI/finbert').to(self.device)
                else:  # yiyanghkust/finbert-tone
                    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
                    model = AutoModel.from_pretrained('yiyanghkust/finbert-tone').to(self.device)
                self.embedding_models[model_name] = ('bert', tokenizer, model)
                
            elif 'bert' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(self.device)
                self.embedding_models[model_name] = ('bert', tokenizer, model)
                
            elif 'roberta' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(self.device)
                self.embedding_models[model_name] = ('bert', tokenizer, model)
                
            elif 'deberta' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModel.from_pretrained(model_name).to(self.device)
                self.embedding_models[model_name] = ('bert', tokenizer, model)
                
            else:
                # Try to load as a general transformer model
                try:
                    tokenizer = AutoTokenizer.from_pretrained(model_name)
                    model = AutoModel.from_pretrained(model_name).to(self.device)
                    self.embedding_models[model_name] = ('bert', tokenizer, model)
                except:
                    # Fall back to sentence transformer
                    model = SentenceTransformer(model_name, device=self.device)
                    self.embedding_models[model_name] = ('sentence_transformer', None, model)
        
        return self.embedding_models[model_name]
    
    def extract_embeddings(self, texts, model_name, batch_size=32):
        """Extract embeddings using specified model"""
        model_type, tokenizer, model = self.load_embedding_model(model_name)
        embeddings = []
        
        # Convert to list if pandas Series
        if hasattr(texts, 'tolist'):
            texts = texts.tolist()
        
        if model_type == 'sentence_transformer':
            # Process with sentence transformers
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i+batch_size]
                # Truncate long texts
                batch = [str(text)[:2048] for text in batch]
                
                batch_embeddings = model.encode(
                    batch, 
                    show_progress_bar=False,
                    convert_to_numpy=True
                )
                embeddings.extend(batch_embeddings)
                
                if i % 1000 == 0 and i > 0:
                    print(f"    Processed {i}/{len(texts)} texts...")
        
        else:  # BERT-style models
            model.eval()
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch = texts[i:i+batch_size]
                    batch = [str(text)[:512] for text in batch]
                    
                    # Tokenize
                    inputs = tokenizer(
                        batch,
                        padding=True,
                        truncation=True,
                        max_length=512,
                        return_tensors='pt'
                    ).to(self.device)
                    
                    # Get embeddings
                    outputs = model(**inputs)
                    
                    # Use CLS token embeddings
                    batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                    embeddings.extend(batch_embeddings)
                    
                    if i % 1000 == 0 and i > 0:
                        print(f"    Processed {i}/{len(texts)} texts...")
        
        return np.array(embeddings)
    
    def create_combined_features(self, embeddings, llm_features):
        """Combine embeddings with LLM features"""
        # Scale LLM features
        if self.feature_scaler is None:
            self.feature_scaler = StandardScaler()
            llm_scaled = self.feature_scaler.fit_transform(llm_features)
        else:
            llm_scaled = self.feature_scaler.transform(llm_features)
        
        # Combine embeddings with scaled LLM features
        combined = np.hstack([embeddings, llm_scaled])
        return combined
    
    def train_and_evaluate(self, X_train, X_val, X_test, y_train, y_val, y_test, 
                          task_type, model_name):
        """Train and evaluate model for given task"""
        
        if task_type == 'regression':
            # MLP Regressor for volatility prediction
            model = MLPRegressor(**MLP_REGRESSION_PARAMS)
            
            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_val = model.predict(X_val)
            y_pred_test = model.predict(X_test)
            
            # Ensure non-negative for volatility
            y_pred_test = np.maximum(y_pred_test, 0)
            
            # Calculate metrics
            pearson_r, pearson_p = pearsonr(y_test, y_pred_test)
            spearman_r, _ = spearmanr(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            r2 = r2_score(y_test, y_pred_test)
            
            # Overfitting check
            train_corr, _ = pearsonr(y_train, y_pred_train)
            overfit_gap = train_corr - pearson_r
            
            return {
                'embedding_model': model_name,
                'task': 'volatility_regression',
                'test_correlation': pearson_r,
                'test_spearman': spearman_r,
                'test_r2': r2,
                'test_mae': mae,
                'test_rmse': rmse,
                'train_correlation': train_corr,
                'overfit_gap': overfit_gap,
                'train_time': train_time,
                'p_value': pearson_p
            }
            
        else:  # classification
            # Logistic Regression for high volatility classification
            model = LogisticRegression(**LOGREG_PARAMS)
            
            # Train
            start_time = time.time()
            model.fit(X_train, y_train)
            train_time = time.time() - start_time
            
            # Predict
            y_pred_test = model.predict(X_test)
            y_proba_test = model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred_test)
            precision = precision_score(y_test, y_pred_test)
            recall = recall_score(y_test, y_pred_test)
            f1 = f1_score(y_test, y_pred_test)
            auc = roc_auc_score(y_test, y_proba_test)
            cm = confusion_matrix(y_test, y_pred_test)
            
            return {
                'embedding_model': model_name,
                'task': 'high_vol_classification',
                'test_accuracy': accuracy,
                'test_precision': precision,
                'test_recall': recall,
                'test_f1': f1,
                'test_auc': auc,
                'confusion_matrix': cm.tolist(),
                'train_time': train_time
            }
    
    def run_experiments(self):
        """Run all experiments comparing different embeddings"""
        
        # Load data
        train_df, val_df, test_df = self.load_data()
        
        # Prepare targets
        targets = self.prepare_targets(train_df, val_df, test_df)
        y_train_reg, y_val_reg, y_test_reg = targets['regression']
        y_train_clf, y_val_clf, y_test_clf = targets['classification']
        
        # Extract LLM features
        print("\n" + "="*70)
        print("EXTRACTING LLM FEATURES")
        print("="*70)
        
        X_train_llm = self.extract_llm_features(train_df, is_training=True)
        X_val_llm = self.extract_llm_features(val_df)
        X_test_llm = self.extract_llm_features(test_df)
        print(f"LLM features shape: {X_train_llm.shape}")
        
        # Define embedding models to test
        embedding_models = [
            # Sentence-BERT variants
            ('sentence-transformers/all-MiniLM-L6-v2', 'SBERT-MiniLM'),
            ('sentence-transformers/all-mpnet-base-v2', 'SBERT-MPNet'),
            ('sentence-transformers/all-distilroberta-v1', 'SBERT-DistilRoBERTa'),
            
            # Original BERT models
            ('bert-base-uncased', 'BERT-base'),
            ('distilbert-base-uncased', 'DistilBERT'),
            
            # Financial domain-specific
            ('yiyanghkust/finbert-tone', 'FinBERT'),
            ('ProsusAI/finbert', 'FinBERT-Prosus'),
            
            # RoBERTa variants
            ('roberta-base', 'RoBERTa-base'),
            
            # State-of-the-art embeddings
            ('BAAI/bge-base-en-v1.5', 'BGE-base'),
            ('intfloat/e5-base-v2', 'E5-base'),
            
            # Uncomment these for more comprehensive testing (requires more GPU memory):
            # ('sentence-transformers/all-roberta-large-v1', 'SBERT-RoBERTa-large'),
            # ('microsoft/deberta-v3-base', 'DeBERTa-v3'),
            # ('BAAI/bge-large-en-v1.5', 'BGE-large'),
        ]
        
        # Store all results
        all_results = []
        
        print("\n" + "="*70)
        print("TESTING EMBEDDING MODELS")
        print("="*70)
        
        for model_path, model_name in embedding_models:
            print(f"\n{'='*50}")
            print(f"Testing: {model_name}")
            print('='*50)
            
            # Extract embeddings
            print(f"Extracting embeddings...")
            start_time = time.time()
            
            # Determine batch size based on model and GPU
            batch_size = 64 if self.device == 'cuda' else 16
            if 'large' in model_path.lower():
                batch_size = batch_size // 2  # Smaller batch for large models
            
            X_train_emb = self.extract_embeddings(
                train_df['summary'].fillna(''), model_path, batch_size
            )
            X_val_emb = self.extract_embeddings(
                val_df['summary'].fillna(''), model_path, batch_size
            )
            X_test_emb = self.extract_embeddings(
                test_df['summary'].fillna(''), model_path, batch_size
            )
            
            embedding_time = time.time() - start_time
            print(f"  Embedding extraction took {embedding_time:.1f}s")
            print(f"  Embedding shape: {X_train_emb.shape}")
            
            # Combine with LLM features
            print(f"Combining with LLM features...")
            X_train_combined = self.create_combined_features(X_train_emb, X_train_llm)
            X_val_combined = self.create_combined_features(X_val_emb, X_val_llm)
            X_test_combined = self.create_combined_features(X_test_emb, X_test_llm)
            print(f"  Combined shape: {X_train_combined.shape}")
            
            # Test 1: Volatility Regression (MLP)
            print(f"\n1. Volatility Regression (MLP)...")
            reg_results = self.train_and_evaluate(
                X_train_combined, X_val_combined, X_test_combined,
                y_train_reg, y_val_reg, y_test_reg,
                'regression', model_name
            )
            reg_results['embedding_time'] = embedding_time
            all_results.append(reg_results)
            
            print(f"   Correlation: {reg_results['test_correlation']:.4f}")
            print(f"   R²: {reg_results['test_r2']:.4f}")
            print(f"   MAE: {reg_results['test_mae']:.3f}%")
            
            # Test 2: High Volatility Classification (Logistic Regression)
            print(f"\n2. High Volatility Classification (LogReg)...")
            clf_results = self.train_and_evaluate(
                X_train_combined, X_val_combined, X_test_combined,
                y_train_clf, y_val_clf, y_test_clf,
                'classification', model_name
            )
            clf_results['embedding_time'] = embedding_time
            all_results.append(clf_results)
            
            print(f"   Accuracy: {clf_results['test_accuracy']:.1%}")
            print(f"   F1 Score: {clf_results['test_f1']:.3f}")
            print(f"   AUC: {clf_results['test_auc']:.3f}")
            
            # Also test embeddings only (without LLM features) for comparison
            print(f"\n3. Testing embeddings only (no LLM features)...")
            
            # Regression with embeddings only
            reg_only_results = self.train_and_evaluate(
                X_train_emb, X_val_emb, X_test_emb,
                y_train_reg, y_val_reg, y_test_reg,
                'regression', f"{model_name}_only"
            )
            reg_only_results['features'] = 'embeddings_only'
            all_results.append(reg_only_results)
            
            # Classification with embeddings only
            clf_only_results = self.train_and_evaluate(
                X_train_emb, X_val_emb, X_test_emb,
                y_train_clf, y_val_clf, y_test_clf,
                'classification', f"{model_name}_only"
            )
            clf_only_results['features'] = 'embeddings_only'
            all_results.append(clf_only_results)
            
            print(f"   Embeddings only - Regression Corr: {reg_only_results['test_correlation']:.4f}")
            print(f"   Embeddings only - Classification F1: {clf_only_results['test_f1']:.3f}")
        
        self.results = pd.DataFrame(all_results)
        return self.results
    
    def create_comparison_plots(self):
        """Create comprehensive comparison plots"""
        
        print("\n" + "="*70)
        print("CREATING VISUALIZATIONS")
        print("="*70)
        
        # Separate results by task and feature type
        reg_combined = self.results[
            (self.results['task'] == 'volatility_regression') & 
            (~self.results['embedding_model'].str.contains('_only'))
        ]
        reg_only = self.results[
            (self.results['task'] == 'volatility_regression') & 
            (self.results['embedding_model'].str.contains('_only'))
        ]
        clf_combined = self.results[
            (self.results['task'] == 'high_vol_classification') & 
            (~self.results['embedding_model'].str.contains('_only'))
        ]
        clf_only = self.results[
            (self.results['task'] == 'high_vol_classification') & 
            (self.results['embedding_model'].str.contains('_only'))
        ]
        
        # Create figure with subplots
        fig = plt.figure(figsize=(20, 12))
        
        # 1. Regression Correlation Comparison
        ax1 = plt.subplot(2, 3, 1)
        models = reg_combined['embedding_model'].values
        x_pos = np.arange(len(models))
        
        # Plot combined and embeddings-only side by side
        width = 0.35
        ax1.bar(x_pos - width/2, reg_combined['test_correlation'].values, 
                width, label='Embeddings + LLM features', alpha=0.8, color='steelblue')
        
        # Get embeddings-only results in same order
        emb_only_corr = []
        for model in models:
            only_result = reg_only[reg_only['embedding_model'] == f"{model}_only"]
            if not only_result.empty:
                emb_only_corr.append(only_result['test_correlation'].values[0])
            else:
                emb_only_corr.append(0)
        
        ax1.bar(x_pos + width/2, emb_only_corr, 
                width, label='Embeddings Only', alpha=0.8, color='coral')
        
        ax1.set_xlabel('Embedding Model')
        ax1.set_ylabel('Pearson Correlation')
        ax1.set_title('Volatility Prediction: Correlation Comparison')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(models, rotation=45, ha='right')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0.3, color='green', linestyle='--', alpha=0.5, label='Good (>0.3)')
        ax1.axhline(y=0.2, color='orange', linestyle='--', alpha=0.5, label='Moderate (>0.2)')
        
        # 2. Classification F1 Score Comparison
        ax2 = plt.subplot(2, 3, 2)
        
        ax2.bar(x_pos - width/2, clf_combined['test_f1'].values, 
                width, label='Embeddings + LLM features', alpha=0.8, color='steelblue')
        
        # Get embeddings-only F1 scores
        emb_only_f1 = []
        for model in models:
            only_result = clf_only[clf_only['embedding_model'] == f"{model}_only"]
            if not only_result.empty:
                emb_only_f1.append(only_result['test_f1'].values[0])
            else:
                emb_only_f1.append(0)
        
        ax2.bar(x_pos + width/2, emb_only_f1, 
                width, label='Embeddings Only', alpha=0.8, color='coral')
        
        ax2.set_xlabel('Embedding Model')
        ax2.set_ylabel('F1 Score')
        ax2.set_title('High Volatility Classification: F1 Score Comparison')
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(models, rotation=45, ha='right')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. MAE Comparison for Regression
        ax3 = plt.subplot(2, 3, 3)
        ax3.bar(x_pos, reg_combined['test_mae'].values, alpha=0.8, color='seagreen')
        ax3.set_xlabel('Embedding Model')
        ax3.set_ylabel('MAE (%)')
        ax3.set_title('Volatility Prediction: Mean Absolute Error')
        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(models, rotation=45, ha='right')
        ax3.grid(True, alpha=0.3)
        
        # 4. AUC Comparison for Classification
        ax4 = plt.subplot(2, 3, 4)
        ax4.bar(x_pos, clf_combined['test_auc'].values, alpha=0.8, color='purple')
        ax4.set_xlabel('Embedding Model')
        ax4.set_ylabel('AUC Score')
        ax4.set_title('High Volatility Classification: AUC Score')
        ax4.set_xticks(x_pos)
        ax4.set_xticklabels(models, rotation=45, ha='right')
        ax4.grid(True, alpha=0.3)
        ax4.axhline(y=0.7, color='green', linestyle='--', alpha=0.5)
        ax4.axhline(y=0.6, color='orange', linestyle='--', alpha=0.5)
        
        # 5. Training Time Comparison
        ax5 = plt.subplot(2, 3, 5)
        
        # Combine regression and classification training times
        total_train_time = []
        for model in models:
            reg_time = reg_combined[reg_combined['embedding_model'] == model]['train_time'].values[0]
            clf_time = clf_combined[clf_combined['embedding_model'] == model]['train_time'].values[0]
            total_train_time.append(reg_time + clf_time)
        
        ax5.bar(x_pos, total_train_time, alpha=0.8, color='orange')
        ax5.set_xlabel('Embedding Model')
        ax5.set_ylabel('Total Training Time (s)')
        ax5.set_title('Training Time: Regression + Classification')
        ax5.set_xticks(x_pos)
        ax5.set_xticklabels(models, rotation=45, ha='right')
        ax5.grid(True, alpha=0.3)
        
        # 6. Overfitting Analysis
        ax6 = plt.subplot(2, 3, 6)
        
        overfitting = reg_combined['overfit_gap'].values
        colors = ['red' if x > 0.1 else 'yellow' if x > 0.05 else 'green' for x in overfitting]
        ax6.bar(x_pos, overfitting, alpha=0.8, color=colors)
        ax6.set_xlabel('Embedding Model')
        ax6.set_ylabel('Train-Test Correlation Gap')
        ax6.set_title('Overfitting Analysis (Lower is Better)')
        ax6.set_xticks(x_pos)
        ax6.set_xticklabels(models, rotation=45, ha='right')
        ax6.grid(True, alpha=0.3)
        ax6.axhline(y=0.1, color='red', linestyle='--', alpha=0.5, label='High overfit')
        ax6.axhline(y=0.05, color='orange', linestyle='--', alpha=0.5, label='Moderate')
        
        plt.suptitle('Embedding Model Comparison for Volatility Prediction', fontsize=16, y=1.02)
        plt.tight_layout()
        output_path = os.path.join(OUTPUT_DIR, 'embedding_comparison_results.png')
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plots saved to: {output_path}")
        
        # Create a second figure for detailed analysis
        fig2, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # 1. Correlation heatmap
        ax = axes[0, 0]
        
        # Create correlation matrix between different metrics
        metrics_df = reg_combined[['embedding_model', 'test_correlation', 'test_r2', 'test_mae']]
        metrics_df = metrics_df.set_index('embedding_model')
        
        # Normalize metrics for comparison
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler()
        metrics_normalized = pd.DataFrame(
            scaler.fit_transform(metrics_df),
            columns=metrics_df.columns,
            index=metrics_df.index
        )
        
        sns.heatmap(metrics_normalized.T, annot=True, fmt='.2f', cmap='YlOrRd', ax=ax)
        ax.set_title('Normalized Regression Metrics Heatmap')
        ax.set_xlabel('Embedding Model')
        
        # 2. Precision-Recall for Classification
        ax = axes[0, 1]
        
        for idx, model in enumerate(models):
            clf_result = clf_combined[clf_combined['embedding_model'] == model].iloc[0]
            ax.scatter(clf_result['test_recall'], clf_result['test_precision'], 
                      s=100, label=model, alpha=0.7)
        
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title('Precision-Recall: High Volatility Detection')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        
        # 3. Feature contribution analysis
        ax = axes[1, 0]
        
        # Calculate improvement from adding LLM features
        improvements = []
        for model in models:
            combined_corr = reg_combined[reg_combined['embedding_model'] == model]['test_correlation'].values[0]
            only_corr = reg_only[reg_only['embedding_model'] == f"{model}_only"]['test_correlation'].values[0] if not reg_only[reg_only['embedding_model'] == f"{model}_only"].empty else 0
            improvement = ((combined_corr - only_corr) / only_corr * 100) if only_corr > 0 else 0
            improvements.append(improvement)
        
        ax.bar(x_pos, improvements, alpha=0.8, color='teal')
        ax.set_xlabel('Embedding Model')
        ax.set_ylabel('Improvement (%)')
        ax.set_title('Performance Gain from Adding LLM Features')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.grid(True, alpha=0.3)
        ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)
        
        # 4. Embedding dimension vs performance
        ax = axes[1, 1]
        
        # Estimate embedding dimensions (you may need to adjust these)
        embedding_dims = {
            'MiniLM-L6': 384,
            'MPNet': 768,
            'Paraphrase-MiniLM': 384,
            'FinBERT': 768,
            'BGE-base': 768,
            'RoBERTa': 768,
        }
        
        dims = [embedding_dims.get(m, 768) for m in models]
        correlations = reg_combined['test_correlation'].values
        
        ax.scatter(dims, correlations, s=100, alpha=0.7)
        for i, model in enumerate(models):
            ax.annotate(model, (dims[i], correlations[i]), fontsize=8, alpha=0.7)
        
        ax.set_xlabel('Embedding Dimension')
        ax.set_ylabel('Test Correlation')
        ax.set_title('Embedding Size vs Performance')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Detailed Embedding Analysis', fontsize=14, y=1.02)
        plt.tight_layout()
        output_path2 = os.path.join(OUTPUT_DIR, 'embedding_detailed_analysis.png')
        plt.savefig(output_path2, dpi=150, bbox_inches='tight')
        print(f"Detailed analysis saved to: {output_path2}")
    
    def save_results(self):
        """Save all results to files"""
        
        print("\n" + "="*70)
        print("SAVING RESULTS")
        print("="*70)
        
        # Save detailed results
        csv_path = os.path.join(OUTPUT_DIR, 'embedding_comparison_results.csv')
        self.results.to_csv(csv_path, index=False)
        print(f"Detailed results saved to: {csv_path}")
        
        # Create summary
        summary = {
            'timestamp': datetime.now().isoformat(),
            'gpu_available': self.device == 'cuda',
            'best_regression': {},
            'best_classification': {},
            'feature_contribution': {}
        }
        
        # Find best models
        reg_results = self.results[self.results['task'] == 'volatility_regression']
        clf_results = self.results[self.results['task'] == 'high_vol_classification']
        
        if not reg_results.empty:
            best_reg = reg_results.loc[reg_results['test_correlation'].idxmax()]
            summary['best_regression'] = {
                'model': best_reg['embedding_model'],
                'correlation': float(best_reg['test_correlation']),
                'r2': float(best_reg['test_r2']),
                'mae': float(best_reg['test_mae'])
            }
        
        if not clf_results.empty:
            best_clf = clf_results.loc[clf_results['test_f1'].idxmax()]
            summary['best_classification'] = {
                'model': best_clf['embedding_model'],
                'f1': float(best_clf['test_f1']),
                'auc': float(best_clf['test_auc']),
                'accuracy': float(best_clf['test_accuracy'])
            }
        
        # Calculate average feature contribution
        reg_combined = reg_results[~reg_results['embedding_model'].str.contains('_only')]
        reg_only = reg_results[reg_results['embedding_model'].str.contains('_only')]
        
        if not reg_combined.empty and not reg_only.empty:
            avg_combined = reg_combined['test_correlation'].mean()
            avg_only = reg_only['test_correlation'].mean()
            summary['feature_contribution'] = {
                'avg_correlation_combined': float(avg_combined),
                'avg_correlation_embeddings_only': float(avg_only),
                'improvement_percent': float((avg_combined - avg_only) / avg_only * 100)
            }
        
        json_path = os.path.join(OUTPUT_DIR, 'embedding_comparison_summary.json')
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)
        print(f"Summary saved to: {json_path}")
        
        # Print summary
        print("\n" + "="*70)
        print("FINAL SUMMARY")
        print("="*70)
        
        print(f"\n🏆 BEST MODELS:")
        print(f"\nVolatility Regression (MLP):")
        if summary['best_regression']:
            print(f"  Model: {summary['best_regression']['model']}")
            print(f"  Correlation: {summary['best_regression']['correlation']:.4f}")
            print(f"  R²: {summary['best_regression']['r2']:.4f}")
            print(f"  MAE: {summary['best_regression']['mae']:.3f}%")
        
        print(f"\nHigh Volatility Classification (RF):")
        if summary['best_classification']:
            print(f"  Model: {summary['best_classification']['model']}")
            print(f"  F1 Score: {summary['best_classification']['f1']:.3f}")
            print(f"  AUC: {summary['best_classification']['auc']:.3f}")
            print(f"  Accuracy: {summary['best_classification']['accuracy']:.1%}")
        
        if summary['feature_contribution']:
            print(f"\n📊 FEATURE CONTRIBUTION:")
            print(f"  Embeddings + LLM features avg: {summary['feature_contribution']['avg_correlation_combined']:.4f}")
            print(f"  Embeddings only avg: {summary['feature_contribution']['avg_correlation_embeddings_only']:.4f}")
            print(f"  Improvement: {summary['feature_contribution']['improvement_percent']:.1f}%")
        
        print("\n✅ All results saved successfully!")

def main():
    """Main execution function"""
    
    print("="*70)
    print("EMBEDDING MODEL COMPARISON FOR VOLATILITY PREDICTION")
    print("="*70)
    print("\nComparing various text embeddings combined with LLM features")
    print("Tasks: 1) Volatility regression (MLP)")
    print("       2) High volatility classification (Random Forest)")
    
    # Initialize comparison framework
    comparison = EmbeddingComparison()
    
    # Run all experiments
    results = comparison.run_experiments()
    
    # Create visualizations
    comparison.create_comparison_plots()
    
    # Save results
    comparison.save_results()
    
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print("\nOutputs:")
    print("  - embedding_comparison_results.csv (detailed results)")
    print("  - embedding_comparison_summary.json (summary)")
    print("  - embedding_comparison_results.png (main plots)")
    print("  - embedding_detailed_analysis.png (detailed analysis)")

if __name__ == "__main__":
    main()