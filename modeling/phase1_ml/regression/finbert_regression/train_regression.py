#!/usr/bin/env python3
"""
Regression-based approach for stock return prediction
Predicts actual return magnitudes instead of binary up/down
More suitable for noisy financial data
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModel, AutoTokenizer
import warnings
warnings.filterwarnings('ignore')

class ReturnRegressionDataset(Dataset):
    """Dataset for return magnitude regression"""
    
    def __init__(self, df, tokenizer, max_length=256):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        # Use actual returns as targets
        self.returns = df['adjusted_return_pct'].values.astype(np.float32)
        
        # Normalize returns for better training
        self.scaler = StandardScaler()
        self.returns_normalized = self.scaler.fit_transform(self.returns.reshape(-1, 1)).flatten()
        
        print(f"Return statistics:")
        print(f"  Mean: {self.returns.mean():.3f}%")
        print(f"  Std: {self.returns.std():.3f}%")
        print(f"  Min: {self.returns.min():.3f}%")
        print(f"  Max: {self.returns.max():.3f}%")
        print(f"  Skew: {pd.Series(self.returns).skew():.3f}")
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        # Create text focusing on magnitude prediction
        text = f"""Predict the 5-day return magnitude for this 8-K filing.

Company: {row.get('ticker', 'N/A')}
Sector: {row.get('sector', 'N/A')}
Filing Date: {row.get('filing_date', 'N/A')}
Items: {row.get('items_present', 'N/A')}
Recent Momentum: 7d={row.get('momentum_7d', 0):.1f}%, 30d={row.get('momentum_30d', 0):.1f}%

Summary: {str(row.get('summary', ''))[:400]}

Historical context: Most 8-K filings result in returns between -3% and +3%.
Predict the exact return percentage:"""
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'return': self.returns[idx],
            'return_normalized': self.returns_normalized[idx],
            'ticker': row.get('ticker', 'N/A')
        }

class FinBERTRegressor(nn.Module):
    """FinBERT-based regression model"""
    
    def __init__(self, model_name='yiyanghkust/finbert-tone', dropout=0.2):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        
        # Regression head with multiple layers for better capacity
        self.regressor = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.last_hidden_state[:, 0]  # [CLS] token
        pooled = self.dropout(pooled)
        return self.regressor(pooled).squeeze()

def analyze_predictions(y_true, y_pred, tickers=None):
    """Comprehensive analysis of regression predictions"""
    
    # Basic metrics
    mse = mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    # Correlation metrics
    pearson_corr, pearson_p = pearsonr(y_true, y_pred)
    spearman_corr, spearman_p = spearmanr(y_true, y_pred)
    
    print("\n" + "="*60)
    print("REGRESSION METRICS")
    print("="*60)
    print(f"MSE: {mse:.4f}")
    print(f"MAE: {mae:.4f}%")
    print(f"R²: {r2:.4f}")
    print(f"Pearson correlation: {pearson_corr:.4f} (p={pearson_p:.4e})")
    print(f"Spearman correlation: {spearman_corr:.4f} (p={spearman_p:.4e})")
    
    # Analyze by magnitude buckets
    print("\n" + "="*60)
    print("PERFORMANCE BY RETURN MAGNITUDE")
    print("="*60)
    
    # Create magnitude buckets
    abs_true = np.abs(y_true)
    buckets = [
        (abs_true < 1, "Small (<1%)"),
        ((abs_true >= 1) & (abs_true < 3), "Medium (1-3%)"),
        ((abs_true >= 3) & (abs_true < 5), "Large (3-5%)"),
        (abs_true >= 5, "Extreme (>5%)")
    ]
    
    for mask, label in buckets:
        if mask.sum() > 0:
            bucket_mae = mean_absolute_error(y_true[mask], y_pred[mask])
            bucket_corr = pearsonr(y_true[mask], y_pred[mask])[0] if mask.sum() > 1 else 0
            print(f"{label:20s}: n={mask.sum():4d}, MAE={bucket_mae:.3f}%, corr={bucket_corr:.3f}")
    
    # Analyze extreme predictions
    print("\n" + "="*60)
    print("EXTREME PREDICTIONS ANALYSIS")
    print("="*60)
    
    # Find top 10% most extreme predictions
    n_extreme = max(10, len(y_pred) // 10)
    extreme_pred_idx = np.argsort(np.abs(y_pred))[-n_extreme:]
    
    extreme_true = y_true[extreme_pred_idx]
    extreme_pred = y_pred[extreme_pred_idx]
    
    # Check if extreme predictions are better
    extreme_corr = pearsonr(extreme_true, extreme_pred)[0] if len(extreme_true) > 1 else 0
    print(f"Top {n_extreme} extreme predictions:")
    print(f"  Correlation: {extreme_corr:.4f}")
    print(f"  MAE: {mean_absolute_error(extreme_true, extreme_pred):.3f}%")
    
    # Direction accuracy for extreme predictions
    direction_correct = np.sign(extreme_true) == np.sign(extreme_pred)
    print(f"  Direction accuracy: {direction_correct.mean():.1%}")
    
    # Volatility prediction
    print("\n" + "="*60)
    print("VOLATILITY PREDICTION")
    print("="*60)
    
    # Can we predict absolute magnitude?
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)
    vol_corr = pearsonr(abs_true, abs_pred)[0]
    print(f"Correlation of absolute returns: {vol_corr:.4f}")
    
    # Rank correlation (can we identify which will move more?)
    rank_corr = spearmanr(abs_true, abs_pred)[0]
    print(f"Rank correlation of magnitudes: {rank_corr:.4f}")
    
    return {
        'mse': mse,
        'mae': mae,
        'r2': r2,
        'pearson': pearson_corr,
        'spearman': spearman_corr,
        'extreme_corr': extreme_corr,
        'vol_corr': vol_corr
    }

def create_diagnostic_plots(y_true, y_pred, save_path="regression_analysis.png"):
    """Create diagnostic plots for regression analysis"""
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # 1. Scatter plot of predictions vs actual
    ax = axes[0, 0]
    ax.scatter(y_true, y_pred, alpha=0.5, s=10)
    ax.plot([-10, 10], [-10, 10], 'r--', lw=2)
    ax.set_xlabel('Actual Return (%)')
    ax.set_ylabel('Predicted Return (%)')
    ax.set_title('Predictions vs Actual')
    ax.grid(True, alpha=0.3)
    
    # Add correlation text
    corr = pearsonr(y_true, y_pred)[0]
    ax.text(0.05, 0.95, f'r={corr:.3f}', transform=ax.transAxes, 
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 2. Residual plot
    ax = axes[0, 1]
    residuals = y_true - y_pred
    ax.scatter(y_pred, residuals, alpha=0.5, s=10)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('Predicted Return (%)')
    ax.set_ylabel('Residual (%)')
    ax.set_title('Residual Plot')
    ax.grid(True, alpha=0.3)
    
    # 3. Distribution comparison
    ax = axes[0, 2]
    ax.hist(y_true, bins=50, alpha=0.5, label='Actual', density=True)
    ax.hist(y_pred, bins=50, alpha=0.5, label='Predicted', density=True)
    ax.set_xlabel('Return (%)')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 4. Q-Q plot
    ax = axes[1, 0]
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot of Residuals')
    ax.grid(True, alpha=0.3)
    
    # 5. Absolute return correlation
    ax = axes[1, 1]
    abs_true = np.abs(y_true)
    abs_pred = np.abs(y_pred)
    ax.scatter(abs_true, abs_pred, alpha=0.5, s=10)
    ax.plot([0, 10], [0, 10], 'r--', lw=2)
    ax.set_xlabel('|Actual Return| (%)')
    ax.set_ylabel('|Predicted Return| (%)')
    ax.set_title('Volatility Prediction')
    ax.grid(True, alpha=0.3)
    
    vol_corr = pearsonr(abs_true, abs_pred)[0]
    ax.text(0.05, 0.95, f'r={vol_corr:.3f}', transform=ax.transAxes,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # 6. Calibration plot (binned)
    ax = axes[1, 2]
    n_bins = 10
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_centers = []
    bin_actuals = []
    
    for i in range(n_bins):
        mask = (y_pred >= bin_edges[i]) & (y_pred < bin_edges[i + 1])
        if mask.sum() > 0:
            bin_centers.append(y_pred[mask].mean())
            bin_actuals.append(y_true[mask].mean())
    
    ax.scatter(bin_centers, bin_actuals, s=100)
    ax.plot([-5, 5], [-5, 5], 'r--', lw=2)
    ax.set_xlabel('Mean Predicted Return (%)')
    ax.set_ylabel('Mean Actual Return (%)')
    ax.set_title('Calibration Plot')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nDiagnostic plots saved to {save_path}")

def train_regression_model():
    """Main training function for regression"""
    
    print("="*60)
    print("RETURN MAGNITUDE REGRESSION")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    train_df = pd.read_csv('../../data/train.csv')
    val_df = pd.read_csv('../../data/val.csv')
    test_df = pd.read_csv('../../data/test.csv')
    
    # Filter for specific items if needed (e.g., earnings)
    item_filter = "2.02"
    if item_filter:
        print(f"Filtering for Item {item_filter}...")
        train_df = train_df[train_df['items_present'].astype(str).str.contains(item_filter, na=False)]
        val_df = val_df[val_df['items_present'].astype(str).str.contains(item_filter, na=False)]
        test_df = test_df[test_df['items_present'].astype(str).str.contains(item_filter, na=False)]
    
    print(f"Dataset sizes - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    # Initialize tokenizer and model
    print("\nInitializing model...")
    tokenizer = AutoTokenizer.from_pretrained('yiyanghkust/finbert-tone')
    model = FinBERTRegressor()
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    
    # Create datasets
    train_dataset = ReturnRegressionDataset(train_df, tokenizer)
    val_dataset = ReturnRegressionDataset(val_df, tokenizer)
    test_dataset = ReturnRegressionDataset(test_df, tokenizer)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.MSELoss()
    
    # Training loop (simplified for demonstration)
    print("\nTraining...")
    model.train()
    for epoch in range(3):
        total_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
            targets = batch['return_normalized'].cuda() if torch.cuda.is_available() else batch['return_normalized']
            
            predictions = model(input_ids, attention_mask)
            loss = criterion(predictions, targets)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
    
    # Evaluation
    print("\nEvaluating on test set...")
    model.eval()
    all_preds = []
    all_true = []
    all_tickers = []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].cuda() if torch.cuda.is_available() else batch['input_ids']
            attention_mask = batch['attention_mask'].cuda() if torch.cuda.is_available() else batch['attention_mask']
            
            predictions = model(input_ids, attention_mask)
            
            # Denormalize predictions
            preds_denorm = train_dataset.scaler.inverse_transform(
                predictions.cpu().numpy().reshape(-1, 1)
            ).flatten()
            
            all_preds.extend(preds_denorm)
            all_true.extend(batch['return'].numpy())
            all_tickers.extend(batch['ticker'])
    
    all_preds = np.array(all_preds)
    all_true = np.array(all_true)
    
    # Analyze results
    metrics = analyze_predictions(all_true, all_preds, all_tickers)
    
    # Create diagnostic plots
    create_diagnostic_plots(all_true, all_preds)
    
    # Save predictions for further analysis
    results_df = pd.DataFrame({
        'ticker': all_tickers,
        'actual_return': all_true,
        'predicted_return': all_preds,
        'error': all_true - all_preds,
        'abs_error': np.abs(all_true - all_preds)
    })
    results_df.to_csv('regression_predictions.csv', index=False)
    print(f"\nPredictions saved to regression_predictions.csv")
    
    return metrics, results_df

if __name__ == "__main__":
    metrics, results = train_regression_model()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS")
    print("="*60)
    
    if metrics['pearson'] > 0.1:
        print("✓ Model shows some predictive power (correlation > 0.1)")
    else:
        print("⚠ Weak correlation - as expected for 5-day returns")
    
    if metrics['vol_corr'] > metrics['pearson']:
        print("✓ Better at predicting volatility than direction")
    
    if metrics['extreme_corr'] > metrics['pearson']:
        print("✓ Model performs better on extreme events")
    
    print("\nConclusion: Even with low R², the model may identify")
    print("which filings will have larger impacts (volatility prediction)")
    print("This is more actionable than binary up/down classification.")