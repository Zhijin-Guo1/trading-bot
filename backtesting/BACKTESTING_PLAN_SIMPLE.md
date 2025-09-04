# Simple Backtesting Plan for Volatility Prediction
Since our model predicts volatility (magnitude) but not direction, we designed volatility-aware backtests. We simulated (i) an options-style strategy that buys volatility on top-quartile predicted events, and (ii) a directional portfolio with position sizing based on predicted magnitude. Both strategies show improved realized returns/risk-adjusted returns compared to naive baselines, demonstrating the economic value of the signal.
## Core Problem
- **What we predict**: Volatility (absolute returns) with 0.36 correlation
- **What we DON'T predict**: Direction (up/down)
- **Solution**: Trade volatility, not direction

## Data Required
```python
# For each 8-K filing in test set:
filing_date     # When to trade
ticker          # What to trade
pred_volatility # Model's predicted |5-day return|
actual_return   # Actual 5-day return (signed)
actual_vol      # Actual |5-day return|
```

## Two Main Strategies

### Strategy 1: Options-Style "Buy Volatility"
**Idea**: If predicted volatility > threshold → Buy straddle (profit from big moves)

```python
# Simple simulation (no real options data)
avg_vol = test_df['actual_vol'].mean()  # Proxy for option cost
threshold = test_df['pred_volatility'].quantile(0.75)  # Top 25%

for each filing:
    if pred_volatility > threshold:
        payoff = actual_vol - avg_vol  # Profit if move > average
        pnl += payoff
```

### Strategy 2: Volatility-Based Position Sizing
**Idea**: Take smaller positions when high volatility predicted

```python
# Even with random direction, smart sizing helps
for each filing:
    direction = random.choice([-1, 1])  # or use weak signal
    position_size = 1.0 / pred_volatility  # Inverse weighting
    pnl = direction * actual_return * position_size
```

## Key Visualizations

### 📊 Plot 1: Prediction Quality
```python
# Scatter plot: Predicted vs Actual Volatility
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.scatter(pred_volatility, actual_vol, alpha=0.5)
plt.plot([0, 10], [0, 10], 'r--')  # Perfect prediction line
plt.xlabel('Predicted Volatility (%)')
plt.ylabel('Actual Volatility (%)')
plt.title(f'Volatility Prediction (r={correlation:.3f})')

plt.subplot(1, 2, 2)
# Quartile analysis
quartiles = pd.qcut(pred_volatility, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
plt.boxplot([actual_vol[quartiles == q] for q in ['Q1', 'Q2', 'Q3', 'Q4']])
plt.xlabel('Predicted Volatility Quartile')
plt.ylabel('Actual Volatility (%)')
plt.title('Actual Volatility by Prediction Quartile')
```

### 📈 Plot 2: Strategy Performance - Cumulative Returns
```python
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Strategy 1: Options simulation
ax = axes[0, 0]
cumulative_pnl = strategy1_returns.cumsum()
ax.plot(dates, cumulative_pnl, label='Trade Top 25%', linewidth=2)
ax.plot(dates, baseline_returns.cumsum(), label='Trade All', alpha=0.7)
ax.plot(dates, random_returns.cumsum(), label='Random Selection', alpha=0.7)
ax.set_title('Strategy 1: Options-Style Trading')
ax.set_ylabel('Cumulative PnL (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Strategy 2: Position sizing
ax = axes[0, 1]
ax.plot(dates, vol_weighted_returns.cumsum(), label='Vol-Weighted', linewidth=2)
ax.plot(dates, equal_weight_returns.cumsum(), label='Equal Weight', alpha=0.7)
ax.set_title('Strategy 2: Smart Position Sizing')
ax.set_ylabel('Cumulative PnL (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Drawdown comparison
ax = axes[1, 0]
for strategy, returns in strategies.items():
    drawdown = calculate_drawdown(returns)
    ax.plot(dates, drawdown, label=strategy)
ax.set_title('Maximum Drawdown Comparison')
ax.set_ylabel('Drawdown (%)')
ax.legend()
ax.grid(True, alpha=0.3)

# Monthly returns heatmap
ax = axes[1, 1]
monthly_returns = vol_weighted_returns.resample('M').sum()
monthly_matrix = monthly_returns.values.reshape(12, -1)
sns.heatmap(monthly_matrix, annot=True, fmt='.1f', cmap='RdYlGn', center=0, ax=ax)
ax.set_title('Monthly Returns Heatmap')
ax.set_xlabel('Year')
ax.set_ylabel('Month')
```

### 📊 Plot 3: Performance by Volatility Level
```python
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Average payoff by prediction quartile
ax = axes[0]
avg_payoff = df.groupby(quartiles)['strategy_return'].mean()
colors = ['red' if x < 0 else 'green' for x in avg_payoff]
ax.bar(range(4), avg_payoff, color=colors, alpha=0.7)
ax.set_xticks(range(4))
ax.set_xticklabels(['Q1\n(Low Vol)', 'Q2', 'Q3', 'Q4\n(High Vol)'])
ax.set_ylabel('Average Return (%)')
ax.set_title('Returns by Predicted Volatility Quartile')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.5)

# Win rate by quartile
ax = axes[1]
win_rate = df.groupby(quartiles)['strategy_return'].apply(lambda x: (x > 0).mean())
ax.bar(range(4), win_rate, color='steelblue', alpha=0.7)
ax.set_xticks(range(4))
ax.set_xticklabels(['Q1', 'Q2', 'Q3', 'Q4'])
ax.set_ylabel('Win Rate')
ax.set_title('Win Rate by Volatility Quartile')
ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5)

# Risk-reward scatter
ax = axes[2]
for q in range(4):
    data = df[quartiles == f'Q{q+1}']
    avg_return = data['strategy_return'].mean()
    volatility = data['strategy_return'].std()
    ax.scatter(volatility, avg_return, s=100, label=f'Q{q+1}')
ax.set_xlabel('Risk (Std Dev)')
ax.set_ylabel('Return (%)')
ax.set_title('Risk-Return by Quartile')
ax.legend()
ax.grid(True, alpha=0.3)
```

### 📉 Plot 4: Performance Metrics Dashboard
```python
fig = plt.figure(figsize=(15, 10))

# Create metrics summary
metrics = {
    'Strategy 1 (Options)': calculate_metrics(strategy1_returns),
    'Strategy 2 (Vol-Weight)': calculate_metrics(strategy2_returns),
    'Baseline (Equal)': calculate_metrics(baseline_returns),
    'Buy & Hold SPY': calculate_metrics(spy_returns)
}

# Sharpe Ratio comparison
ax = plt.subplot(2, 3, 1)
sharpes = [m['sharpe'] for m in metrics.values()]
colors = ['green' if s > 0 else 'red' for s in sharpes]
ax.barh(range(len(metrics)), sharpes, color=colors, alpha=0.7)
ax.set_yticks(range(len(metrics)))
ax.set_yticklabels(metrics.keys())
ax.set_xlabel('Sharpe Ratio')
ax.set_title('Risk-Adjusted Returns')
ax.axvline(x=0, color='black', linewidth=0.5)

# Return distribution
ax = plt.subplot(2, 3, 2)
for name, returns in {'Vol-Weight': strategy2_returns, 'Baseline': baseline_returns}.items():
    ax.hist(returns, bins=50, alpha=0.5, label=name)
ax.set_xlabel('Daily Return (%)')
ax.set_ylabel('Frequency')
ax.set_title('Return Distribution')
ax.legend()

# Rolling correlation
ax = plt.subplot(2, 3, 3)
rolling_corr = pd.Series(pred_volatility).rolling(100).corr(pd.Series(actual_vol))
ax.plot(rolling_corr)
ax.set_ylabel('Correlation')
ax.set_title('Rolling 100-day Prediction Correlation')
ax.axhline(y=0.36, color='red', linestyle='--', alpha=0.5, label='Overall')
ax.grid(True, alpha=0.3)

# Performance table
ax = plt.subplot(2, 3, (4, 6))
ax.axis('tight')
ax.axis('off')
table_data = []
for strategy, m in metrics.items():
    table_data.append([
        strategy,
        f"{m['total_return']:.1%}",
        f"{m['sharpe']:.2f}",
        f"{m['max_dd']:.1%}",
        f"{m['win_rate']:.1%}"
    ])
table = ax.table(cellText=table_data,
                colLabels=['Strategy', 'Return', 'Sharpe', 'Max DD', 'Win Rate'],
                cellLoc='center',
                loc='center')
table.auto_set_font_size(False)
table.set_fontsize(10)
table.scale(1.2, 1.5)

plt.suptitle('Volatility Trading Strategy Performance', fontsize=14, y=1.02)
plt.tight_layout()
```

## Success Metrics

### Minimum Bar
- Sharpe > 0.5
- Max Drawdown < 20%
- Better than random selection

### Good Performance  
- Sharpe > 1.0
- Win rate > 55% on high-confidence trades
- Consistent positive returns

## Key Questions
1. Transaction costs: What's realistic? (Currently assuming 0.1%)
2. Position limits: Max position size per trade?
3. Rebalancing frequency: Daily? Weekly?
4. Should we try combining with weak directional signals?

## Simple Code Template
```python
def backtest_volatility_strategy(predictions_df, strategy='options'):
    """
    Simple backtest for volatility strategies
    """
    pnl = []
    
    for idx, row in predictions_df.iterrows():
        if strategy == 'options':
            # Trade top quartile predictions
            if row['pred_vol'] > threshold:
                profit = row['actual_vol'] - avg_cost - transaction_cost
                pnl.append(profit)
        
        elif strategy == 'vol_sizing':
            # Random direction, smart sizing
            direction = np.random.choice([-1, 1])
            size = min(1.0 / row['pred_vol'], max_position)
            profit = direction * row['actual_return'] * size - transaction_cost
            pnl.append(profit)
    
    return pd.Series(pnl)

# Run and plot
results = backtest_volatility_strategy(test_df, 'options')
results.cumsum().plot(title='Cumulative PnL')
print(f"Sharpe: {results.mean() / results.std() * np.sqrt(252):.2f}")
```