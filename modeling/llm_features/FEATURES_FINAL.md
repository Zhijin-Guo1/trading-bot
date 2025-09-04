# LLM Features - Final Documentation

## Dataset Structure
- **Original columns**: 20 (from 8-K filing data)
- **LLM-generated features**: 24 (extracted via GPT-5-mini)
- **Total columns**: 44

## LLM-Generated Features

### 1. Sub-topic Classification (2 features)

| Feature | Type | Values/Range | Description |
|---------|------|--------------|-------------|
| `sub_topic` | Categorical | 22 categories | Primary event type (e.g., "earnings_beat", "guidance_raised", "acquisition") |
| `sub_topic_confidence` | Continuous | 0.0-1.0 | Model confidence in the classification |

**Categories**: earnings_beat, earnings_miss, earnings_inline, guidance_raised, guidance_lowered, guidance_withdrawn, acquisition, divestiture, share_buyback, dividend_change, debt_restructuring, ceo_change, cfo_change, board_changes, compensation_change, restructuring, expansion, contract_win, regulatory_update, clinical_trial, fda_action, store_metrics, production_update, litigation, accounting_change, cybersecurity, routine_filing

### 2. Novelty Assessment (4 features)

| Feature | Type | Values/Range | Description |
|---------|------|--------------|-------------|
| `novelty_score` | Continuous | 0.0-1.0 | How surprising/unexpected the filing is |
| `change_tags` | JSON List | String array | Key developments or changes announced |
| `is_material` | Boolean | True/False | Whether the filing contains material information |
| `surprise_level` | Categorical | "low", "medium", "high" | Categorical surprise assessment |

### 3. Event Salience (5 features)

| Feature | Type | Values/Range | Description |
|---------|------|--------------|-------------|
| `salience_score` | Continuous | 0.0-1.0 | Importance/materiality to the business |
| `impact_magnitude` | Categorical | "minimal", "moderate", "significant", "transformational" | Size of business impact |
| `time_horizon` | Categorical | "immediate", "short_term", "medium_term", "long_term" | When impact will be felt |
| `uncertainty_level` | Categorical | "low", "medium", "high" | Certainty of outcomes |
| `business_impact` | Categorical | "core", "peripheral", "unclear" | Which part of business affected |

### 4. Financial Tone (4 features)

| Feature | Type | Values/Range | Description |
|---------|------|--------------|-------------|
| `tone_score` | Continuous | -1.0 to 1.0 | Sentiment score (negative to positive) |
| `tone_confidence` | Continuous | 0.0-1.0 | Confidence in tone assessment |
| `tone_drivers` | JSON List | String array | Key phrases driving the tone |
| `quantitative_support` | Boolean | True/False | Whether tone is backed by numbers |

### 5. Risk & Opportunity (4 features)

| Feature | Type | Values/Range | Description |
|---------|------|--------------|-------------|
| `risk_factors` | JSON List | String array | Identified risks (e.g., ["liquidity", "competition"]) |
| `positive_catalysts` | JSON List | String array | Opportunities (e.g., ["growth", "efficiency"]) |
| `net_risk_assessment` | Categorical | "high_risk", "balanced", "high_opportunity" | Overall risk vs opportunity |
| `action_orientation` | Categorical | "defensive", "neutral", "offensive" | Strategic stance implied |

### 6. Volatility Signal (5 features)

| Feature | Type | Values/Range | Description |
|---------|------|--------------|-------------|
| `volatility_score` | Continuous | 0.0-1.0 | Expected price movement magnitude |
| `surprise_factor` | Categorical | "low", "medium", "high" | How unexpected for market |
| `outcome_clarity` | Categorical | "clear", "developing", "uncertain" | Clarity of outcomes |
| `event_type` | Categorical | "binary", "continuous", "mixed" | Nature of the event |
| `expected_reaction` | Categorical | "muted", "moderate", "significant" | Anticipated market reaction |

## Feature Types Summary

### Continuous Features (6)
1. `sub_topic_confidence` (0.0-1.0)
2. `novelty_score` (0.0-1.0)
3. `salience_score` (0.0-1.0)
4. `tone_score` (-1.0 to 1.0)
5. `tone_confidence` (0.0-1.0)
6. `volatility_score` (0.0-1.0)

### Boolean Features (2)
1. `is_material`
2. `quantitative_support`

### Categorical Features (11)
1. `sub_topic` (22 categories)
2. `surprise_level` (3 levels)
3. `impact_magnitude` (4 levels)
4. `time_horizon` (4 periods)
5. `uncertainty_level` (3 levels)
6. `business_impact` (3 types)
7. `net_risk_assessment` (3 assessments)
8. `action_orientation` (3 orientations)
9. `surprise_factor` (3 levels)
10. `outcome_clarity` (3 states)
11. `event_type` (3 types)
12. `expected_reaction` (3 levels)

### JSON List Features (5)
1. `change_tags`
2. `tone_drivers`
3. `risk_factors`
4. `positive_catalysts`

## Usage in ML Models

### Direct Use
- All continuous features can be used directly
- Boolean features can be used as 0/1

### Encoding Required
- Categorical features need one-hot or label encoding
- JSON lists need parsing and vectorization

### Example Processing
```python
# Parse JSON lists
df['risk_factors_list'] = df['risk_factors'].apply(json.loads)
df['num_risks'] = df['risk_factors_list'].apply(len)

# Encode categorical
df['tone_direction'] = df['net_risk_assessment'].map({
    'high_risk': -1, 
    'balanced': 0, 
    'high_opportunity': 1
})

# Combine features
df['risk_adjusted_salience'] = df['salience_score'] * (1 - df['uncertainty_level'].map({
    'low': 0.2, 'medium': 0.5, 'high': 0.8
}))
```

## Key Insights

1. **No Previous Filing Dependencies**: All features extracted from current filing only
2. **Balanced Feature Types**: Mix of continuous (for regression) and categorical (for tree models)
3. **Interpretable Features**: Each feature has clear business meaning
4. **Multi-Task Extraction**: 6 parallel tasks capture different aspects
5. **JSON Flexibility**: List features allow variable-length information

## Target Variable
- `adjusted_return_pct`: 5-day return vs S&P 500 (what we're predicting)
- `binary_label`: UP/DOWN classification target