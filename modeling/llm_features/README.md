# LLM Feature Extraction - Instructions

## Setup
Ensure your OpenAI API key is in the `.env` file:
```
OPENAI_API_KEY=sk-proj-ZosL8HM...
```

## Files in this Directory

| File | Purpose |
|------|---------|
| `extract_llm_features_final.py` | Main extraction engine |
| `prompts.py` | All 6 task prompts |
| `test_20_with_api.py` | Test with 20 samples |
| `run_full_extraction.py` | Full-scale extraction |
| `FEATURES_FINAL.md` | Feature documentation |

## How to Run

### 0. Quick Test (No API Cost)
```bash
cd modeling/llm_features
python test_mock.py
```
- **Cost**: $0 (uses mock responses)
- **Time**: 2 seconds
- **Purpose**: Verify everything works

### 1. Test with 20 Samples (Recommended Next)
```bash
cd modeling/llm_features
python test_20_with_api.py
```
- **Cost**: ~$0.03
- **Time**: ~2 minutes
- **Output**: `test_20_api_results.csv`

### 2. Full-Scale Extraction (All Datasets)
```bash
cd modeling/llm_features
python run_full_extraction.py
```
- **Processes**: train (14,864), val (3,185), test (3,186)
- **Estimated cost**: ~$30
- **Time**: ~2-3 hours
- **Outputs**: 
  - `enhanced_train.csv`
  - `enhanced_val.csv`
  - `enhanced_test.csv`

## Parallel Processing

The pipeline automatically handles parallelization:
- **10 concurrent filings** processed at once
- **6 tasks per filing** run in parallel
- **50 filings per batch**
- **Checkpoints** saved every 500 rows

## Features Generated

24 features across 6 categories:
1. **Sub-topic Classification** (2 features)
2. **Novelty Assessment** (4 features)
3. **Event Salience** (5 features)
4. **Financial Tone** (4 features)
5. **Risk & Opportunity** (4 features)
6. **Volatility Signal** (5 features)

See `FEATURES_FINAL.md` for detailed documentation.

## Cost Estimates

| Dataset | Rows | Estimated Cost |
|---------|------|----------------|
| Test (20 samples) | 20 | $0.03 |
| Train | 14,864 | $21 |
| Validation | 3,185 | $4.50 |
| Test | 3,186 | $4.50 |
| **Total** | **21,235** | **~$30** |

## Monitoring Progress

Watch the console output for:
```
Processing batch 15/289 (train)
Cost so far: $1.05
Checkpoint saved: 750 rows
```

Check `full_extraction.log` for detailed logs.

## Error Handling

- **Automatic retries**: 3 attempts with exponential backoff
- **Checkpoints**: Resume from last checkpoint if interrupted
- **Default values**: Failed tasks get sensible defaults

## Output Format

Enhanced CSV files contain:
- All 20 original columns
- 24 new LLM-generated features
- Total: 44 columns

## Troubleshooting

1. **API Key Error**: Check `.env` file exists and contains valid key
2. **Rate Limits**: Pipeline automatically handles with retries
3. **Out of Memory**: Reduce batch_size in code (default: 50)
4. **Network Issues**: Checkpoints allow resuming

## Next Steps

After extraction completes:
1. Load enhanced CSVs for ML modeling
2. Parse JSON columns before use
3. Encode categorical features as needed