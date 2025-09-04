"""
Test script for GPT-5 three-class experiment
Tests with a small sample to verify everything works
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the experiment module
from modeling.phase2_llm_api.llm_experiment_gpt5 import (
    StockPredictionGPT5,
    load_gpt5_summaries,
    get_balanced_examples
)

def test_data_loading():
    """Test that we can load the GPT-5 summaries"""
    print("\n" + "="*60)
    print("Testing Data Loading")
    print("="*60)
    
    try:
        train_data, test_data = load_gpt5_summaries(year=2018)
        print(f"✓ Successfully loaded data")
        print(f"  Train: {len(train_data)} samples")
        print(f"  Test: {len(test_data)} samples")
        
        # Check data structure
        sample = test_data[0]
        required_fields = ['summary', 'true_label', 'true_return', 'ticker', 'filing_date']
        missing_fields = [f for f in required_fields if f not in sample]
        
        if missing_fields:
            print(f"✗ Missing fields: {missing_fields}")
        else:
            print(f"✓ All required fields present")
            
        # Check label distribution
        labels = [d['true_label'] for d in test_data]
        print(f"\nTest set label distribution:")
        print(f"  UP: {labels.count('UP')} ({labels.count('UP')/len(labels)*100:.1f}%)")
        print(f"  DOWN: {labels.count('DOWN')} ({labels.count('DOWN')/len(labels)*100:.1f}%)")
        print(f"  STAY: {labels.count('STAY')} ({labels.count('STAY')/len(labels)*100:.1f}%)")
        
        return train_data, test_data
        
    except Exception as e:
        print(f"✗ Error loading data: {e}")
        return None, None

def test_prompts():
    """Test prompt generation"""
    print("\n" + "="*60)
    print("Testing Prompt Generation")
    print("="*60)
    
    # Create predictor
    predictor = StockPredictionGPT5()
    
    # Sample summary
    sample_summary = """Advanced Micro Devices reported Q4 revenue of $1.11B, up 15% YoY. 
    The company announced new Ryzen processors and Vega GPU architecture. 
    Management provided Q1 guidance expecting 11% sequential decline."""
    
    # Test zero-shot prompt
    print("\n1. Zero-shot prompt:")
    print("-"*40)
    prompt = predictor.create_zero_shot_prompt(sample_summary)
    print(prompt[:300] + "...")
    
    # Test few-shot prompt with dummy examples
    print("\n2. Few-shot prompt:")
    print("-"*40)
    examples = [
        {"summary": "Strong earnings beat...", "true_return": 5.2, "true_label": "UP"},
        {"summary": "CEO resignation...", "true_return": -3.1, "true_label": "DOWN"},
        {"summary": "Minor guidance update...", "true_return": 0.5, "true_label": "STAY"}
    ]
    prompt = predictor.create_few_shot_prompt(sample_summary, examples)
    print(prompt[:500] + "...")
    
    # Test chain-of-thought prompt
    print("\n3. Chain-of-thought prompt:")
    print("-"*40)
    prompt = predictor.create_cot_prompt(sample_summary)
    print(prompt[:400] + "...")
    
    print("\n✓ All prompts generated successfully")

def test_single_prediction():
    """Test a single prediction with minimal effort"""
    print("\n" + "="*60)
    print("Testing Single Prediction")
    print("="*60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("✗ OPENAI_API_KEY not set, skipping API test")
        return
    
    try:
        # Load a sample
        _, test_data = load_gpt5_summaries(year=2018)
        if not test_data:
            print("✗ No test data available")
            return
            
        sample = test_data[0]
        
        # Create predictor
        predictor = StockPredictionGPT5()
        
        # Test with zero-shot
        prompt = predictor.create_zero_shot_prompt(sample['summary'][:500])
        
        print(f"\nTesting on sample:")
        print(f"  Ticker: {sample['ticker']}")
        print(f"  Date: {sample['filing_date']}")
        print(f"  True label: {sample['true_label']} ({sample['true_return']:.2f}%)")
        print(f"\nMaking prediction...")
        
        prediction, confidence = predictor.get_prediction(prompt, effort="minimal")
        
        print(f"\n✓ Prediction successful")
        print(f"  Predicted: {prediction}")
        print(f"  Confidence: {confidence:.2f}")
        print(f"  Match: {'✓' if prediction == sample['true_label'] else '✗'}")
        
    except Exception as e:
        print(f"✗ Error during prediction: {e}")

def main():
    """Run all tests"""
    print("\n" + "="*60)
    print("GPT-5 EXPERIMENT TEST SUITE")
    print("="*60)
    
    # Test data loading
    train_data, test_data = test_data_loading()
    
    if train_data and test_data:
        # Test prompt generation
        test_prompts()
        
        # Test balanced example selection
        print("\n" + "="*60)
        print("Testing Balanced Example Selection")
        print("="*60)
        examples = get_balanced_examples(train_data, n_per_class=2)
        print(f"✓ Selected {len(examples)} examples")
        
        example_labels = [ex['true_label'] for ex in examples]
        print(f"  Distribution: UP={example_labels.count('UP')}, "
              f"DOWN={example_labels.count('DOWN')}, "
              f"STAY={example_labels.count('STAY')}")
    
    # Test single prediction (optional - requires API key)
    print("\n" + "="*60)
    print("Optional: Test API Call")
    print("="*60)
    
    response = input("Test actual GPT-5 API call? (y/n): ")
    if response.lower() == 'y':
        test_single_prediction()
    else:
        print("Skipping API test")
    
    print("\n" + "="*60)
    print("TEST SUITE COMPLETE")
    print("="*60)

if __name__ == "__main__":
    main()