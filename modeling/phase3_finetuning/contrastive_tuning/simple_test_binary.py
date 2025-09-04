#!/usr/bin/env python3
"""
Simple Test Script for Binary Classification - POSITIVE/NEGATIVE only
Tests if model correctly identifies outperformance vs underperformance
"""

import os
import sys

# Set HF mirror before importing transformers
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['TRANSFORMERS_OFFLINE'] = '1'  # Use cached model if available

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def test_model(model_path: str = "./output_contrastive"):
    """Simple test to verify model works with binary classification."""
    
    print("="*60)
    print("BINARY CLASSIFICATION TEST (POSITIVE/NEGATIVE)")
    print("="*60)
    
    print(f"Using HF Mirror: {os.environ.get('HF_ENDPOINT')}")
    print(f"Model path: {model_path}")
    
    # Check if model path exists
    if not os.path.exists(model_path):
        print(f"❌ Error: Model path {model_path} does not exist!")
        return False
    
    print("\n1. Loading base model and tokenizer...")
    base_model = "Qwen/Qwen2.5-7B-Instruct"
    
    try:
        # Try loading from cache first
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, 
            trust_remote_code=True,
            local_files_only=True
        )
        print("   ✅ Tokenizer loaded from cache")
    except:
        print("   📥 Downloading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    try:
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            local_files_only=True
        )
        print("   ✅ Base model loaded from cache")
    except:
        print("   📥 Downloading base model (this may take a while)...")
        model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
    
    print("\n2. Loading fine-tuned LoRA weights...")
    model = PeftModel.from_pretrained(model, model_path)
    model.eval()
    print("   ✅ LoRA weights loaded successfully")
    
    # Test 1: Positive case (should outperform)
    print("\n3. Testing POSITIVE case (earnings beat)...")
    test1 = """Analyze this 8-K filing and predict if the stock will outperform or underperform the S&P 500 index over the next 5 days.

Company: AAPL
Event Type: 7.01 - Regulation FD Disclosure
The company reported quarterly earnings of $2.50 per share, significantly beating analyst estimates of $2.00. Revenue grew 15% year-over-year. Management raised full-year guidance.

Will this stock outperform (POSITIVE) or underperform (NEGATIVE) the index? Answer with POSITIVE or NEGATIVE and brief reasoning."""

    inputs = tokenizer(test1, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nModel Response:\n{response}")
    
    # Check if response is correct
    test1_pass = False
    if "POSITIVE" in response.upper():
        print("✅ Test 1 PASSED: Correctly predicted POSITIVE for earnings beat")
        test1_pass = True
    else:
        print("❌ Test 1 FAILED: Should predict POSITIVE for earnings beat")
    
    # Test 2: Negative case (should underperform)
    print("\n4. Testing NEGATIVE case (earnings miss)...")
    test2 = """Analyze this 8-K filing and predict if the stock will outperform or underperform the S&P 500 index over the next 5 days.

Company: MSFT
Event Type: 2.02 - Results of Operations
The company reported quarterly earnings of $1.20 per share, missing analyst estimates of $1.50. Revenue declined 8% year-over-year. Management lowered guidance citing weak demand.

Will this stock outperform (POSITIVE) or underperform (NEGATIVE) the index? Answer with POSITIVE or NEGATIVE and brief reasoning."""

    inputs = tokenizer(test2, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nModel Response:\n{response}")
    
    # Check if response is correct
    test2_pass = False
    if "NEGATIVE" in response.upper():
        print("✅ Test 2 PASSED: Correctly predicted NEGATIVE for earnings miss")
        test2_pass = True
    else:
        print("❌ Test 2 FAILED: Should predict NEGATIVE for earnings miss")
    
    # Test 3: Comparison test (which outperforms)
    print("\n5. Testing comparison (which outperforms)...")
    test3 = """Compare these two filings and determine which stock will outperform the S&P 500 index.

Filing A:
Company: NVDA - Record revenue, raised guidance, AI demand surge

Filing B:  
Company: INTC - Revenue miss, lowered guidance, market share loss

Which filing (A or B) will likely outperform the index? Answer with the filing letter and POSITIVE/NEGATIVE classification."""

    inputs = tokenizer(test3, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
    print(f"\nModel Response:\n{response}")
    
    # Check if response makes sense
    test3_pass = False
    if ("Filing A" in response or "A" in response[:20]) and "POSITIVE" in response.upper():
        print("✅ Test 3 PASSED: Correctly identified Filing A as POSITIVE")
        test3_pass = True
    else:
        print("❌ Test 3 FAILED: Should identify Filing A as POSITIVE")
    
    # Overall result
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum([test1_pass, test2_pass, test3_pass])
    total = 3
    
    print(f"Tests passed: {passed}/{total}")
    
    if passed == total:
        print("✅ ALL TESTS PASSED! Model correctly handles binary classification.")
        print("\nNext steps:")
        print("1. Run quick evaluation: python evaluate_full_binary.py --quick")
        print("2. Run full evaluation: python evaluate_full_binary.py")
        print("3. Test custom examples: python test_custom_binary.py")
        return True
    elif passed >= 2:
        print("⚠️ MOSTLY WORKING: Model understands binary classification but may need refinement.")
        return True
    else:
        print("❌ NEEDS ATTENTION: Model struggles with binary classification.")
        print("\nThe model may need retraining with explicit POSITIVE/NEGATIVE labels.")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./output_contrastive"
    test_model(model_path)