#!/usr/bin/env python3
"""
Simple Test Script - Quick verification that model works
This is the FIRST script you should run after training
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
    """Simple test to verify model works."""
    
    print("="*60)
    print("SIMPLE MODEL TEST")
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
    
    # Test 1: Earnings comparison
    print("\n3. Testing earnings comparison...")
    test1 = """Compare these two filings and determine which one led to better 5-day stock performance.

Filing A:
Company: AAPL
The company reported quarterly earnings of $2.50 per share, significantly beating analyst estimates of $2.00. Revenue grew 15% year-over-year.

Filing B:  
Company: MSFT
The company reported quarterly earnings of $1.50 per share, missing analyst estimates of $2.00. Revenue declined 5% year-over-year.

Which filing resulted in better stock performance?"""

    inputs = tokenizer(test1, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nModel Response:\n{response[:300]}")
    
    # Check if response makes sense
    test1_pass = False
    if "Filing A" in response and ("better" in response.lower() or "outperform" in response.lower()):
        print("\n✅ Test 1 PASSED: Model correctly identified Filing A (earnings beat) as better")
        test1_pass = True
    else:
        print("\n❌ Test 1 FAILED: Model did not identify Filing A as better")
    
    # Test 2: Single filing prediction
    print("\n4. Testing single filing prediction...")
    test2 = """Analyze this Regulation FD Disclosure filing and predict the 5-day market impact.

Company: NVDA
Filing Date: 2024-01-15
Event Type: 7.01 - Regulation FD Disclosure

NVIDIA announced record quarterly revenue of $18 billion, up 265% year-over-year, driven by exceptional demand for AI computing. The company also raised full-year guidance by 20%.

Predict the market impact (STRONG_POSITIVE, POSITIVE, NEGATIVE, or STRONG_NEGATIVE) and explain your reasoning."""

    inputs = tokenizer(test2, return_tensors="pt", truncation=True, max_length=1024)
    
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids.to(model.device),
            max_new_tokens=150,
            temperature=0.1,
            do_sample=True,
            pad_token_id=tokenizer.pad_token_id
        )
    
    response = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    print(f"\nModel Response:\n{response[:300]}")
    
    # Check if response makes sense
    test2_pass = False
    response_upper = response.upper()
    if "POSITIVE" in response_upper or "UP" in response_upper:
        print("\n✅ Test 2 PASSED: Model correctly predicted positive impact for strong earnings")
        test2_pass = True
    else:
        print("\n❌ Test 2 FAILED: Model did not predict positive impact")
    
    # Overall result
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    if test1_pass and test2_pass:
        print("✅ ALL TESTS PASSED! Model is working correctly.")
        print("\nNext steps:")
        print("1. Run full evaluation: python evaluate_full.py")
        print("2. Test on your own examples: python test_custom.py")
        return True
    elif test1_pass or test2_pass:
        print("⚠️ PARTIAL SUCCESS: Some tests passed.")
        print("Model may need more training or different parameters.")
        return True
    else:
        print("❌ TESTS FAILED: Model is not working as expected.")
        print("\nPossible issues:")
        print("1. Training didn't converge (check trainer_log.jsonl)")
        print("2. Wrong model loaded")
        print("3. LoRA weights not properly saved")
        return False

if __name__ == "__main__":
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./output_contrastive"
    test_model(model_path)