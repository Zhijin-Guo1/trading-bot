#!/usr/bin/env python3
"""
Debug script to see exactly what the model outputs for pair comparisons
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse

class PairDebugger:
    def __init__(self, model_path: str = "./output_contrastive"):
        """Initialize debugger."""
        print("="*60)
        print("PAIR TRADING DEBUG")
        print("="*60)
        
        base_model = "Qwen/Qwen2.5-7B-Instruct"
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        print("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        print("✅ Model loaded!\n")
    
    def debug_pairs(self, num_samples: int = 5):
        """Debug the first N pair comparisons from test set."""
        
        # Load test data
        with open("data/test.json") as f:
            test_data = json.load(f)
        
        with open("data/test_metadata.json") as f:
            metadata = json.load(f)
        
        # Find pair comparisons
        pair_count = 0
        
        for i, item in enumerate(test_data):
            instruction = item['instruction']
            
            # Check if it's a pair comparison
            if "Filing A" in instruction and "Filing B" in instruction:
                pair_count += 1
                
                print(f"\n{'='*60}")
                print(f"PAIR #{pair_count} (Test index {i})")
                print('='*60)
                
                # Show the instruction (truncated for readability)
                print("\n📝 INSTRUCTION (first 500 chars):")
                print("-"*40)
                print(instruction[:500])
                if len(instruction) > 500:
                    print("...")
                
                # Get metadata
                meta = metadata[i] if i < len(metadata) else {}
                print(f"\n📊 GROUND TRUTH:")
                print("-"*40)
                print(f"Correct answer: {meta.get('correct_answer', 'Unknown')}")
                print(f"Filing A return: {meta.get('filing_a_return', 'N/A'):.2f}%")
                print(f"Filing B return: {meta.get('filing_b_return', 'N/A'):.2f}%")
                print(f"Filing A ticker: {meta.get('filing_a_ticker', 'N/A')}")
                print(f"Filing B ticker: {meta.get('filing_b_ticker', 'N/A')}")
                
                # Generate model response
                print(f"\n🤖 GENERATING MODEL RESPONSE...")
                print("-"*40)
                
                inputs = self.tokenizer(
                    instruction, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1500
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                # Generate with different settings to see full response
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=200,  # More tokens to see full reasoning
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                full_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                print("FULL MODEL OUTPUT:")
                print(full_response)
                
                # Parse what the model predicted
                print(f"\n📈 PARSED PREDICTION:")
                print("-"*40)
                
                response_upper = full_response.upper()
                
                # Check different patterns
                if "FILING A" in response_upper and ("BETTER" in response_upper or "OUTPERFORM" in response_upper):
                    predicted = "A"
                    print("Model predicts: Filing A is better")
                elif "FILING B" in response_upper and ("BETTER" in response_upper or "OUTPERFORM" in response_upper):
                    predicted = "B"
                    print("Model predicts: Filing B is better")
                elif "A RESULTED IN BETTER" in response_upper or "A PERFORMED BETTER" in response_upper:
                    predicted = "A"
                    print("Model predicts: Filing A is better")
                elif "B RESULTED IN BETTER" in response_upper or "B PERFORMED BETTER" in response_upper:
                    predicted = "B"
                    print("Model predicts: Filing B is better")
                else:
                    # Try to find any A or B mention
                    if "FILING A" in full_response[:100]:
                        predicted = "A"
                        print("Model seems to predict: Filing A (found in beginning)")
                    elif "FILING B" in full_response[:100]:
                        predicted = "B"
                        print("Model seems to predict: Filing B (found in beginning)")
                    else:
                        predicted = "UNCLEAR"
                        print("Model prediction is UNCLEAR")
                
                # Check if correct
                correct_answer = meta.get('correct_answer', 'Unknown')
                if correct_answer != 'Unknown':
                    is_correct = predicted == correct_answer
                    print(f"\n✅ Result: {'CORRECT' if is_correct else 'WRONG'}")
                    print(f"   Predicted: {predicted}, Actual: {correct_answer}")
                
                if pair_count >= num_samples:
                    break
        
        print(f"\n{'='*60}")
        print(f"Debugged {pair_count} pair comparisons")
        print("="*60)
    
    def test_custom_pair(self):
        """Test a custom pair comparison."""
        print("\n" + "="*60)
        print("CUSTOM PAIR TEST")
        print("="*60)
        
        # Create a clear test case
        test_instruction = """Compare these two filings and determine which one led to better 5-day stock performance.

Filing A:
Company: TEST_A
The company missed earnings expectations by 20% and lowered guidance for the full year.

Filing B:
Company: TEST_B  
The company beat earnings expectations by 15% and raised guidance for the full year.

Which filing (A or B) resulted in better stock performance? Explain your reasoning."""

        print("Test instruction:")
        print(test_instruction)
        
        print("\n🤖 Model response:")
        inputs = self.tokenizer(test_instruction, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(response)
        
        print("\n📊 Analysis:")
        if "Filing B" in response or "B resulted" in response:
            print("✅ Model correctly identifies Filing B (beat) as better")
        elif "Filing A" in response or "A resulted" in response:
            print("❌ Model incorrectly identifies Filing A (miss) as better")
        else:
            print("⚠️ Model output unclear")

def main():
    parser = argparse.ArgumentParser(description='Debug pair trading predictions')
    parser.add_argument('--model_path', default='./output_contrastive',
                       help='Path to fine-tuned model')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of pairs to debug')
    parser.add_argument('--custom', action='store_true',
                       help='Test a custom pair')
    
    args = parser.parse_args()
    
    debugger = PairDebugger(args.model_path)
    
    if args.custom:
        debugger.test_custom_pair()
    else:
        debugger.debug_pairs(args.num_samples)

if __name__ == "__main__":
    main()