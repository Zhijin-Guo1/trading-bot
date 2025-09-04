#!/usr/bin/env python3
"""
Debug script with FIXED parsing logic for pair comparisons
"""

import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import argparse
import re

class PairDebugger:
    def __init__(self, model_path: str = "./output_contrastive"):
        """Initialize debugger."""
        print("="*60)
        print("PAIR TRADING DEBUG - FIXED PARSING")
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
    
    def parse_pair_prediction(self, response: str) -> str:
        """Better parsing logic for pair predictions."""
        
        # Take only the first sentence/paragraph to avoid repetition issues
        first_part = response.split('\n')[0] if '\n' in response else response
        first_100_chars = response[:100]
        
        # Method 1: Look for explicit statements in first part
        patterns = [
            (r'Filing ([AB]) resulted in better', 1),
            (r'Filing ([AB]) performed better', 1),
            (r'Filing ([AB]) outperformed', 1),
            (r'([AB]) resulted in better', 1),
            (r'([AB]) performed better', 1),
            (r'Filing ([AB]) .*better', 1),
            (r'Answer:\s*Filing ([AB])', 1),
            (r'Answer:\s*([AB])', 1),
        ]
        
        for pattern, group in patterns:
            match = re.search(pattern, first_part, re.IGNORECASE)
            if match:
                return match.group(group).upper()
        
        # Method 2: Count occurrences in full response (but weight early occurrences more)
        lines = response.split('\n')[:3]  # Only first 3 lines to avoid repetition
        text_to_check = ' '.join(lines)
        
        a_count = len(re.findall(r'Filing A', text_to_check, re.IGNORECASE))
        b_count = len(re.findall(r'Filing B', text_to_check, re.IGNORECASE))
        
        # Also check for just "A" or "B" at start
        if re.match(r'^[^a-zA-Z]*A[^a-zA-Z]', first_100_chars):
            a_count += 2  # Weight this heavily
        if re.match(r'^[^a-zA-Z]*B[^a-zA-Z]', first_100_chars):
            b_count += 2  # Weight this heavily
        
        if a_count > b_count:
            return "A"
        elif b_count > a_count:
            return "B"
        
        # Method 3: Default to what appears first
        a_pos = response.lower().find('filing a')
        b_pos = response.lower().find('filing b')
        
        if a_pos != -1 and b_pos != -1:
            return "A" if a_pos < b_pos else "B"
        elif a_pos != -1:
            return "A"
        elif b_pos != -1:
            return "B"
        
        return "UNCLEAR"
    
    def debug_pairs(self, num_samples: int = 5):
        """Debug the first N pair comparisons from test set."""
        
        # Load test data
        with open("data/test.json") as f:
            test_data = json.load(f)
        
        with open("data/test_metadata.json") as f:
            metadata = json.load(f)
        
        # Find pair comparisons
        pair_count = 0
        correct_count = 0
        
        for i, item in enumerate(test_data):
            instruction = item['instruction']
            
            # Check if it's a pair comparison
            if "Filing A" in instruction and "Filing B" in instruction:
                pair_count += 1
                
                print(f"\n{'='*60}")
                print(f"PAIR #{pair_count} (Test index {i})")
                print('='*60)
                
                # Show brief instruction
                print("\n📝 INSTRUCTION (brief):")
                print("-"*40)
                # Extract just the company names
                lines = instruction.split('\n')
                for line in lines:
                    if 'Company:' in line:
                        print(line)
                
                # Get metadata
                meta = metadata[i] if i < len(metadata) else {}
                print(f"\n📊 GROUND TRUTH:")
                print("-"*40)
                correct_answer = meta.get('correct_answer', 'Unknown')
                print(f"Correct answer: {correct_answer}")
                print(f"Filing A ({meta.get('filing_a_ticker', 'N/A')}): {meta.get('filing_a_return', 0):.2f}%")
                print(f"Filing B ({meta.get('filing_b_ticker', 'N/A')}): {meta.get('filing_b_return', 0):.2f}%")
                
                # Generate model response
                print(f"\n🤖 MODEL RESPONSE:")
                print("-"*40)
                
                inputs = self.tokenizer(
                    instruction, 
                    return_tensors="pt", 
                    truncation=True, 
                    max_length=1500
                )
                inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
                
                with torch.no_grad():
                    outputs = self.model.generate(
                        **inputs,
                        max_new_tokens=150,  # Reduced to avoid repetition
                        temperature=0.1,
                        do_sample=True,
                        top_p=0.9,
                        pad_token_id=self.tokenizer.pad_token_id,
                        repetition_penalty=1.2  # Add penalty for repetition
                    )
                
                full_response = self.tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                # Show only first part to avoid repetition spam
                first_part = full_response.split('\n\n')[0] if '\n\n' in full_response else full_response[:200]
                print(f"First part: {first_part}")
                
                # Parse with improved logic
                print(f"\n📈 IMPROVED PARSING:")
                print("-"*40)
                
                predicted = self.parse_pair_prediction(full_response)
                print(f"Parsed prediction: {predicted}")
                
                # Check if correct
                if correct_answer != 'Unknown':
                    is_correct = predicted == correct_answer
                    if is_correct:
                        correct_count += 1
                    print(f"\n{'✅ CORRECT' if is_correct else '❌ WRONG'}")
                    print(f"   Model: {predicted}, Truth: {correct_answer}")
                
                if pair_count >= num_samples:
                    break
        
        print(f"\n{'='*60}")
        print(f"SUMMARY: {correct_count}/{pair_count} correct ({100*correct_count/pair_count:.1f}% accuracy)")
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

Which filing (A or B) resulted in better stock performance?"""

        print("Test instruction:")
        print(test_instruction)
        
        print("\n🤖 Model response:")
        inputs = self.tokenizer(test_instruction, return_tensors="pt", truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=100,
                temperature=0.1,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                repetition_penalty=1.2
            )
        
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        print(response)
        
        predicted = self.parse_pair_prediction(response)
        print(f"\n📊 Parsed: {predicted}")
        
        if predicted == "B":
            print("✅ Model correctly identifies Filing B (beat) as better")
        elif predicted == "A":
            print("❌ Model incorrectly identifies Filing A (miss) as better")
        else:
            print("⚠️ Model output unclear")

def main():
    parser = argparse.ArgumentParser(description='Debug pair trading predictions with fixed parsing')
    parser.add_argument('--model_path', default='./output_contrastive',
                       help='Path to fine-tuned model')
    parser.add_argument('--num_samples', type=int, default=10,
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