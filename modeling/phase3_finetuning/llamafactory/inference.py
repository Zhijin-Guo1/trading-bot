#!/usr/bin/env python3
"""
Inference script for fine-tuned Qwen model
Run predictions on new 8-K filings
"""

import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from typing import Dict, Optional
import argparse

class QwenPredictor:
    def __init__(self, model_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
        """Initialize predictor with fine-tuned model"""
        print(f"Loading model from {model_path}...")
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Load LoRA weights
        self.model = PeftModel.from_pretrained(self.model, model_path)
        self.model.eval()
        
        print("Model loaded successfully!")
        
        # Event type descriptions
        self.event_types = {
            "1.01": "Entry into a Material Definitive Agreement",
            "2.02": "Results of Operations and Financial Condition",
            "5.02": "Departure/Election of Directors or Principal Officers",
            "5.07": "Submission of Matters to a Vote of Security Holders",
            "7.01": "Regulation FD Disclosure",
            "8.01": "Other Events",
            "9.01": "Financial Statements and Exhibits"
        }
    
    def predict(self, 
                text: str, 
                event_type: str = "8.01",
                return_confidence: bool = False) -> Dict:
        """
        Predict stock movement for 8-K filing text
        
        Args:
            text: The 8-K filing text
            event_type: The 8-K item number (e.g., "7.01")
            return_confidence: Whether to return confidence scores
        
        Returns:
            Dictionary with prediction and optional confidence
        """
        # Create instruction
        event_name = self.event_types.get(event_type, "Other Event")
        instruction = (
            f"Analyze this SEC 8-K filing excerpt (Item {event_type}: {event_name}) "
            f"and predict the stock price movement over the next 5 trading days."
        )
        
        # Create system prompt
        system = (
            "You are a financial analyst specializing in SEC filings analysis. "
            "Analyze 8-K filings and predict stock price movements based on the disclosed information. "
            "Consider the materiality, sentiment, and market impact of the disclosed events."
        )
        
        # Format messages
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": f"{instruction}\n\n{text[:2000]}"}  # Truncate if too long
        ]
        
        # Tokenize
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.1,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                return_dict_in_generate=True,
                output_scores=return_confidence
            )
        
        # Decode prediction
        prediction = self.tokenizer.decode(
            outputs.sequences[0][inputs.input_ids.shape[1]:], 
            skip_special_tokens=True
        ).strip()
        
        # Parse prediction
        result = self._parse_prediction(prediction)
        
        # Add confidence if requested
        if return_confidence and outputs.scores:
            # Simple confidence based on generation probability
            confidence = self._calculate_confidence(outputs.scores)
            result["confidence"] = confidence
        
        result["raw_prediction"] = prediction
        result["event_type"] = f"{event_type}: {event_name}"
        
        return result
    
    def _parse_prediction(self, prediction: str) -> Dict:
        """Parse the model's prediction into structured output"""
        pred_lower = prediction.lower()
        
        if "strong" in pred_lower and "negative" in pred_lower:
            return {
                "direction": "DOWN",
                "magnitude": "STRONG",
                "expected_move": ">3% decrease",
                "signal": "SELL"
            }
        elif "negative" in pred_lower and "strong" not in pred_lower:
            return {
                "direction": "DOWN",
                "magnitude": "MODERATE",
                "expected_move": "0-3% decrease",
                "signal": "WEAK_SELL"
            }
        elif "positive" in pred_lower and "strong" not in pred_lower:
            return {
                "direction": "UP",
                "magnitude": "MODERATE", 
                "expected_move": "0-3% increase",
                "signal": "WEAK_BUY"
            }
        elif "strong" in pred_lower and "positive" in pred_lower:
            return {
                "direction": "UP",
                "magnitude": "STRONG",
                "expected_move": ">3% increase",
                "signal": "BUY"
            }
        else:
            return {
                "direction": "NEUTRAL",
                "magnitude": "NONE",
                "expected_move": "No significant movement",
                "signal": "HOLD"
            }
    
    def _calculate_confidence(self, scores) -> float:
        """Calculate confidence score from generation scores"""
        # Simple approach: use mean of top token probabilities
        confidences = []
        for score in scores[:5]:  # Use first 5 tokens
            probs = torch.softmax(score[0], dim=-1)
            top_prob = torch.max(probs).item()
            confidences.append(top_prob)
        return sum(confidences) / len(confidences) if confidences else 0.0
    
    def predict_batch(self, filings: list) -> list:
        """Predict for multiple filings"""
        results = []
        for i, filing in enumerate(filings):
            print(f"Processing {i+1}/{len(filings)}...")
            result = self.predict(
                text=filing.get("text", ""),
                event_type=filing.get("event_type", "8.01"),
                return_confidence=True
            )
            result["ticker"] = filing.get("ticker", "")
            result["filing_date"] = filing.get("filing_date", "")
            results.append(result)
        return results

def interactive_mode(predictor: QwenPredictor):
    """Interactive prediction mode"""
    print("\n" + "="*50)
    print("INTERACTIVE PREDICTION MODE")
    print("="*50)
    print("Enter 'quit' to exit\n")
    
    while True:
        # Get event type
        print("\nAvailable event types:")
        for code, name in predictor.event_types.items():
            print(f"  {code}: {name}")
        
        event_type = input("\nEnter event type (default: 8.01): ").strip()
        if event_type.lower() == 'quit':
            break
        if not event_type:
            event_type = "8.01"
        
        # Get filing text
        print("\nEnter 8-K filing text (press Enter twice to finish):")
        lines = []
        while True:
            line = input()
            if line:
                lines.append(line)
            elif lines:
                break
        
        text = "\n".join(lines)
        if text.lower() == 'quit':
            break
        
        # Make prediction
        print("\nAnalyzing...")
        result = predictor.predict(text, event_type, return_confidence=True)
        
        # Display results
        print("\n" + "-"*40)
        print("PREDICTION RESULTS")
        print("-"*40)
        print(f"Event Type: {result['event_type']}")
        print(f"Direction: {result['direction']}")
        print(f"Magnitude: {result['magnitude']}")
        print(f"Expected Move: {result['expected_move']}")
        print(f"Trading Signal: {result['signal']}")
        if "confidence" in result:
            print(f"Confidence: {result['confidence']:.2%}")
        print(f"\nRaw Output: {result['raw_prediction']}")
        print("-"*40)

def main():
    """Main inference pipeline"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="./outputs/qwen2.5-7b-lora")
    parser.add_argument("--input_file", type=str, help="JSON file with filings to predict")
    parser.add_argument("--output_file", type=str, default="predictions.json")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = QwenPredictor(args.model_path)
    
    if args.interactive:
        # Interactive mode
        interactive_mode(predictor)
    elif args.input_file:
        # Batch prediction
        with open(args.input_file, 'r') as f:
            filings = json.load(f)
        
        print(f"Predicting for {len(filings)} filings...")
        results = predictor.predict_batch(filings)
        
        # Save results
        with open(args.output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\nPredictions saved to {args.output_file}")
        
        # Print summary
        buy_signals = sum(1 for r in results if r['signal'] in ['BUY', 'WEAK_BUY'])
        sell_signals = sum(1 for r in results if r['signal'] in ['SELL', 'WEAK_SELL'])
        hold_signals = sum(1 for r in results if r['signal'] == 'HOLD')
        
        print(f"\nSummary:")
        print(f"  BUY signals: {buy_signals}")
        print(f"  SELL signals: {sell_signals}")
        print(f"  HOLD signals: {hold_signals}")
    else:
        # Demo prediction
        demo_text = """
        The Company announced record quarterly earnings with revenue increasing 35% year-over-year 
        to $2.1 billion. Net income rose 42% to $450 million. The strong performance was driven by 
        robust demand for the company's cloud services and successful launch of new AI products.
        """
        
        print("\nDemo prediction:")
        print(f"Text: {demo_text[:200]}...")
        
        result = predictor.predict(demo_text, "2.02", return_confidence=True)
        
        print(f"\nPrediction: {result['signal']}")
        print(f"Direction: {result['direction']}")
        print(f"Expected Move: {result['expected_move']}")

if __name__ == "__main__":
    main()