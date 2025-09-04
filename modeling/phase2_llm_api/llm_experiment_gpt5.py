"""
GPT-5 Three-Class Stock Prediction Experiment
Uses GPT-5 summarized data from 2018 for training and testing
Predicts: UP (>1%), DOWN (<-1%), or STAY (within 1%) relative to market
"""

import os
import json
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from datetime import datetime
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Load environment variables
from dotenv import load_dotenv
load_dotenv('../../.env')

# OpenAI imports
from openai import OpenAI
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class StockPredictionGPT5:
    """GPT-5 based three-class stock prediction from 8-K summaries"""
    
    def __init__(self, api_key: str = None, model: str = "gpt-5"):
        """Initialize with OpenAI API"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
        self.model = model
        
        # Track metrics
        self.total_tokens = 0
        self.total_cost = 0
        self.response_times = []
        
        # GPT-5 pricing (per 1K tokens) - adjust as needed
        self.input_price = 0.00025  # $0.25 per 1M tokens
        self.output_price = 0.00025
        
    def create_zero_shot_prompt(self, summary: str) -> str:
        """Zero-shot prompt for three-class prediction"""
        return f"""Analyze this 8-K filing summary and predict stock performance relative to the market.

Task: Predict if the stock will:
- UP: Outperform market by >1% over next 5 trading days
- DOWN: Underperform market by >1% over next 5 trading days  
- STAY: Perform within ±1% of market over next 5 trading days

8-K Summary:
{summary}

Answer with only one word: UP, DOWN, or STAY"""

    def create_few_shot_prompt(self, summary: str, examples: List[Dict]) -> str:
        """
        Few-shot prompt with balanced examples (2 UP, 2 DOWN, 2 STAY)
        """
        prompt = """Task: Predict stock performance relative to market based on 8-K summaries.

Categories:
- UP: Outperforms market by >1% over next 5 days
- DOWN: Underperforms market by >1% over next 5 days
- STAY: Within ±1% of market over next 5 days

Here are examples from historical data:
"""
        
        # Use 6 examples: 2 UP, 2 DOWN, 2 STAY for balance
        for i, ex in enumerate(examples, 1):
            summary_preview = ex['summary'][:300] if len(ex['summary']) > 300 else ex['summary']
            prompt += f"""
Example {i}:
8-K Summary: {summary_preview}...
Market-adjusted return: {ex['true_return']:.2f}%
Result: {ex['true_label']}
"""
        
        prompt += f"""
Now analyze this new 8-K summary:
{summary}

Prediction:"""
        return prompt

    def create_cot_prompt(self, summary: str) -> str:
        """Chain-of-thought prompt with reasoning steps"""
        return f"""Analyze this 8-K filing summary step-by-step to predict stock performance relative to the market.

8-K Summary:
{summary}

Analysis steps:
1. Identify the key events or announcements (earnings, M&A, executive changes, etc.)
2. Assess if these events are materially positive, negative, or neutral
3. Consider likely investor reaction and market sentiment
4. Evaluate the magnitude of potential impact (large >1%, small <1%)
5. Consider any forward guidance or future outlook mentioned

Based on your analysis, predict the stock's 5-day performance relative to market:
- UP: Outperform by >1%
- DOWN: Underperform by >1%  
- STAY: Within ±1% of market

Provide brief reasoning (2-3 sentences), then state your final prediction as: UP, DOWN, or STAY"""

    def get_prediction(self, prompt: str, effort: str = "medium") -> Tuple[str, float]:
        """Get prediction from GPT-5 using new API format"""
        try:
            start_time = time.time()
            
            # Use new GPT-5 API format
            response = self.client.responses.create(
                model=self.model,
                input=prompt,
                reasoning={
                    "effort": effort  # Can be "minimal", "medium", or "high"
                }
            )
            
            response_time = time.time() - start_time
            self.response_times.append(response_time)
            
            # Extract response from GPT-5 format
            if hasattr(response, 'output'):
                if isinstance(response.output, list) and len(response.output) > 0:
                    for item in response.output:
                        if hasattr(item, 'content') and item.content:
                            if isinstance(item.content, list) and len(item.content) > 0:
                                full_response = item.content[0].text if hasattr(item.content[0], 'text') else str(item.content[0])
                            else:
                                full_response = str(item.content)
                            break
                    else:
                        full_response = str(response.output)
                else:
                    full_response = str(response.output)
            else:
                full_response = str(response)
            
            # Track tokens and cost
            if hasattr(response, 'usage'):
                tokens = response.usage.total_tokens
                self.total_tokens += tokens
                self.total_cost += tokens * self.input_price / 1000
            
            # Parse prediction for three classes
            response_upper = full_response.upper()
            
            # Look for UP, DOWN, or STAY
            if "UP" in response_upper and "DOWN" not in response_upper[:response_upper.index("UP")]:
                prediction = "UP"
                confidence = 0.9 if response_upper.count("UP") > response_upper.count("DOWN") else 0.7
            elif "DOWN" in response_upper:
                prediction = "DOWN"
                confidence = 0.9 if response_upper.count("DOWN") > response_upper.count("UP") else 0.7
            elif "STAY" in response_upper or "NEUTRAL" in response_upper:
                prediction = "STAY"
                confidence = 0.8
            else:
                # Default to STAY if unclear
                prediction = "STAY"
                confidence = 0.5
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return "STAY", 0.5

    def evaluate_strategy(self, 
                          data: List[Dict], 
                          strategy: str,
                          training_examples: List[Dict] = None,
                          effort: str = "medium") -> Dict:
        """
        Evaluate a prompting strategy on dataset
        
        Args:
            data: Test data (list of dictionaries with summaries)
            strategy: "zero_shot", "few_shot", or "chain_of_thought"
            training_examples: For few-shot only - examples from training set
            effort: GPT-5 reasoning effort level
        """
        
        predictions = []
        true_labels = []
        confidences = []
        
        print(f"\n{'='*60}")
        print(f"Evaluating {strategy} strategy with effort={effort}")
        print(f"Dataset size: {len(data)} samples")
        if strategy == "few_shot" and training_examples:
            print(f"Using {len(training_examples)} training examples")
        print(f"{'='*60}")
        
        for item in tqdm(data, desc=strategy):
            # Create prompt
            if strategy == "zero_shot":
                prompt = self.create_zero_shot_prompt(item['summary'])
            elif strategy == "few_shot":
                if training_examples is None:
                    raise ValueError("Training examples required for few-shot")
                prompt = self.create_few_shot_prompt(item['summary'], training_examples)
            elif strategy == "chain_of_thought":
                prompt = self.create_cot_prompt(item['summary'])
            else:
                raise ValueError(f"Unknown strategy: {strategy}")
            
            # Get prediction
            pred, conf = self.get_prediction(prompt, effort)
            
            # Store results
            predictions.append(pred)
            true_labels.append(item['true_label'])
            confidences.append(conf)
            
            # Rate limiting
            time.sleep(0.5)
        
        # Calculate metrics for 3-class classification
        label_map = {"UP": 0, "DOWN": 1, "STAY": 2}
        pred_numeric = [label_map[p] for p in predictions]
        true_numeric = [label_map[t] for t in true_labels]
        
        accuracy = accuracy_score(true_numeric, pred_numeric)
        
        # Per-class metrics
        precision = precision_score(true_numeric, pred_numeric, average=None, zero_division=0)
        recall = recall_score(true_numeric, pred_numeric, average=None, zero_division=0)
        f1 = f1_score(true_numeric, pred_numeric, average=None, zero_division=0)
        
        # Weighted average metrics
        precision_avg = precision_score(true_numeric, pred_numeric, average='weighted', zero_division=0)
        recall_avg = recall_score(true_numeric, pred_numeric, average='weighted', zero_division=0)
        f1_avg = f1_score(true_numeric, pred_numeric, average='weighted', zero_division=0)
        
        cm = confusion_matrix(true_numeric, pred_numeric)
        
        # Calculate distribution statistics
        pred_dist = {label: predictions.count(label) / len(predictions) for label in ["UP", "DOWN", "STAY"]}
        true_dist = {label: true_labels.count(label) / len(true_labels) for label in ["UP", "DOWN", "STAY"]}
        avg_confidence = np.mean(confidences)
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        results = {
            "strategy": strategy,
            "effort": effort,
            "dataset_size": len(data),
            "accuracy": accuracy,
            "precision_per_class": {"UP": precision[0], "DOWN": precision[1], "STAY": precision[2]},
            "recall_per_class": {"UP": recall[0], "DOWN": recall[1], "STAY": recall[2]},
            "f1_per_class": {"UP": f1[0], "DOWN": f1[1], "STAY": f1[2]},
            "precision_weighted": precision_avg,
            "recall_weighted": recall_avg,
            "f1_weighted": f1_avg,
            "confusion_matrix": cm.tolist(),
            "predicted_distribution": pred_dist,
            "actual_distribution": true_dist,
            "avg_confidence": avg_confidence,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_response_time": avg_response_time
        }
        
        # Print summary
        print(f"\nResults for {strategy}:")
        print(f"Accuracy: {accuracy:.3f}")
        print(f"Weighted Precision: {precision_avg:.3f}")
        print(f"Weighted Recall: {recall_avg:.3f}")
        print(f"Weighted F1 Score: {f1_avg:.3f}")
        print(f"\nPer-class F1 scores:")
        print(f"  UP:   {f1[0]:.3f}")
        print(f"  DOWN: {f1[1]:.3f}")
        print(f"  STAY: {f1[2]:.3f}")
        print(f"\nPredicted distribution: UP={pred_dist['UP']:.2f}, DOWN={pred_dist['DOWN']:.2f}, STAY={pred_dist['STAY']:.2f}")
        print(f"Actual distribution:    UP={true_dist['UP']:.2f}, DOWN={true_dist['DOWN']:.2f}, STAY={true_dist['STAY']:.2f}")
        print(f"\nTotal cost: ${self.total_cost:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"           Predicted:")
        print(f"           UP   DOWN  STAY")
        print(f"Actual UP:   {cm[0][0]:4d}  {cm[0][1]:4d}  {cm[0][2]:4d}")
        print(f"      DOWN:  {cm[1][0]:4d}  {cm[1][1]:4d}  {cm[1][2]:4d}")
        print(f"      STAY:  {cm[2][0]:4d}  {cm[2][1]:4d}  {cm[2][2]:4d}")
        
        return results


def load_gpt5_summaries(year: int = 2018):
    """
    Load GPT-5 summarized data from specified year
    Returns train/test split
    """
    print(f"\nLoading GPT-5 summaries from year {year}...")
    
    # Load from the GPT summarization results
    summary_path = Path(f"../../gpt_summarization/results/{year}/all_summaries.json")
    
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")
    
    with open(summary_path, 'r') as f:
        data = json.load(f)
    
    # Filter successful summaries only
    successful_data = [d for d in data if d.get('status') == 'success']
    
    print(f"Loaded {len(successful_data)} successful summaries from {year}")
    
    # Convert true_label to match our three-class system
    # The data already has UP/DOWN/STAY labels, we just need to ensure they match our thresholds
    # Since the original labels were based on different thresholds, we'll recalculate based on returns
    
    for item in successful_data:
        # Recalculate label based on our thresholds (>1%, <-1%, within ±1%)
        return_val = item.get('true_return', 0)
        if return_val > 1.0:
            item['true_label'] = 'UP'
        elif return_val < -1.0:
            item['true_label'] = 'DOWN'
        else:
            item['true_label'] = 'STAY'
    
    # Print distribution
    labels = [d['true_label'] for d in successful_data]
    print(f"Label distribution: UP={labels.count('UP')}, DOWN={labels.count('DOWN')}, STAY={labels.count('STAY')}")
    
    # Split into train/test (80/20)
    train_data, test_data = train_test_split(successful_data, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    return train_data, test_data


def get_balanced_examples(train_data: List[Dict], n_per_class: int = 2) -> List[Dict]:
    """
    Get balanced training examples for few-shot learning
    Returns n_per_class examples for each of UP, DOWN, STAY
    """
    import random
    random.seed(42)
    
    # Group by label
    up_examples = [d for d in train_data if d['true_label'] == 'UP']
    down_examples = [d for d in train_data if d['true_label'] == 'DOWN']
    stay_examples = [d for d in train_data if d['true_label'] == 'STAY']
    
    # Sample n_per_class from each
    examples = []
    
    if len(up_examples) >= n_per_class:
        examples.extend(random.sample(up_examples, n_per_class))
    else:
        examples.extend(up_examples)
    
    if len(down_examples) >= n_per_class:
        examples.extend(random.sample(down_examples, n_per_class))
    else:
        examples.extend(down_examples)
        
    if len(stay_examples) >= n_per_class:
        examples.extend(random.sample(stay_examples, n_per_class))
    else:
        examples.extend(stay_examples)
    
    print(f"Selected {len(examples)} balanced examples for few-shot learning")
    return examples


def run_gpt5_experiments(effort_level: str = "medium", sample_size: int = None):
    """
    Run GPT-5 experiments with three-class classification
    
    Args:
        effort_level: "minimal", "medium", or "high" for GPT-5 reasoning
        sample_size: If provided, use only this many test samples (for faster testing)
    """
    
    print("\n" + "="*60)
    print(f"Starting GPT-5 Three-Class Experiments")
    print(f"Reasoning effort: {effort_level}")
    print("="*60)
    
    # Load data
    train_data, test_data = load_gpt5_summaries(year=2018)
    
    # Get balanced examples for few-shot
    training_examples = get_balanced_examples(train_data, n_per_class=2)
    
    # Optionally sample test data for faster experiments
    if sample_size and sample_size < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, sample_size)
        print(f"\nUsing sampled test set: {sample_size} samples")
    
    # Create results directory
    results_dir = Path("results_gpt5")
    results_dir.mkdir(exist_ok=True)
    
    # Strategies to test
    strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    
    all_results = {
        "model": "gpt-5",
        "effort": effort_level,
        "data_year": 2018,
        "test_size": len(test_data),
        "train_size": len(train_data),
        "timestamp": datetime.now().isoformat(),
        "strategies": {}
    }
    
    for strategy in strategies:
        # Create new predictor for each strategy to reset costs
        predictor = StockPredictionGPT5(model="gpt-5")
        
        # Run evaluation
        if strategy == "few_shot":
            results = predictor.evaluate_strategy(test_data, strategy, training_examples, effort_level)
        else:
            results = predictor.evaluate_strategy(test_data, strategy, effort=effort_level)
        
        all_results["strategies"][strategy] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"gpt5_results_{effort_level}_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print comparison
    print("\n" + "="*60)
    print("Strategy Comparison:")
    print("-"*60)
    print(f"{'Strategy':<20} {'Accuracy':<10} {'F1 (weighted)':<15} {'Cost':<10}")
    print("-"*60)
    for strategy in strategies:
        res = all_results["strategies"][strategy]
        print(f"{strategy:<20} {res['accuracy']:.3f}      {res['f1_weighted']:.3f}           ${res['total_cost']:.4f}")
    
    print("\n" + "="*60)
    print("Experiments complete!")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    print("\nGPT-5 Three-Class Stock Prediction Experiment")
    print("="*60)
    print("This will predict stock movements relative to market:")
    print("- UP: Outperform by >1%")
    print("- DOWN: Underperform by >1%")
    print("- STAY: Within ±1% of market")
    print("\nUsing GPT-5 summarized data from 2018")
    print("="*60)
    
    print("\nSelect reasoning effort level:")
    print("1. Minimal (fastest, cheapest)")
    print("2. Medium (balanced)")
    print("3. High (best quality, most expensive)")
    
    choice = input("Select option (1, 2, or 3): ")
    
    effort_map = {"1": "minimal", "2": "medium", "3": "high"}
    effort = effort_map.get(choice, "medium")
    
    print(f"\nRunning with effort level: {effort}")
    
    # Option to run on sample for testing
    print("\nRun on full test set or sample?")
    print("1. Sample (100 examples, ~10 minutes)")
    print("2. Full test set (~1000 examples, ~1.5 hours)")
    
    size_choice = input("Select option (1 or 2): ")
    
    sample_size = 100 if size_choice == "1" else None
    
    # Run experiments
    results = run_gpt5_experiments(effort_level=effort, sample_size=sample_size)