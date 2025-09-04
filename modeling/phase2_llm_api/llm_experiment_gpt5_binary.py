"""
GPT-5 Binary Stock Prediction Experiment
Uses GPT-5 summarized data from 2018 for training and testing
Predicts: UP (positive return) or DOWN (negative return)
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


class StockPredictionGPT5Binary:
    """GPT-5 based binary stock prediction from 8-K summaries"""
    
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
        """Zero-shot prompt for binary prediction"""
        return f"""Analyze this 8-K filing summary and predict stock performance relative to the market.

Task: Predict if the stock will:
- UP: Have positive market-adjusted return over next 5 trading days
- DOWN: Have negative market-adjusted return over next 5 trading days

8-K Summary:
{summary}

Answer with only one word: UP or DOWN"""

    def create_few_shot_prompt(self, summary: str, examples: List[Dict]) -> str:
        """
        Few-shot prompt with balanced examples (3 UP, 3 DOWN)
        """
        prompt = """Task: Predict stock performance relative to market based on 8-K summaries.

Categories:
- UP: Positive market-adjusted return over next 5 days
- DOWN: Negative market-adjusted return over next 5 days

Here are examples from historical data:
"""
        
        # Use 6 examples: 3 UP, 3 DOWN for balance
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
2. Assess if these events are materially positive or negative
3. Consider likely investor reaction and market sentiment
4. Evaluate the magnitude of potential impact
5. Consider any forward guidance or future outlook mentioned

Based on your analysis, predict the stock's 5-day performance relative to market:
- UP: Positive market-adjusted return
- DOWN: Negative market-adjusted return

Provide brief reasoning (2-3 sentences), then state your final prediction as: UP or DOWN"""

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
            
            # Parse prediction for binary classification
            response_upper = full_response.upper()
            
            # Look for UP or DOWN
            if "UP" in response_upper and "DOWN" not in response_upper[:response_upper.index("UP")]:
                prediction = "UP"
                confidence = 0.9 if response_upper.count("UP") > response_upper.count("DOWN") else 0.7
            elif "DOWN" in response_upper:
                prediction = "DOWN"
                confidence = 0.9 if response_upper.count("DOWN") > response_upper.count("UP") else 0.7
            else:
                # Default to DOWN if unclear (conservative approach)
                prediction = "DOWN"
                confidence = 0.5
            
            return prediction, confidence
            
        except Exception as e:
            print(f"Error getting prediction: {e}")
            return "DOWN", 0.5

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
        
        # Calculate metrics for binary classification
        label_map = {"UP": 1, "DOWN": 0}
        pred_numeric = [label_map[p] for p in predictions]
        true_numeric = [label_map[t] for t in true_labels]
        
        accuracy = accuracy_score(true_numeric, pred_numeric)
        precision = precision_score(true_numeric, pred_numeric, pos_label=1, zero_division=0)
        recall = recall_score(true_numeric, pred_numeric, pos_label=1, zero_division=0)
        f1 = f1_score(true_numeric, pred_numeric, pos_label=1, zero_division=0)
        
        # Calculate metrics for both classes
        precision_both = precision_score(true_numeric, pred_numeric, average=None, zero_division=0)
        recall_both = recall_score(true_numeric, pred_numeric, average=None, zero_division=0)
        f1_both = f1_score(true_numeric, pred_numeric, average=None, zero_division=0)
        
        cm = confusion_matrix(true_numeric, pred_numeric)
        
        # Calculate distribution statistics
        pred_dist = {label: predictions.count(label) / len(predictions) for label in ["UP", "DOWN"]}
        true_dist = {label: true_labels.count(label) / len(true_labels) for label in ["UP", "DOWN"]}
        avg_confidence = np.mean(confidences)
        avg_response_time = np.mean(self.response_times) if self.response_times else 0
        
        results = {
            "strategy": strategy,
            "effort": effort,
            "dataset_size": len(data),
            "accuracy": accuracy,
            "precision": precision,  # For UP class (positive)
            "recall": recall,        # For UP class (positive)
            "f1_score": f1,         # For UP class (positive)
            "precision_per_class": {"DOWN": precision_both[0], "UP": precision_both[1]},
            "recall_per_class": {"DOWN": recall_both[0], "UP": recall_both[1]},
            "f1_per_class": {"DOWN": f1_both[0], "UP": f1_both[1]},
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
        print(f"Precision (UP): {precision:.3f}")
        print(f"Recall (UP): {recall:.3f}")
        print(f"F1 Score (UP): {f1:.3f}")
        print(f"\nPer-class F1 scores:")
        print(f"  DOWN: {f1_both[0]:.3f}")
        print(f"  UP:   {f1_both[1]:.3f}")
        print(f"\nPredicted distribution: UP={pred_dist['UP']:.2f}, DOWN={pred_dist['DOWN']:.2f}")
        print(f"Actual distribution:    UP={true_dist['UP']:.2f}, DOWN={true_dist['DOWN']:.2f}")
        print(f"\nTotal cost: ${self.total_cost:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"           Predicted:")
        print(f"           DOWN   UP")
        print(f"Actual DOWN:  {cm[0][0]:4d}  {cm[0][1]:4d}")
        print(f"       UP:    {cm[1][0]:4d}  {cm[1][1]:4d}")
        
        return results


def load_gpt5_summaries(year: int = 2018):
    """
    Load GPT-5 summarized data from specified year
    Returns train/test split with binary labels
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
    
    # Convert to binary labels based on true_return
    # Positive return = UP, Negative return = DOWN
    
    for item in successful_data:
        return_val = item.get('true_return', 0)
        if return_val > 0:
            item['true_label'] = 'UP'
        else:
            item['true_label'] = 'DOWN'
    
    # Print distribution
    labels = [d['true_label'] for d in successful_data]
    print(f"Label distribution: UP={labels.count('UP')}, DOWN={labels.count('DOWN')}")
    
    # Split into train/test (80/20)
    train_data, test_data = train_test_split(successful_data, test_size=0.2, random_state=42, stratify=labels)
    
    print(f"Train set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    return train_data, test_data


def get_balanced_examples(train_data: List[Dict], n_per_class: int = 3) -> List[Dict]:
    """
    Get balanced training examples for few-shot learning
    Returns n_per_class examples for each of UP and DOWN
    """
    import random
    random.seed(42)
    
    # Group by label
    up_examples = [d for d in train_data if d['true_label'] == 'UP']
    down_examples = [d for d in train_data if d['true_label'] == 'DOWN']
    
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
    
    print(f"Selected {len(examples)} balanced examples for few-shot learning")
    return examples


def run_gpt5_binary_experiments(effort_level: str = "medium", sample_size: int = None):
    """
    Run GPT-5 experiments with binary classification
    
    Args:
        effort_level: "minimal", "medium", or "high" for GPT-5 reasoning
        sample_size: If provided, use only this many test samples (for faster testing)
    """
    
    print("\n" + "="*60)
    print(f"Starting GPT-5 Binary Classification Experiments")
    print(f"Reasoning effort: {effort_level}")
    print("="*60)
    
    # Load data
    train_data, test_data = load_gpt5_summaries(year=2018)
    
    # Get balanced examples for few-shot
    training_examples = get_balanced_examples(train_data, n_per_class=3)
    
    # Optionally sample test data for faster experiments
    if sample_size and sample_size < len(test_data):
        import random
        random.seed(42)
        test_data = random.sample(test_data, sample_size)
        print(f"\nUsing sampled test set: {sample_size} samples")
    
    # Create results directory
    results_dir = Path("results_gpt5_binary")
    results_dir.mkdir(exist_ok=True)
    
    # Strategies to test
    strategies = ["zero_shot", "few_shot", "chain_of_thought"]
    
    all_results = {
        "model": "gpt-5",
        "classification_type": "binary",
        "effort": effort_level,
        "data_year": 2018,
        "test_size": len(test_data),
        "train_size": len(train_data),
        "timestamp": datetime.now().isoformat(),
        "strategies": {}
    }
    
    for strategy in strategies:
        # Create new predictor for each strategy to reset costs
        predictor = StockPredictionGPT5Binary(model="gpt-5")
        
        # Run evaluation
        if strategy == "few_shot":
            results = predictor.evaluate_strategy(test_data, strategy, training_examples, effort_level)
        else:
            results = predictor.evaluate_strategy(test_data, strategy, effort=effort_level)
        
        all_results["strategies"][strategy] = results
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = results_dir / f"gpt5_binary_results_{effort_level}_{timestamp}.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Print comparison
    print("\n" + "="*60)
    print("Strategy Comparison:")
    print("-"*60)
    print(f"{'Strategy':<20} {'Accuracy':<10} {'F1 (UP)':<10} {'Cost':<10}")
    print("-"*60)
    for strategy in strategies:
        res = all_results["strategies"][strategy]
        print(f"{strategy:<20} {res['accuracy']:.3f}      {res['f1_score']:.3f}      ${res['total_cost']:.4f}")
    
    print("\n" + "="*60)
    print("Binary Classification Experiments Complete!")
    print("="*60)
    
    return all_results


if __name__ == "__main__":
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        exit(1)
    
    print("\nGPT-5 Binary Stock Prediction Experiment")
    print("="*60)
    print("This will predict stock movements relative to market:")
    print("- UP: Positive market-adjusted return")
    print("- DOWN: Negative market-adjusted return")
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
    results = run_gpt5_binary_experiments(effort_level=effort, sample_size=sample_size)