"""
Automated script to run GPT-5 experiment with medium effort and 100 samples
"""

import sys
import os
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import the experiment module
from modeling.phase2_llm_api.llm_experiment_gpt5 import run_gpt5_experiments

def main():
    """Run GPT-5 experiment with predefined settings"""
    
    # Check for API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY environment variable not set")
        return 1
    
    print("\n" + "="*60)
    print("AUTOMATED GPT-5 EXPERIMENT")
    print("="*60)
    print("Settings:")
    print("  - Model: GPT-5")
    print("  - Effort: medium")
    print("  - Sample size: 100 examples")
    print("  - Strategies: zero_shot, few_shot, chain_of_thought")
    print("="*60)
    
    # Run experiments with medium effort and 100 samples
    try:
        results = run_gpt5_experiments(effort_level="medium", sample_size=100)
        print("\n✓ Experiment completed successfully!")
        return 0
    except Exception as e:
        print(f"\n✗ Error during experiment: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)