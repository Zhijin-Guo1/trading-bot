"""
Test Direct GPT Summarization with a few examples
"""
import sys
from pathlib import Path

# Add paths for imports
sys.path.append(str(Path(__file__).parent.parent))
sys.path.append(str(Path(__file__).parent.parent / 'rag_system'))

from gpt_summarization import DirectFilingSummarizer

def test_with_samples():
    """Run direct summarization on just 5 samples"""
    print("\n" + "="*60)
    print("TESTING DIRECT GPT SUMMARIZATION")
    print("Testing with 5 samples from 2019 data")
    print("="*60)
    
    summarizer = DirectFilingSummarizer()
    
    # Run on just 5 samples
    results = summarizer.run_full_pipeline(n_samples=5)
    
    print("\n" + "="*60)
    print("TEST COMPLETE!")
    print(f"Processed {len(results)} filings")
    print("="*60)
    
    # Display sample summaries
    print("\n" + "="*60)
    print("SAMPLE SUMMARIES")
    print("="*60)
    
    for i, result in enumerate(results[:3], 1):
        if result['status'] == 'success':
            print(f"\n{i}. {result['ticker']} ({result['filing_date']})")
            print(f"   True Label: {result['true_label']} ({result['true_return']:.2f}%)")
            print(f"   Text Length: {result.get('text_length', 'N/A')} chars")
            print(f"   Tokens Used: {result.get('tokens_used', 'N/A')}")
            print(f"   Cost: ${result.get('cost', 0):.4f}")
            print(f"   Summary Preview: {result['summary'][:200]}...")
    
    return results

if __name__ == "__main__":
    results = test_with_samples()