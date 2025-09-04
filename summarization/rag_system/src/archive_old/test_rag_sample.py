"""
Test RAG pipeline with a small sample
"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CHUNKS_DIR, RESULTS_DIR


def test_sample():
    """Test with pre-computed sample data"""
    
    # Load chunks
    chunks_file = CHUNKS_DIR / "chunks_2019.json"
    with open(chunks_file, 'r') as f:
        all_chunks = json.load(f)
    
    print(f"Total chunks available: {len(all_chunks)}")
    
    # Group by filing and take first 3 filings
    filings = {}
    for chunk in all_chunks:
        accession = chunk['accession']
        if accession not in filings:
            filings[accession] = []
        filings[accession].append(chunk)
        
        if len(filings) >= 3:
            break
    
    print(f"Testing with {len(filings)} filings")
    
    # Simulate retrieval (take top 5 chunks per filing)
    results = []
    for accession, chunks in filings.items():
        # Sort by length (proxy for importance)
        sorted_chunks = sorted(chunks, key=lambda x: x['chunk_length'], reverse=True)[:5]
        
        # Combine text
        combined_text = " [SEP] ".join([c['chunk_text'] for c in sorted_chunks])
        
        result = {
            'accession': accession,
            'ticker': chunks[0]['ticker'],
            'filing_date': chunks[0]['filing_date'],
            'retrieved_chunks': sorted_chunks,
            'combined_text': combined_text,
            'total_chars': len(combined_text),
            'signal': chunks[0].get('signal', 'UNKNOWN'),
            'adjusted_return_pct': chunks[0].get('adjusted_return_pct', 0.0)
        }
        results.append(result)
    
    # Save sample retrieved data
    sample_file = RESULTS_DIR / "sample_retrieved_chunks.json"
    with open(sample_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nSample data saved to: {sample_file}")
    
    # Show statistics
    for r in results:
        print(f"\nFiling: {r['ticker']} - {r['filing_date']}")
        print(f"  Chunks: {len(r['retrieved_chunks'])}")
        print(f"  Total chars: {r['total_chars']}")
        print(f"  True label: {r['signal']}")
        print(f"  True return: {r['adjusted_return_pct']:.2f}%")
    
    return results


def test_llm_generation():
    """Test LLM generation on sample"""
    from llm_generator import LLMGenerator
    
    # Load sample data
    sample_file = RESULTS_DIR / "sample_retrieved_chunks.json"
    with open(sample_file, 'r') as f:
        sample_data = json.load(f)
    
    # Initialize generator
    generator = LLMGenerator()
    
    # Test with first filing
    test_filing = sample_data[0]
    print(f"\nTesting LLM generation for: {test_filing['ticker']}")
    
    # Generate analysis
    analysis = generator.generate_analysis(test_filing)
    
    print("\nGenerated Analysis:")
    print(f"  Summary: {analysis['summary']}")
    print(f"  Sentiment: {analysis['sentiment']}")
    print(f"  Prediction: {analysis['prediction']}")
    print(f"  Confidence: {analysis['confidence']}")
    print(f"  Key Factors: {', '.join(analysis['key_factors'])}")
    print(f"  Reasoning: {analysis['reasoning']}")
    
    # Compare with true label
    print(f"\nTrue label: {test_filing['signal']}")
    print(f"True return: {test_filing['adjusted_return_pct']:.2f}%")
    
    # Save result
    output_file = RESULTS_DIR / "test_llm_result.json"
    with open(output_file, 'w') as f:
        json.dump({
            'filing_info': {
                'ticker': test_filing['ticker'],
                'date': test_filing['filing_date'],
                'true_label': test_filing['signal'],
                'true_return': test_filing['adjusted_return_pct']
            },
            'llm_analysis': analysis
        }, f, indent=2)
    
    print(f"\nResult saved to: {output_file}")


if __name__ == "__main__":
    print("=== Testing RAG Pipeline ===\n")
    
    # Step 1: Create sample data
    print("Step 1: Creating sample data...")
    test_sample()
    
    # Step 2: Test LLM generation
    print("\nStep 2: Testing LLM generation...")
    test_llm_generation()
    
    print("\n=== Test Complete ===")