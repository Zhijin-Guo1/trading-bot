"""
Test the enhanced RAG pipeline with 10 samples
"""
from pipeline_with_summarization import EnhancedRAGPipeline
import json
from pathlib import Path


def test_with_10_samples():
    """Run pipeline with 10 samples"""
    print("\n" + "="*60)
    print("TESTING RAG PIPELINE WITH 10 SAMPLES")
    print("="*60)
    
    # Initialize pipeline
    pipeline = EnhancedRAGPipeline()
    
    # Load and chunk 10 samples
    pipeline.load_and_chunk_filings(n=20)
    
    # Continue with rest of pipeline
    pipeline.setup_chromadb_and_tfidf()
    results = pipeline.process_all_filings()
    pipeline.evaluate_summaries(results)
    
    print("\n" + "="*60)
    print("10-SAMPLE TEST COMPLETE!")
    print(f"All outputs saved to: {pipeline.output_dir}")
    print("="*60)
    
    # Detailed analysis for 10 samples
    print("\n" + "="*60)
    print("DETAILED ANALYSIS OF 10 SAMPLES")
    print("="*60)
    
    # Overall statistics
    total_chunks = sum(len(r['retrieved_chunks']) for r in results)
    avg_summary_length = sum(len(r['summary']) for r in results) / len(results)
    avg_score = sum(r['avg_retrieval_score'] for r in results) / len(results)
    
    print(f"\n=== Overall Statistics ===")
    print(f"Total filings processed: {len(results)}")
    print(f"Total chunks created: {len(pipeline.chunks)}")
    print(f"Total chunks retrieved: {total_chunks}")
    print(f"Average chunks per filing: {total_chunks/len(results):.1f}")
    print(f"Average summary length: {avg_summary_length:.0f} characters")
    print(f"Average retrieval score: {avg_score:.3f}")
    
    # Label distribution
    labels = [r['true_label'] for r in results]
    print(f"\n=== Label Distribution ===")
    print(f"UP:   {labels.count('UP')} filings ({labels.count('UP')/len(labels)*100:.0f}%)")
    print(f"DOWN: {labels.count('DOWN')} filings ({labels.count('DOWN')/len(labels)*100:.0f}%)")
    print(f"STAY: {labels.count('STAY')} filings ({labels.count('STAY')/len(labels)*100:.0f}%)")
    
    # Return statistics
    returns = [r['true_return'] for r in results]
    print(f"\n=== Return Statistics ===")
    print(f"Min return:  {min(returns):>7.2f}%")
    print(f"Max return:  {max(returns):>7.2f}%")
    print(f"Avg return:  {sum(returns)/len(returns):>7.2f}%")
    print(f"Volatility:  {(sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns))**0.5:>7.2f}%")
    
    # Company distribution
    tickers = [r['ticker'] for r in results]
    unique_tickers = list(set(tickers))
    print(f"\n=== Company Distribution ===")
    print(f"Unique companies: {len(unique_tickers)}")
    for ticker in unique_tickers:
        count = tickers.count(ticker)
        print(f"  {ticker}: {count} filing{'s' if count > 1 else ''}")
    
    # Summary quality by filing type
    print(f"\n=== Summary Examples by Outcome ===")
    
    # Find best UP example
    up_samples = [r for r in results if r['true_label'] == 'UP']
    if up_samples:
        best_up = max(up_samples, key=lambda x: x['true_return'])
        print(f"\nBest UP example ({best_up['ticker']}, +{best_up['true_return']:.2f}%):")
        print(f"  Summary preview: {best_up['summary'][:200]}...")
    
    # Find worst DOWN example
    down_samples = [r for r in results if r['true_label'] == 'DOWN']
    if down_samples:
        worst_down = min(down_samples, key=lambda x: x['true_return'])
        print(f"\nWorst DOWN example ({worst_down['ticker']}, {worst_down['true_return']:.2f}%):")
        print(f"  Summary preview: {worst_down['summary'][:200]}...")
    
    # Create summary report
    report = {
        'run_id': pipeline.run_id,
        'num_samples': len(results),
        'total_chunks_created': len(pipeline.chunks),
        'avg_chunks_retrieved': total_chunks/len(results),
        'avg_summary_length': avg_summary_length,
        'avg_retrieval_score': avg_score,
        'label_distribution': {
            'UP': labels.count('UP'),
            'DOWN': labels.count('DOWN'),
            'STAY': labels.count('STAY')
        },
        'return_stats': {
            'min': min(returns),
            'max': max(returns),
            'avg': sum(returns)/len(returns),
            'volatility': (sum((r - sum(returns)/len(returns))**2 for r in returns) / len(returns))**0.5
        },
        'companies': unique_tickers
    }
    
    # Save report
    report_file = pipeline.output_dir / "analysis_report.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n=== Analysis report saved to: {report_file} ===")
    
    return results


if __name__ == "__main__":
    results = test_with_10_samples()