"""
Complete RAG pipeline: Retrieval + Generation for stock prediction
"""
import json
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
from tqdm import tqdm
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from rag_retriever import RAGRetriever
from llm_generator import LLMGenerator
from config.settings import CHUNKS_DIR, RESULTS_DIR, TOP_K_CHUNKS


class RAGPipeline:
    """Complete RAG pipeline for 8-K analysis"""
    
    def __init__(self, year="2019"):
        self.year = year
        self.retriever = None
        self.generator = None
        
    def initialize(self):
        """Initialize retrieval and generation components"""
        print("Initializing RAG pipeline...")
        self.retriever = RAGRetriever(year=self.year)
        self.generator = LLMGenerator()
        print("Pipeline initialized successfully")
    
    def process_filing(self, accession: str) -> Dict:
        """Process a single filing through the RAG pipeline"""
        
        # Step 1: Retrieve relevant chunks
        retrieved_chunks = self.retriever.retrieve_for_filing(accession, top_k=TOP_K_CHUNKS)
        
        if not retrieved_chunks:
            return None
        
        # Get filing metadata
        first_chunk = retrieved_chunks[0]
        combined_text = " [SEP] ".join([c['chunk_text'] for c in retrieved_chunks])
        
        retrieved_data = {
            'accession': accession,
            'ticker': first_chunk['ticker'],
            'filing_date': first_chunk['filing_date'],
            'retrieved_chunks': retrieved_chunks,
            'combined_text': combined_text,
            'total_chars': len(combined_text),
            'signal': first_chunk.get('signal', 'UNKNOWN'),
            'adjusted_return_pct': first_chunk.get('adjusted_return_pct', 0.0)
        }
        
        # Step 2: Generate analysis with LLM
        analysis = self.generator.generate_analysis(retrieved_data)
        
        # Step 3: Combine everything
        result = {
            **retrieved_data,
            'llm_analysis': analysis,
            'llm_summary': analysis['summary'],
            'llm_sentiment': analysis['sentiment'],
            'llm_key_factors': analysis['key_factors'],
            'llm_prediction': analysis['prediction'],
            'llm_confidence': analysis['confidence'],
            'llm_reasoning': analysis['reasoning']
        }
        
        return result
    
    def process_batch(self, accessions: List[str], save_intermediate: bool = True) -> List[Dict]:
        """Process multiple filings"""
        results = []
        failed = []
        
        for accession in tqdm(accessions, desc="Processing filings"):
            try:
                result = self.process_filing(accession)
                if result:
                    results.append(result)
                    
                    # Save intermediate results
                    if save_intermediate and len(results) % 10 == 0:
                        self.save_intermediate_results(results)
                else:
                    failed.append(accession)
                    
            except Exception as e:
                print(f"Error processing {accession}: {e}")
                failed.append(accession)
        
        if failed:
            print(f"Failed to process {len(failed)} filings: {failed[:5]}...")
        
        return results
    
    def save_intermediate_results(self, results: List[Dict]):
        """Save intermediate results during processing"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = RESULTS_DIR / f"rag_results_intermediate_{timestamp}.json"
        
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"Saved {len(results)} intermediate results to {output_file}")
    
    def evaluate_predictions(self, results: List[Dict]) -> Dict:
        """Evaluate prediction accuracy"""
        correct = 0
        total = 0
        
        predictions_by_class = {'UP': [], 'DOWN': [], 'STAY': []}
        
        for result in results:
            true_label = result.get('signal', 'UNKNOWN')
            pred_label = result.get('llm_prediction', 'UNKNOWN')
            
            if true_label != 'UNKNOWN' and pred_label != 'UNKNOWN':
                total += 1
                if true_label == pred_label:
                    correct += 1
                
                predictions_by_class[true_label].append(pred_label)
        
        accuracy = correct / total if total > 0 else 0
        
        # Calculate per-class metrics
        class_metrics = {}
        for true_class in ['UP', 'DOWN', 'STAY']:
            preds = predictions_by_class[true_class]
            if preds:
                class_correct = sum(1 for p in preds if p == true_class)
                class_accuracy = class_correct / len(preds)
                class_metrics[true_class] = {
                    'accuracy': class_accuracy,
                    'total': len(preds),
                    'correct': class_correct
                }
        
        return {
            'overall_accuracy': accuracy,
            'total_evaluated': total,
            'correct_predictions': correct,
            'class_metrics': class_metrics
        }
    
    def save_results(self, results: List[Dict], filename: Optional[str] = None):
        """Save final results"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"rag_results_{self.year}_{timestamp}.json"
        
        output_file = RESULTS_DIR / filename
        
        # Save full results
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Create summary DataFrame
        summary_data = []
        for r in results:
            summary_data.append({
                'accession': r['accession'],
                'ticker': r['ticker'],
                'filing_date': r['filing_date'],
                'true_label': r['signal'],
                'true_return': r['adjusted_return_pct'],
                'llm_prediction': r['llm_prediction'],
                'llm_confidence': r['llm_confidence'],
                'llm_sentiment': r['llm_sentiment'],
                'num_chunks': len(r['retrieved_chunks']),
                'total_chars': r['total_chars']
            })
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = RESULTS_DIR / f"rag_summary_{self.year}_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        print(f"Results saved to: {output_file}")
        print(f"Summary saved to: {summary_file}")
        
        return output_file, summary_file


def main():
    """Run full RAG pipeline on sample data"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run RAG pipeline')
    parser.add_argument('--year', type=str, default='2019', help='Year to process')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to process')
    parser.add_argument('--full', action='store_true', help='Process all filings')
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = RAGPipeline(year=args.year)
    pipeline.initialize()
    
    # Get accessions to process
    if args.full:
        accessions = list(pipeline.retriever.filing_to_chunks.keys())
        print(f"Processing all {len(accessions)} filings...")
    else:
        accessions = list(pipeline.retriever.filing_to_chunks.keys())[:args.samples]
        print(f"Processing {len(accessions)} sample filings...")
    
    # Process filings
    results = pipeline.process_batch(accessions)
    
    # Evaluate predictions
    if results:
        evaluation = pipeline.evaluate_predictions(results)
        print("\n=== Evaluation Results ===")
        print(f"Overall Accuracy: {evaluation['overall_accuracy']:.2%}")
        print(f"Total Evaluated: {evaluation['total_evaluated']}")
        
        print("\nPer-Class Accuracy:")
        for class_name, metrics in evaluation['class_metrics'].items():
            print(f"  {class_name}: {metrics['accuracy']:.2%} ({metrics['correct']}/{metrics['total']})")
        
        # Save results
        output_file, summary_file = pipeline.save_results(results)
        
        # Show sample result
        print("\n=== Sample Result ===")
        sample = results[0]
        print(f"Ticker: {sample['ticker']}")
        print(f"Date: {sample['filing_date']}")
        print(f"True Label: {sample['signal']}")
        print(f"Prediction: {sample['llm_prediction']} (confidence: {sample['llm_confidence']:.2f})")
        print(f"Summary: {sample['llm_summary'][:200]}...")
        print(f"Key Factors: {', '.join(sample['llm_key_factors'])}")


if __name__ == "__main__":
    main()