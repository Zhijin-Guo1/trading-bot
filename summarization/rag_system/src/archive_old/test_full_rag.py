"""
Test full RAG pipeline with proper retrieval and multiple samples
"""
import json
import numpy as np
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from llm_generator import LLMGenerator
from config.settings import (
    CHUNKS_DIR, EMBEDDINGS_DIR, RESULTS_DIR,
    QUERY_TEMPLATES, EMBEDDING_MODEL
)


def test_with_proper_retrieval(n_samples=3):
    """Test RAG with actual phase1_embedding-based retrieval"""
    
    # Check if embeddings exist
    embeddings_file = EMBEDDINGS_DIR / "chunk_embeddings_2019.npy"
    if not embeddings_file.exists():
        print("Embeddings not found. Please run create_embeddings.py first.")
        return
    
    # Load embeddings
    print("Loading embeddings...")
    embeddings = np.load(embeddings_file)
    
    # Load chunks
    with open(CHUNKS_DIR / "chunks_2019.json", 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks with {embeddings.shape[0]} embeddings")
    
    # Group chunks by filing
    filings = {}
    for i, chunk in enumerate(chunks):
        accession = chunk['accession']
        if accession not in filings:
            filings[accession] = []
        filings[accession].append((i, chunk))
    
    # Initialize phase1_embedding model for queries
    print("Loading query encoder...")
    if EMBEDDING_MODEL == "ProsusAI/finbert":
        model = SentenceTransformer('ProsusAI/finbert')
    else:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Process first n_samples filings
    sample_filings = list(filings.items())[:n_samples]
    results = []
    
    print(f"\nProcessing {n_samples} filings with proper retrieval...")
    
    for accession, filing_chunks in sample_filings:
        print(f"\nFiling: {accession}")
        
        # Get indices and embeddings for this filing
        indices = [i for i, _ in filing_chunks]
        filing_embeddings = embeddings[indices]
        
        # Score chunks against all queries
        all_scores = np.zeros(len(filing_chunks))
        
        for query_type, query_text in QUERY_TEMPLATES.items():
            # Encode query
            query_embedding = model.encode([query_text], convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, filing_embeddings)[0]
            all_scores += similarities
            
            print(f"  Query '{query_type}': max similarity = {similarities.max():.3f}")
        
        # Get top 5 chunks
        top_indices = np.argsort(all_scores)[-5:][::-1]
        
        # Retrieve top chunks
        retrieved_chunks = []
        for idx in top_indices:
            _, chunk = filing_chunks[idx]
            chunk_copy = chunk.copy()
            chunk_copy['retrieval_score'] = float(all_scores[idx])
            retrieved_chunks.append(chunk_copy)
        
        # Prepare result
        combined_text = " [SEP] ".join([c['chunk_text'] for c in retrieved_chunks])
        
        result = {
            'accession': accession,
            'ticker': retrieved_chunks[0]['ticker'],
            'filing_date': retrieved_chunks[0]['filing_date'],
            'retrieved_chunks': retrieved_chunks,
            'combined_text': combined_text,
            'total_chars': len(combined_text),
            'signal': retrieved_chunks[0].get('signal', 'UNKNOWN'),
            'adjusted_return_pct': retrieved_chunks[0].get('adjusted_return_pct', 0.0)
        }
        
        print(f"  Retrieved {len(retrieved_chunks)} chunks, total {len(combined_text)} chars")
        print(f"  Top chunk item: {retrieved_chunks[0]['item_number']}")
        print(f"  True label: {result['signal']}, Return: {result['adjusted_return_pct']:.2f}%")
        
        results.append(result)
    
    # Save retrieved results
    retrieved_file = RESULTS_DIR / "properly_retrieved_chunks.json"
    with open(retrieved_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRetrieved chunks saved to: {retrieved_file}")
    
    # Generate LLM analysis for ALL samples
    print("\n=== Testing LLM Generation for All Samples ===")
    generator = LLMGenerator()
    
    llm_results = []
    for i, result in enumerate(results, 1):
        print(f"\n{i}. Generating analysis for {result['ticker']} - {result['filing_date']}")
        
        analysis = generator.generate_analysis(result)
        
        llm_result = {
            'filing_info': {
                'accession': result['accession'],
                'ticker': result['ticker'],
                'date': result['filing_date'],
                'true_label': result['signal'],
                'true_return': result['adjusted_return_pct']
            },
            'llm_analysis': analysis
        }
        
        print(f"   Prediction: {analysis['prediction']} (confidence: {analysis['confidence']:.2f})")
        print(f"   True label: {result['signal']}")
        print(f"   Match: {'✓' if analysis['prediction'] == result['signal'] else '✗'}")
        
        llm_results.append(llm_result)
    
    # Save all LLM results
    llm_file = RESULTS_DIR / "all_llm_results.json"
    with open(llm_file, 'w') as f:
        json.dump(llm_results, f, indent=2)
    
    print(f"\nAll LLM results saved to: {llm_file}")
    
    # Calculate accuracy
    correct = sum(1 for r in llm_results 
                  if r['llm_analysis']['prediction'] == r['filing_info']['true_label'])
    accuracy = correct / len(llm_results) if llm_results else 0
    
    print(f"\n=== Final Results ===")
    print(f"Accuracy: {accuracy:.1%} ({correct}/{len(llm_results)})")
    
    return llm_results


if __name__ == "__main__":
    test_with_proper_retrieval(n_samples=3)