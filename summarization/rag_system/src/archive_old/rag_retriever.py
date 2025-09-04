"""
Simple retrieval system using numpy and cosine similarity
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    CHUNKS_DIR, EMBEDDINGS_DIR, QUERY_TEMPLATES,
    TOP_K_CHUNKS, EMBEDDING_MODEL
)


class RAGRetriever:
    """Retrieve relevant chunks using cosine similarity"""
    
    def __init__(self, year="2019"):
        self.year = year
        self.chunks = []
        self.embeddings = None
        self.filing_to_chunks = {}  # Map filing to chunk indices
        
        # Initialize phase1_embedding model
        print(f"Loading embedding model...")
        if EMBEDDING_MODEL == "ProsusAI/finbert":
            self.embedder = SentenceTransformer('ProsusAI/finbert')
        else:
            self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Load chunks and embeddings
        self.load_data()
    
    def load_data(self):
        """Load chunks and create/load embeddings"""
        # Load chunks
        chunks_file = CHUNKS_DIR / f"chunks_{self.year}.json"
        print(f"Loading chunks from: {chunks_file}")
        
        with open(chunks_file, 'r') as f:
            self.chunks = json.load(f)
        
        print(f"Loaded {len(self.chunks)} chunks")
        
        # Group chunks by filing
        for idx, chunk in enumerate(self.chunks):
            accession = chunk['accession']
            if accession not in self.filing_to_chunks:
                self.filing_to_chunks[accession] = []
            self.filing_to_chunks[accession].append(idx)
        
        # Load or create embeddings
        embeddings_file = EMBEDDINGS_DIR / f"chunk_embeddings_{self.year}.npy"
        
        if embeddings_file.exists():
            print(f"Loading existing embeddings from: {embeddings_file}")
            self.embeddings = np.load(embeddings_file)
        else:
            print(f"Creating embeddings for {len(self.chunks)} chunks...")
            self.create_embeddings(embeddings_file)
    
    def create_embeddings(self, output_file: Path):
        """Create embeddings for all chunks"""
        texts = [chunk['chunk_text'] for chunk in self.chunks]
        
        # Batch encode
        batch_size = 32
        embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Creating embeddings"):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        self.embeddings = np.vstack(embeddings)
        
        # Save embeddings
        np.save(output_file, self.embeddings)
        print(f"Saved embeddings to: {output_file}")
    
    def retrieve_for_filing(self, accession: str, top_k: int = TOP_K_CHUNKS) -> List[Dict]:
        """Retrieve top-k chunks for a specific filing"""
        if accession not in self.filing_to_chunks:
            print(f"Warning: Accession {accession} not found")
            return []
        
        # Get chunk indices for this filing
        filing_chunk_indices = self.filing_to_chunks[accession]
        filing_embeddings = self.embeddings[filing_chunk_indices]
        
        # Create query embeddings
        all_scores = np.zeros(len(filing_chunk_indices))
        
        for query_type, query_text in QUERY_TEMPLATES.items():
            # Encode query
            query_embedding = self.embedder.encode([query_text], convert_to_numpy=True)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, filing_embeddings)[0]
            
            # Add to cumulative scores
            all_scores += similarities
        
        # Get top-k chunk indices
        top_indices = np.argsort(all_scores)[-top_k:][::-1]
        
        # Get the actual chunks
        retrieved_chunks = []
        for idx in top_indices:
            global_idx = filing_chunk_indices[idx]
            chunk = self.chunks[global_idx].copy()
            chunk['retrieval_score'] = float(all_scores[idx])
            retrieved_chunks.append(chunk)
        
        return retrieved_chunks
    
    def process_all_filings(self) -> List[Dict]:
        """Process all filings and save retrieved chunks"""
        results = []
        
        # Get unique accessions
        accessions = list(self.filing_to_chunks.keys())
        
        print(f"Processing {len(accessions)} filings...")
        
        for accession in tqdm(accessions[:100], desc="Retrieving chunks"):  # Process first 100 for testing
            # Retrieve top chunks
            retrieved_chunks = self.retrieve_for_filing(accession)
            
            if retrieved_chunks:
                # Get filing metadata from first chunk
                first_chunk = retrieved_chunks[0]
                
                # Combine chunk texts
                combined_text = " [SEP] ".join([c['chunk_text'] for c in retrieved_chunks])
                
                result = {
                    'accession': accession,
                    'ticker': first_chunk['ticker'],
                    'filing_date': first_chunk['filing_date'],
                    'retrieved_chunks': [
                        {
                            'chunk_id': c['chunk_id'],
                            'item_number': c['item_number'],
                            'chunk_text': c['chunk_text'],
                            'retrieval_score': c['retrieval_score']
                        } for c in retrieved_chunks
                    ],
                    'combined_text': combined_text,
                    'total_chars': len(combined_text),
                    'signal': first_chunk.get('signal', 'UNKNOWN'),
                    'adjusted_return_pct': first_chunk.get('adjusted_return_pct', 0.0)
                }
                results.append(result)
        
        return results


def main():
    """Run retrieval on 2019 data"""
    retriever = RAGRetriever(year="2019")
    
    # Process all filings
    results = retriever.process_all_filings()
    
    # Save results
    output_file = CHUNKS_DIR / "retrieved_chunks_2019.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nRetrieval complete!")
    print(f"Processed {len(results)} filings")
    print(f"Results saved to: {output_file}")
    
    # Show statistics
    if results:
        avg_chars = np.mean([r['total_chars'] for r in results])
        print(f"Average combined text length: {avg_chars:.0f} chars")
        
        # Show sample
        sample = results[0]
        print(f"\nSample retrieval:")
        print(f"  Ticker: {sample['ticker']}")
        print(f"  Date: {sample['filing_date']}")
        print(f"  Retrieved {len(sample['retrieved_chunks'])} chunks")
        print(f"  Total chars: {sample['total_chars']}")
        print(f"  Top chunk item: {sample['retrieved_chunks'][0]['item_number']}")


if __name__ == "__main__":
    main()