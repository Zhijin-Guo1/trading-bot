"""
Set up ChromaDB vector store with chunk embeddings
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    CHUNKS_DIR, CHROMADB_DIR, EMBEDDINGS_DIR,
    EMBEDDING_MODEL, SIMILARITY_METRIC
)


class ChromaDBSetup:
    """Set up and populate ChromaDB with chunk embeddings"""
    
    def __init__(self, collection_name="eight_k_2019"):
        self.collection_name = collection_name
        
        # Initialize ChromaDB with persistent storage
        self.client = chromadb.PersistentClient(
            path=str(CHROMADB_DIR),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Initialize phase1_embedding model
        print(f"Loading embedding model: {EMBEDDING_MODEL}")
        if EMBEDDING_MODEL == "ProsusAI/finbert":
            # Use FinBERT for financial text
            self.embedder = SentenceTransformer('ProsusAI/finbert')
        else:
            # Default to general sentence transformer
            self.embedder = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
        
        # Create or get collection
        self.setup_collection()
    
    def setup_collection(self):
        """Create or reset the collection"""
        # Delete existing collection if it exists
        try:
            self.client.delete_collection(name=self.collection_name)
            print(f"Deleted existing collection: {self.collection_name}")
        except:
            pass
        
        # Create new collection with cosine similarity
        self.collection = self.client.create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": SIMILARITY_METRIC}  # Use cosine similarity
        )
        print(f"Created collection: {self.collection_name} with {SIMILARITY_METRIC} similarity")
    
    def batch_encode(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """Encode texts in batches for efficiency"""
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = self.embedder.encode(
                batch,
                convert_to_numpy=True,
                show_progress_bar=False
            )
            embeddings.append(batch_embeddings)
        
        return np.vstack(embeddings) if embeddings else np.array([])
    
    def add_chunks_to_collection(self, chunks: List[Dict], batch_size: int = 100):
        """Add chunks to ChromaDB collection in batches"""
        total_chunks = len(chunks)
        print(f"Adding {total_chunks} chunks to ChromaDB...")
        
        for i in tqdm(range(0, total_chunks, batch_size), desc="Adding to ChromaDB"):
            batch = chunks[i:i + batch_size]
            
            # Prepare data for ChromaDB
            ids = []
            texts = []
            metadatas = []
            
            for chunk in batch:
                # Create unique ID
                chunk_id = f"{chunk['accession']}_{chunk['chunk_id']}"
                ids.append(chunk_id)
                
                # Get text
                texts.append(chunk['chunk_text'])
                
                # Prepare metadata
                metadata = {
                    'accession': chunk['accession'],
                    'ticker': chunk['ticker'],
                    'filing_date': chunk['filing_date'],
                    'item_number': chunk['item_number'],
                    'chunk_position': chunk['chunk_position'],
                    'chunk_length': chunk['chunk_length'],
                    'signal': chunk.get('signal', 'UNKNOWN'),
                    'adjusted_return_pct': chunk.get('adjusted_return_pct', 0.0)
                }
                metadatas.append(metadata)
            
            # Generate embeddings
            embeddings = self.batch_encode(texts, batch_size=32)
            
            # Add to collection
            self.collection.add(
                ids=ids,
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas
            )
        
        print(f"Successfully added {total_chunks} chunks to ChromaDB")
    
    def test_retrieval(self, query: str, n_results: int = 5):
        """Test retrieval with a sample query"""
        print(f"\nTesting retrieval with query: '{query}'")
        
        # Encode query
        query_embedding = self.embedder.encode([query], convert_to_numpy=True)[0]
        
        # Search
        results = self.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results
        )
        
        print(f"Found {len(results['ids'][0])} results:")
        for i, (doc, metadata, distance) in enumerate(zip(
            results['documents'][0],
            results['metadatas'][0],
            results['distances'][0]
        )):
            print(f"\n{i+1}. Distance: {distance:.4f}")
            print(f"   Ticker: {metadata['ticker']}, Date: {metadata['filing_date']}")
            print(f"   Item: {metadata['item_number']}")
            print(f"   Text preview: {doc[:150]}...")
    
    def save_collection_stats(self):
        """Save statistics about the collection"""
        count = self.collection.count()
        
        stats = {
            'collection_name': self.collection_name,
            'total_chunks': count,
            'embedding_model': EMBEDDING_MODEL,
            'similarity_metric': SIMILARITY_METRIC
        }
        
        stats_file = CHROMADB_DIR / 'collection_stats.json'
        with open(stats_file, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nCollection statistics saved to: {stats_file}")
        return stats


def main():
    """Set up ChromaDB with 2019 8-K chunks"""
    
    # Load chunks
    chunks_file = CHUNKS_DIR / "chunks_2019.json"
    print(f"Loading chunks from: {chunks_file}")
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize ChromaDB
    db = ChromaDBSetup(collection_name="eight_k_2019")
    
    # Add chunks to collection
    db.add_chunks_to_collection(chunks, batch_size=100)
    
    # Save statistics
    stats = db.save_collection_stats()
    print(f"\nChromaDB setup complete!")
    print(f"Total chunks indexed: {stats['total_chunks']}")
    
    # Test retrieval with sample queries
    test_queries = [
        "quarterly earnings revenue profit guidance forecast",
        "CEO resignation departure executive change",
        "merger acquisition partnership agreement"
    ]
    
    for query in test_queries:
        db.test_retrieval(query, n_results=3)


if __name__ == "__main__":
    main()