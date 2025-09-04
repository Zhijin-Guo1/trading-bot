"""
Set up ChromaDB with OpenAI embeddings (avoiding sentence-transformers conflict)
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
import chromadb
from chromadb.utils import embedding_functions
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    CHUNKS_DIR, CHROMADB_DIR, OPENAI_API_KEY
)


def setup_chromadb_with_openai():
    """Set up ChromaDB using OpenAI embeddings"""
    
    print("=== Setting up ChromaDB with OpenAI Embeddings ===")
    
    # 1. Initialize ChromaDB client
    print("\n1. Initializing ChromaDB client...")
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    
    # 2. Create OpenAI phase1_embedding function
    print("2. Setting up OpenAI phase1_embedding function...")
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-phase1_embedding-ada-002"
    )
    
    # 3. Delete existing collection if exists
    try:
        client.delete_collection(name="eight_k_chunks")
        print("   Deleted existing collection")
    except:
        print("   No existing collection to delete")
    
    # 4. Create new collection
    print("3. Creating new collection with cosine similarity...")
    collection = client.create_collection(
        name="eight_k_chunks",
        embedding_function=openai_ef,
        metadata={"hnsw:space": "cosine"}
    )
    
    # 5. Load chunks from recent test run
    # Try to find most recent pipeline run
    pipeline_runs = list(Path(CHROMADB_DIR).parent.glob("pipeline_run_*"))
    
    if pipeline_runs:
        # Use most recent run
        latest_run = sorted(pipeline_runs)[-1]
        chunks_file = latest_run / "chunks.json"
        print(f"\n4. Loading chunks from {chunks_file}")
        
        with open(chunks_file, 'r') as f:
            test_chunks = json.load(f)
    else:
        # Fallback: create sample chunks
        print("\n4. No existing chunks found. Creating sample chunks...")
        from test_complete_pipeline_v2 import CompletePipelineV2
        pipeline = CompletePipelineV2()
        filings = pipeline.load_sample_filings(n=3)
        test_chunks = []
        for filing in filings:
            chunks = pipeline.create_chunks(filing['text'], filing)
            test_chunks.extend(chunks)
    
    print(f"   Loaded {len(test_chunks)} chunks for testing")
    
    # 6. Prepare data for ChromaDB
    print("\n5. Adding chunks to ChromaDB...")
    
    ids = []
    documents = []
    metadatas = []
    
    for i, chunk in enumerate(test_chunks):
        # Create unique ID
        chunk_id = f"{chunk['accession']}_{i}"
        ids.append(chunk_id)
        
        # Document text
        documents.append(chunk['chunk_text'])
        
        # Metadata
        metadata = {
            'accession': chunk['accession'],
            'ticker': chunk['ticker'],
            'filing_date': chunk['filing_date'],
            'item_number': chunk.get('item_number', 'UNKNOWN'),
            'chunk_position': chunk.get('chunk_position', i),
            'signal': chunk.get('signal', 'UNKNOWN'),
            'return': float(chunk.get('adjusted_return_pct', 0.0))
        }
        metadatas.append(metadata)
    
    # Add to collection in batches
    batch_size = 20
    for i in tqdm(range(0, len(ids), batch_size), desc="Adding to ChromaDB"):
        batch_ids = ids[i:i+batch_size]
        batch_docs = documents[i:i+batch_size]
        batch_meta = metadatas[i:i+batch_size]
        
        collection.add(
            ids=batch_ids,
            documents=batch_docs,
            metadatas=batch_meta
        )
    
    print(f"\n6. ChromaDB setup complete!")
    print(f"   Collection: eight_k_chunks")
    print(f"   Documents: {collection.count()}")
    
    # 7. Test retrieval
    print("\n7. Testing retrieval...")
    test_query = "quarterly earnings revenue profit guidance"
    
    results = collection.query(
        query_texts=[test_query],
        n_results=3
    )
    
    print(f"\n   Query: '{test_query}'")
    print(f"   Found {len(results['ids'][0])} results:")
    
    for i, (doc_id, distance, metadata) in enumerate(zip(
        results['ids'][0],
        results['distances'][0],
        results['metadatas'][0]
    )):
        print(f"\n   Result {i+1}:")
        print(f"     Ticker: {metadata['ticker']}")
        print(f"     Date: {metadata['filing_date']}")
        print(f"     Distance: {distance:.4f}")
        print(f"     Preview: {results['documents'][0][i][:100]}...")
    
    return collection


def test_chromadb_queries():
    """Test various queries on ChromaDB"""
    
    print("\n=== Testing ChromaDB Queries ===")
    
    # Connect to existing collection
    client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-phase1_embedding-ada-002"
    )
    
    collection = client.get_collection(
        name="eight_k_chunks",
        embedding_function=openai_ef
    )
    
    # Test different query types
    queries = [
        "CEO resignation departure executive change",
        "merger acquisition partnership deal",
        "guidance outlook forecast future expectations"
    ]
    
    for query in queries:
        print(f"\nQuery: '{query[:50]}...'")
        
        results = collection.query(
            query_texts=[query],
            n_results=2
        )
        
        if results['ids'][0]:
            for metadata in results['metadatas'][0]:
                print(f"  - {metadata['ticker']} ({metadata['filing_date']})")
        else:
            print("  No results found")
    
    print(f"\nTotal documents in collection: {collection.count()}")


if __name__ == "__main__":
    # Set up ChromaDB
    collection = setup_chromadb_with_openai()
    
    # Test queries
    test_chromadb_queries()