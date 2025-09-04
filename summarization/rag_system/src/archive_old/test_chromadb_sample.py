"""
Test ChromaDB setup with a small sample of chunks
"""
import json
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from setup_chromadb import ChromaDBSetup
from config.settings import CHUNKS_DIR


def test_with_sample(n_samples=100):
    """Test ChromaDB with a small sample"""
    
    # Load chunks
    chunks_file = CHUNKS_DIR / "chunks_2019.json"
    print(f"Loading chunks from: {chunks_file}")
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    # Take sample
    sample_chunks = chunks[:n_samples]
    print(f"Testing with {len(sample_chunks)} chunks")
    
    # Initialize ChromaDB
    db = ChromaDBSetup(collection_name="eight_k_test")
    
    # Add chunks
    db.add_chunks_to_collection(sample_chunks, batch_size=20)
    
    # Test retrieval
    print("\nTesting retrieval...")
    db.test_retrieval("earnings revenue profit", n_results=3)
    
    print("\nTest complete!")


if __name__ == "__main__":
    test_with_sample(100)