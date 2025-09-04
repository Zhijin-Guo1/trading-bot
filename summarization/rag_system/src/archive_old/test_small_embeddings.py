"""
Create embeddings for just a small sample to test
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CHUNKS_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL


def create_sample_embeddings(n_chunks=100):
    """Create embeddings for first n chunks only"""
    
    # Load chunks
    chunks_file = CHUNKS_DIR / "chunks_2019.json"
    with open(chunks_file, 'r') as f:
        all_chunks = json.load(f)
    
    # Take sample
    chunks = all_chunks[:n_chunks]
    print(f"Creating embeddings for {len(chunks)} sample chunks")
    
    # Initialize model
    print(f"Loading model: {EMBEDDING_MODEL}")
    if EMBEDDING_MODEL == "ProsusAI/finbert":
        model = SentenceTransformer('ProsusAI/finbert')
    else:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Create embeddings
    texts = [chunk['chunk_text'] for chunk in chunks]
    print("Encoding...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    
    # Save
    output_file = EMBEDDINGS_DIR / "sample_embeddings.npy"
    np.save(output_file, embeddings)
    print(f"Saved {embeddings.shape} to: {output_file}")
    
    # Save chunk info
    info_file = EMBEDDINGS_DIR / "sample_chunks.json"
    with open(info_file, 'w') as f:
        json.dump(chunks, f, indent=2)
    
    return embeddings


if __name__ == "__main__":
    embeddings = create_sample_embeddings(100)
    print(f"Shape: {embeddings.shape}")
    print(f"First 3 chunks embedded successfully")