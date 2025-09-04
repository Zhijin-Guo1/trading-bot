"""
Create embeddings for all chunks (required for proper retrieval)
"""
import json
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import CHUNKS_DIR, EMBEDDINGS_DIR, EMBEDDING_MODEL


def create_embeddings():
    """Create embeddings for all 2019 chunks"""
    
    # Load chunks
    chunks_file = CHUNKS_DIR / "chunks_2019.json"
    print(f"Loading chunks from: {chunks_file}")
    
    with open(chunks_file, 'r') as f:
        chunks = json.load(f)
    
    print(f"Loaded {len(chunks)} chunks")
    
    # Initialize phase1_embedding model
    print(f"Loading phase1_embedding model: {EMBEDDING_MODEL}")
    if EMBEDDING_MODEL == "ProsusAI/finbert":
        model = SentenceTransformer('ProsusAI/finbert')
    else:
        model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    
    # Extract texts
    texts = [chunk['chunk_text'] for chunk in chunks]
    
    # Create embeddings in batches
    print("Creating embeddings...")
    batch_size = 32
    embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding chunks"):
        batch = texts[i:i + batch_size]
        batch_embeddings = model.encode(
            batch,
            convert_to_numpy=True,
            show_progress_bar=False
        )
        embeddings.append(batch_embeddings)
    
    # Combine all embeddings
    all_embeddings = np.vstack(embeddings)
    
    # Save embeddings
    output_file = EMBEDDINGS_DIR / "chunk_embeddings_2019.npy"
    np.save(output_file, all_embeddings)
    print(f"Saved {all_embeddings.shape} embeddings to: {output_file}")
    
    # Save chunk index for reference
    index_file = EMBEDDINGS_DIR / "chunk_index_2019.json"
    chunk_index = {
        i: {
            'chunk_id': chunk['chunk_id'],
            'accession': chunk['accession'],
            'ticker': chunk['ticker'],
            'item_number': chunk['item_number']
        }
        for i, chunk in enumerate(chunks)
    }
    
    with open(index_file, 'w') as f:
        json.dump(chunk_index, f, indent=2)
    
    print(f"Saved chunk index to: {index_file}")
    
    return all_embeddings, chunks


if __name__ == "__main__":
    embeddings, chunks = create_embeddings()
    print(f"\nEmbedding statistics:")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Mean: {embeddings.mean():.4f}")
    print(f"  Std: {embeddings.std():.4f}")