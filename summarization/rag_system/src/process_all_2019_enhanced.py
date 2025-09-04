"""
Process all 2019 8-K filings using the enhanced RAG pipeline
With parallel processing, checkpointing, and rate limiting
"""
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import sys
import time
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from collections import deque

sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    OPENAI_API_KEY, SOURCE_DATA_DIR, CHROMADB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, RESULTS_DIR
)
from config.improved_prompts import RETRIEVAL_QUERIES

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# Rate limiting setup
class RateLimiter:
    def __init__(self, max_calls=50, time_window=60):
        self.max_calls = max_calls
        self.time_window = time_window
        self.calls = deque()
        self.lock = threading.Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            # Remove old calls outside the time window
            while self.calls and self.calls[0] < now - self.time_window:
                self.calls.popleft()
            
            # If at max capacity, wait
            if len(self.calls) >= self.max_calls:
                sleep_time = self.time_window - (now - self.calls[0]) + 0.1
                if sleep_time > 0:
                    print(f"Rate limit reached. Sleeping {sleep_time:.1f}s...")
                    time.sleep(sleep_time)
            
            # Record this call
            self.calls.append(now)

# Global rate limiter
rate_limiter = RateLimiter(max_calls=50, time_window=60)


class ParallelRAGProcessor:
    """Process all 2019 filings with parallel processing and checkpointing"""
    
    def __init__(self, num_workers=10):
        self.num_workers = num_workers
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = RESULTS_DIR / f"all_2019_{self.run_id}"
        self.output_dir.mkdir(exist_ok=True)
        
        # Checkpoint files
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.checkpoint_dir.mkdir(exist_ok=True)
        self.progress_file = self.checkpoint_dir / "progress.json"
        self.chunks_file = self.checkpoint_dir / "chunks.pkl"
        self.results_file = self.checkpoint_dir / "results.json"
        self.tfidf_file = self.checkpoint_dir / "tfidf.pkl"
        
        # Data structures
        self.filings = []
        self.chunks = []
        self.collection = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_docs = []
        
        print(f"Parallel RAG Processor initialized")
        print(f"Run ID: {self.run_id}")
        print(f"Workers: {self.num_workers}")
        print(f"Output: {self.output_dir}")
    
    def load_all_2019_data(self) -> List[Dict]:
        """Load all 2019 filings and labels"""
        print("\n" + "="*60)
        print("LOADING ALL 2019 DATA")
        print("="*60)
        
        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"
        
        print(f"Loading from: {texts_file}")
        with open(texts_file, 'r') as f:
            all_texts = json.load(f)
        
        print(f"Loading labels from: {labels_file}")
        labels_df = pd.read_csv(labels_file)
        
        all_filings = []
        for accession, data in all_texts.items():
            label_row = labels_df[labels_df['accession'] == accession]
            if not label_row.empty:
                row = label_row.iloc[0]
                data['accession'] = accession
                data['signal'] = row['signal']
                data['adjusted_return_pct'] = row['adjusted_return_pct']
                all_filings.append(data)
        
        self.filings = all_filings
        print(f"Loaded {len(all_filings)} filings with labels")
        
        # Save filing metadata
        filing_info = pd.DataFrame([
            {
                'accession': f['accession'],
                'ticker': f['ticker'],
                'date': f['filing_date'],
                'signal': f['signal'],
                'return': f['adjusted_return_pct']
            }
            for f in all_filings
        ])
        filing_info.to_csv(self.output_dir / "filing_metadata.csv", index=False)
        
        return all_filings
    
    def chunk_single_filing(self, filing: Dict) -> List[Dict]:
        """Chunk a single filing"""
        text = filing['text']
        filing_chunks = []
        
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            if len(chunk_text) < 100:
                continue
            
            chunk = {
                'chunk_id': f"{filing['accession']}_{len(filing_chunks)}",
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'filing_date': filing['filing_date'],
                'chunk_text': chunk_text,
                'chunk_position': len(filing_chunks),
                'signal': filing['signal'],
                'adjusted_return_pct': filing['adjusted_return_pct']
            }
            filing_chunks.append(chunk)
        
        return filing_chunks
    
    def chunk_all_filings_parallel(self) -> List[Dict]:
        """Chunk all filings using parallel processing"""
        print("\n" + "="*60)
        print("CHUNKING ALL FILINGS")
        print("="*60)
        
        # Check if chunks already exist
        if self.chunks_file.exists():
            print("Loading existing chunks from checkpoint...")
            with open(self.chunks_file, 'rb') as f:
                self.chunks = pickle.load(f)
            print(f"Loaded {len(self.chunks)} chunks from checkpoint")
            return self.chunks
        
        print(f"Chunking {len(self.filings)} filings in parallel...")
        all_chunks = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit all chunking tasks
            future_to_filing = {
                executor.submit(self.chunk_single_filing, filing): filing 
                for filing in self.filings
            }
            
            # Process as completed
            for future in tqdm(as_completed(future_to_filing), 
                             total=len(self.filings), 
                             desc="Chunking filings"):
                filing = future_to_filing[future]
                try:
                    filing_chunks = future.result()
                    all_chunks.extend(filing_chunks)
                except Exception as e:
                    print(f"\nError chunking {filing['ticker']}: {e}")
        
        self.chunks = all_chunks
        print(f"Created {len(all_chunks)} total chunks")
        
        # Save chunks checkpoint
        with open(self.chunks_file, 'wb') as f:
            pickle.dump(all_chunks, f)
        print(f"Saved chunks to checkpoint")
        
        return all_chunks
    
    def setup_chromadb_and_tfidf(self):
        """Set up ChromaDB and TF-IDF for hybrid search"""
        print("\n" + "="*60)
        print("SETTING UP CHROMADB AND TF-IDF")
        print("="*60)
        
        # Check if TF-IDF already exists
        if self.tfidf_file.exists():
            print("Loading existing TF-IDF from checkpoint...")
            with open(self.tfidf_file, 'rb') as f:
                saved_data = pickle.load(f)
                self.tfidf_vectorizer = saved_data['vectorizer']
                self.tfidf_matrix = saved_data['matrix']
                self.chunk_docs = saved_data['docs']
            print(f"Loaded TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        else:
            # Setup TF-IDF
            print("Building TF-IDF matrix...")
            self.chunk_docs = [chunk['chunk_text'] for chunk in self.chunks]
            
            self.tfidf_vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words='english',
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.95
            )
            
            self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_docs)
            print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
            
            # Save TF-IDF checkpoint
            with open(self.tfidf_file, 'wb') as f:
                pickle.dump({
                    'vectorizer': self.tfidf_vectorizer,
                    'matrix': self.tfidf_matrix,
                    'docs': self.chunk_docs
                }, f)
            print("Saved TF-IDF to checkpoint")
        
        # Setup ChromaDB
        print("\nSetting up ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-phase1_embedding-ada-002"
        )
        
        collection_name = f"rag_2019_{self.run_id}"
        
        # Delete if exists
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add chunks to ChromaDB in batches
        print("Adding chunks to ChromaDB...")
        batch_size = 100
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Adding to ChromaDB"):
            batch = self.chunks[i:i + batch_size]
            
            ids = [chunk['chunk_id'] for chunk in batch]
            documents = [chunk['chunk_text'] for chunk in batch]
            metadatas = [
                {
                    'accession': chunk['accession'],
                    'ticker': chunk['ticker'],
                    'filing_date': chunk['filing_date'],
                    'chunk_position': chunk['chunk_position']
                }
                for chunk in batch
            ]
            
            # Add with rate limiting
            rate_limiter.wait_if_needed()
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        
        print(f"Added {self.collection.count()} chunks to ChromaDB")
    
    def hybrid_search(self, accession: str, query: str, alpha: float = 0.7) -> List[Dict]:
        """Hybrid search combining semantic and keyword matching"""
        filing_chunks = [c for c in self.chunks if c['accession'] == accession]
        filing_indices = [i for i, c in enumerate(self.chunks) if c['accession'] == accession]
        
        if not filing_chunks:
            return []
        
        # Semantic search using ChromaDB
        semantic_results = self.collection.query(
            query_texts=[query],
            where={"accession": accession},
            n_results=min(10, len(filing_chunks))
        )
        
        semantic_scores = {}
        if semantic_results['ids'][0]:
            for chunk_id, distance in zip(semantic_results['ids'][0], semantic_results['distances'][0]):
                similarity = 1.0 - distance
                semantic_scores[chunk_id] = similarity
        
        # Keyword search using TF-IDF
        query_vector = self.tfidf_vectorizer.transform([query])
        filing_tfidf = self.tfidf_matrix[filing_indices]
        keyword_similarities = sklearn_cosine_similarity(query_vector, filing_tfidf).flatten()
        
        keyword_scores = {}
        for i, idx in enumerate(filing_indices):
            chunk_id = self.chunks[idx]['chunk_id']
            keyword_scores[chunk_id] = float(keyword_similarities[i])
        
        # Combine scores
        all_chunk_ids = set(semantic_scores.keys()) | set(keyword_scores.keys())
        combined_scores = {}
        
        for chunk_id in all_chunk_ids:
            semantic = semantic_scores.get(chunk_id, 0.0)
            keyword = keyword_scores.get(chunk_id, 0.0)
            combined = alpha * semantic + (1 - alpha) * keyword
            combined_scores[chunk_id] = {
                'chunk_id': chunk_id,
                'combined_score': combined,
                'semantic_score': semantic,
                'keyword_score': keyword
            }
        
        sorted_chunks = sorted(combined_scores.values(), 
                             key=lambda x: x['combined_score'], 
                             reverse=True)
        
        return sorted_chunks
    
    def retrieve_with_hybrid_search(self, accession: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using hybrid search with multiple queries"""
        all_retrieved = {}
        
        # Use first 4 queries for speed
        for query_type, query_text in list(RETRIEVAL_QUERIES.items())[:4]:
            results = self.hybrid_search(accession, query_text, alpha=0.7)
            
            for i, result in enumerate(results[:3]):  # Top 3 per query
                chunk_id = result['chunk_id']
                if chunk_id not in all_retrieved or result['combined_score'] > all_retrieved[chunk_id]['score']:
                    chunk = next(c for c in self.chunks if c['chunk_id'] == chunk_id)
                    all_retrieved[chunk_id] = {
                        'chunk_id': chunk_id,
                        'text': chunk['chunk_text'],
                        'score': result['combined_score'],
                        'semantic_score': result['semantic_score'],
                        'keyword_score': result['keyword_score'],
                        'query_type': query_type
                    }
        
        sorted_chunks = sorted(all_retrieved.values(), 
                             key=lambda x: x['score'], 
                             reverse=True)[:top_k]
        
        return sorted_chunks
    
    def generate_summary(self, retrieved_chunks: List[Dict], filing_info: Dict) -> Dict:
        """Generate summary with rate limiting and retry"""
        chunks_text = "\n\n".join([
            f"[Excerpt {i+1}]\n{chunk['text'][:600]}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""You are a financial analyst. Summarize the key information from these 8-K filing excerpts.

Company: {filing_info['ticker']}
Filing Date: {filing_info['filing_date']}

Excerpts:
{chunks_text}

Create a concise summary (200-300 words) that captures:
1. The most important events or announcements
2. Any financial metrics or changes
3. Material impacts on the business
4. Forward-looking statements or guidance

Focus on information that would impact stock price. Be specific about numbers and facts.

Summary:"""
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                rate_limiter.wait_if_needed()
                
                response = client.chat.completions.create(
                    model="gpt-3.5-turbo",
                    messages=[
                        {"role": "system", "content": "You are a financial analyst creating concise summaries."},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3,
                    max_tokens=400,
                    timeout=30
                )
                
                return {
                    'summary': response.choices[0].message.content,
                    'tokens_used': response.usage.total_tokens
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    print(f"API error: {str(e)[:50]}... Retrying in {wait_time}s")
                    time.sleep(wait_time)
                else:
                    return {
                        'summary': f"Failed to generate summary: {str(e)[:100]}",
                        'tokens_used': 0
                    }
    
    def process_single_filing(self, filing: Dict) -> Dict:
        """Process a single filing: retrieve and summarize"""
        try:
            # Retrieve chunks
            retrieved_chunks = self.retrieve_with_hybrid_search(filing['accession'], top_k=5)
            
            if not retrieved_chunks:
                return {
                    'accession': filing['accession'],
                    'ticker': filing['ticker'],
                    'filing_date': filing['filing_date'],
                    'true_label': filing['signal'],
                    'true_return': filing['adjusted_return_pct'],
                    'summary': 'No relevant chunks found',
                    'status': 'error'
                }
            
            # Generate summary
            summary_result = self.generate_summary(retrieved_chunks, filing)
            
            return {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'filing_date': filing['filing_date'],
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct'],
                'summary': summary_result['summary'],
                'tokens_used': summary_result['tokens_used'],
                'num_chunks_retrieved': len(retrieved_chunks),
                'avg_retrieval_score': np.mean([c['score'] for c in retrieved_chunks]),
                'status': 'success'
            }
            
        except Exception as e:
            return {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'filing_date': filing['filing_date'],
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct'],
                'summary': f'Error: {str(e)[:100]}',
                'status': 'error'
            }
    
    def load_checkpoint(self) -> Tuple[List[str], List[Dict]]:
        """Load checkpoint if exists"""
        if self.progress_file.exists():
            with open(self.progress_file, 'r') as f:
                checkpoint = json.load(f)
            
            if self.results_file.exists():
                with open(self.results_file, 'r') as f:
                    results = json.load(f)
            else:
                results = []
            
            print(f"Loaded checkpoint: {len(checkpoint['completed'])} completed")
            return checkpoint['completed'], results
        
        return [], []
    
    def save_checkpoint(self, completed_accessions: List[str], results: List[Dict]):
        """Save checkpoint"""
        checkpoint = {
            'timestamp': datetime.now().isoformat(),
            'completed': completed_accessions,
            'total_processed': len(completed_accessions)
        }
        
        with open(self.progress_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
        
        with open(self.results_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def process_batch_parallel(self, batch_filings: List[Dict]) -> List[Dict]:
        """Process a batch of filings in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            future_to_filing = {
                executor.submit(self.process_single_filing, filing): filing 
                for filing in batch_filings
            }
            
            for future in tqdm(as_completed(future_to_filing), 
                             total=len(batch_filings), 
                             desc="Processing batch"):
                filing = future_to_filing[future]
                try:
                    result = future.result(timeout=60)
                    results.append(result)
                except Exception as e:
                    print(f"\nError processing {filing['ticker']}: {e}")
                    results.append({
                        'accession': filing['accession'],
                        'ticker': filing['ticker'],
                        'filing_date': filing['filing_date'],
                        'true_label': filing['signal'],
                        'true_return': filing['adjusted_return_pct'],
                        'summary': f'Processing error: {str(e)[:100]}',
                        'status': 'error'
                    })
        
        return results
    
    def run_full_pipeline(self):
        """Run the complete pipeline with checkpointing"""
        print("\n" + "="*60)
        print("ENHANCED RAG PIPELINE FOR ALL 2019 DATA")
        print("="*60)
        
        # Step 1: Load all data
        self.load_all_2019_data()
        
        # Step 2: Chunk all filings
        self.chunk_all_filings_parallel()
        
        # Step 3: Setup ChromaDB and TF-IDF
        self.setup_chromadb_and_tfidf()
        
        # Step 4: Process filings with checkpointing
        print("\n" + "="*60)
        print("PROCESSING ALL FILINGS")
        print("="*60)
        
        # Load checkpoint
        completed_accessions, existing_results = self.load_checkpoint()
        completed_set = set(completed_accessions)
        
        # Filter remaining filings
        remaining_filings = [
            f for f in self.filings 
            if f['accession'] not in completed_set
        ]
        
        print(f"Progress:")
        print(f"  Total: {len(self.filings)}")
        print(f"  Completed: {len(completed_accessions)}")
        print(f"  Remaining: {len(remaining_filings)}")
        
        if not remaining_filings:
            print("All filings already processed!")
            return existing_results
        
        # Process in batches
        batch_size = 50
        all_results = existing_results.copy()
        
        for i in range(0, len(remaining_filings), batch_size):
            batch = remaining_filings[i:i+batch_size]
            batch_num = (len(completed_accessions) + i) // batch_size + 1
            total_batches = (len(self.filings) + batch_size - 1) // batch_size
            
            print(f"\n{'='*40}")
            print(f"Batch {batch_num}/{total_batches} ({len(batch)} filings)")
            print(f"{'='*40}")
            
            # Process batch
            batch_results = self.process_batch_parallel(batch)
            
            # Update results
            all_results.extend(batch_results)
            completed_accessions.extend([r['accession'] for r in batch_results])
            
            # Save checkpoint
            self.save_checkpoint(completed_accessions, all_results)
            
            # Stats
            successful = sum(1 for r in batch_results if r['status'] == 'success')
            failed = sum(1 for r in batch_results if r['status'] == 'error')
            
            print(f"Batch complete: {successful} success, {failed} failed")
            
            # Small delay between batches
            time.sleep(2)
        
        # Save final results
        self.save_final_results(all_results)
        
        return all_results
    
    def save_final_results(self, results: List[Dict]):
        """Save final results in multiple formats"""
        print("\n" + "="*60)
        print("SAVING FINAL RESULTS")
        print("="*60)
        
        # Full results
        with open(self.output_dir / "all_summaries.json", 'w') as f:
            json.dump(results, f, indent=2)
        
        # Simplified version
        simplified = [
            {
                'accession': r['accession'],
                'ticker': r['ticker'],
                'date': r['filing_date'],
                'summary': r.get('summary', 'No summary'),
                'true_label': r['true_label'],
                'true_return': r['true_return']
            }
            for r in results if r['status'] == 'success'
        ]
        
        with open(self.output_dir / "summaries_simplified.json", 'w') as f:
            json.dump(simplified, f, indent=2)
        
        # CSV format
        df = pd.DataFrame(simplified)
        df.to_csv(self.output_dir / "summaries.csv", index=False)
        
        # Statistics
        successful = sum(1 for r in results if r['status'] == 'success')
        failed = sum(1 for r in results if r['status'] == 'error')
        
        labels = [r['true_label'] for r in results if r['status'] == 'success']
        returns = [r['true_return'] for r in results if r['status'] == 'success']
        
        stats = {
            'total_processed': len(results),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(results) * 100,
            'label_distribution': {
                'UP': labels.count('UP'),
                'DOWN': labels.count('DOWN'),
                'STAY': labels.count('STAY')
            },
            'return_stats': {
                'min': min(returns) if returns else 0,
                'max': max(returns) if returns else 0,
                'avg': sum(returns) / len(returns) if returns else 0,
                'std': np.std(returns) if returns else 0
            },
            'total_chunks': len(self.chunks),
            'avg_chunks_per_filing': len(self.chunks) / len(self.filings)
        }
        
        with open(self.output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"\nResults saved to: {self.output_dir}")
        print(f"\nFinal Statistics:")
        print(f"  Total processed: {stats['total_processed']}")
        print(f"  Successful: {stats['successful']} ({stats['success_rate']:.1f}%)")
        print(f"  Failed: {stats['failed']}")
        print(f"\nLabel Distribution:")
        for label, count in stats['label_distribution'].items():
            print(f"  {label}: {count}")
        print(f"\nReturn Statistics:")
        print(f"  Min: {stats['return_stats']['min']:.2f}%")
        print(f"  Max: {stats['return_stats']['max']:.2f}%")
        print(f"  Avg: {stats['return_stats']['avg']:.2f}%")


def main():
    """Run the enhanced parallel pipeline"""
    processor = ParallelRAGProcessor(num_workers=10)
    results = processor.run_full_pipeline()
    
    print("\n" + "="*60)
    print("PIPELINE COMPLETE!")
    print(f"Processed {len(results)} filings")
    print("="*60)
    
    return results


if __name__ == "__main__":
    main()