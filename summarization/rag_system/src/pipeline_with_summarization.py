"""
Enhanced RAG Pipeline with Hybrid Search and LLM Summarization
1. Hybrid retrieval (semantic + keyword)
2. LLM summarization of retrieved chunks
3. Save summaries for flexible downstream prediction
"""
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Set
from datetime import datetime
import pandas as pd
import numpy as np
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    OPENAI_API_KEY, SOURCE_DATA_DIR, CHROMADB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, RESULTS_DIR
)
from config.improved_prompts import RETRIEVAL_QUERIES

client = OpenAI(api_key=OPENAI_API_KEY)


class EnhancedRAGPipeline:
    """Pipeline with hybrid search and summarization"""
    
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = RESULTS_DIR / f"enhanced_pipeline_{self.run_id}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Enhanced Pipeline Run ID: {self.run_id}")
        print(f"Output directory: {self.output_dir}\n")
        
        self.filings = []
        self.chunks = []
        self.collection = None
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.chunk_docs = []
        
    def load_and_chunk_filings(self, n=3):
        """Load and chunk sample filings"""
        print("="*60)
        print("STEP 1: LOAD & CHUNK FILINGS")
        print("="*60)
        
        # Load filings
        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"
        
        with open(texts_file, 'r') as f:
            all_texts = json.load(f)
        
        labels_df = pd.read_csv(labels_file)
        
        sample_filings = []
        all_chunks = []
        
        for i, (accession, data) in enumerate(all_texts.items()):
            if i >= n:
                break
                
            label_row = labels_df[labels_df['accession'] == accession]
            if not label_row.empty:
                row = label_row.iloc[0]
                data['accession'] = accession
                data['signal'] = row['signal']
                data['adjusted_return_pct'] = row['adjusted_return_pct']
                sample_filings.append(data)
                
                # Chunk the filing
                text = data['text']
                filing_chunks = []
                
                for j in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                    chunk_text = text[j:j + CHUNK_SIZE]
                    if len(chunk_text) < 100:
                        continue
                    
                    chunk = {
                        'chunk_id': f"{accession}_{len(filing_chunks)}",
                        'accession': accession,
                        'ticker': data['ticker'],
                        'filing_date': data['filing_date'],
                        'chunk_text': chunk_text,
                        'chunk_position': len(filing_chunks),
                        'signal': data['signal'],
                        'adjusted_return_pct': data['adjusted_return_pct']
                    }
                    filing_chunks.append(chunk)
                
                all_chunks.extend(filing_chunks)
                print(f"{i+1}. {data['ticker']} - {data['filing_date']}: {len(filing_chunks)} chunks")
        
        self.filings = sample_filings
        self.chunks = all_chunks
        
        print(f"\nTotal: {len(all_chunks)} chunks from {len(sample_filings)} filings")
        
        # Save
        with open(self.output_dir / "chunks.json", 'w') as f:
            json.dump(all_chunks, f, indent=2)
        
        return all_chunks
    
    def setup_chromadb_and_tfidf(self):
        """Set up both ChromaDB (semantic) and TF-IDF (keyword) search"""
        print("\n" + "="*60)
        print("STEP 2: SETUP HYBRID SEARCH (CHROMADB + TF-IDF)")
        print("="*60)
        
        # A. ChromaDB for semantic search
        print("\nA. Setting up ChromaDB for semantic search...")
        chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-phase1_embedding-ada-002"
        )
        
        collection_name = f"hybrid_{self.run_id}"
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        
        self.collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        
        # Add chunks to ChromaDB
        batch_size = 20
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
            
            self.collection.add(ids=ids, documents=documents, metadatas=metadatas)
        
        print(f"Added {self.collection.count()} chunks to ChromaDB")
        
        # B. TF-IDF for keyword search
        print("\nB. Setting up TF-IDF for keyword search...")
        self.chunk_docs = [chunk['chunk_text'] for chunk in self.chunks]
        
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),  # Include bigrams
            min_df=1,
            max_df=0.95
        )
        
        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.chunk_docs)
        print(f"TF-IDF matrix shape: {self.tfidf_matrix.shape}")
        
        return self.collection
    
    def hybrid_search(self, accession: str, query: str, alpha: float = 0.7) -> List[Dict]:
        """
        Hybrid search combining semantic and keyword matching
        alpha: weight for semantic search (1-alpha for keyword search)
        """
        filing_chunks = [c for c in self.chunks if c['accession'] == accession]
        filing_indices = [i for i, c in enumerate(self.chunks) if c['accession'] == accession]
        
        if not filing_chunks:
            return []
        
        # A. Semantic search using ChromaDB
        semantic_results = self.collection.query(
            query_texts=[query],
            where={"accession": accession},
            n_results=min(10, len(filing_chunks))
        )
        
        # Build semantic scores dict (convert distance to similarity)
        semantic_scores = {}
        if semantic_results['ids'][0]:
            for chunk_id, distance in zip(semantic_results['ids'][0], semantic_results['distances'][0]):
                # Convert distance to similarity: similarity = 1 - distance
                similarity = 1.0 - distance
                semantic_scores[chunk_id] = similarity
        
        # B. Keyword search using TF-IDF
        query_vector = self.tfidf_vectorizer.transform([query])
        
        # Get TF-IDF scores for this filing's chunks
        filing_tfidf = self.tfidf_matrix[filing_indices]
        keyword_similarities = sklearn_cosine_similarity(query_vector, filing_tfidf).flatten()
        
        keyword_scores = {}
        for i, idx in enumerate(filing_indices):
            chunk_id = self.chunks[idx]['chunk_id']
            keyword_scores[chunk_id] = float(keyword_similarities[i])
        
        # C. Combine scores
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
        
        # Sort by combined score
        sorted_chunks = sorted(combined_scores.values(), 
                             key=lambda x: x['combined_score'], 
                             reverse=True)
        
        return sorted_chunks
    
    def retrieve_with_hybrid_search(self, accession: str, top_k: int = 5) -> List[Dict]:
        """Retrieve chunks using hybrid search with multiple queries"""
        print(f"\nHybrid retrieval for {accession}:")
        
        all_retrieved = {}
        
        # Run each query with hybrid search
        for query_type, query_text in list(RETRIEVAL_QUERIES.items())[:4]:  # Use first 4 queries for speed
            results = self.hybrid_search(accession, query_text, alpha=0.7)
            
            for i, result in enumerate(results[:3]):  # Top 3 per query
                chunk_id = result['chunk_id']
                if chunk_id not in all_retrieved or result['combined_score'] > all_retrieved[chunk_id]['score']:
                    # Find the actual chunk
                    chunk = next(c for c in self.chunks if c['chunk_id'] == chunk_id)
                    all_retrieved[chunk_id] = {
                        'chunk_id': chunk_id,
                        'text': chunk['chunk_text'],
                        'score': result['combined_score'],
                        'semantic_score': result['semantic_score'],
                        'keyword_score': result['keyword_score'],
                        'query_type': query_type
                    }
        
        # Sort by combined score and take top-k
        sorted_chunks = sorted(all_retrieved.values(), 
                             key=lambda x: x['score'], 
                             reverse=True)[:top_k]
        
        print(f"  Retrieved {len(sorted_chunks)} chunks")
        for chunk in sorted_chunks[:3]:
            print(f"  - {chunk['query_type']}: score={chunk['score']:.3f} "
                  f"(semantic={chunk['semantic_score']:.3f}, keyword={chunk['keyword_score']:.3f})")
        
        return sorted_chunks
    
    def generate_summary(self, retrieved_chunks: List[Dict], filing_info: Dict) -> Dict:
        """Generate a focused summary of retrieved chunks"""
        
        # Format chunks for prompt
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

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst creating concise summaries."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=400
            )
            
            summary = response.choices[0].message.content
            tokens_used = response.usage.total_tokens
            
        except Exception as e:
            print(f"Error generating summary: {e}")
            summary = "Failed to generate summary"
            tokens_used = 0
        
        return {
            'summary': summary,
            'tokens_used': tokens_used,
            'num_chunks': len(retrieved_chunks),
            'avg_chunk_score': np.mean([c['score'] for c in retrieved_chunks])
        }
    
    def process_all_filings(self):
        """Process all filings: retrieve → summarize → save"""
        print("\n" + "="*60)
        print("STEP 3: RETRIEVE & SUMMARIZE")
        print("="*60)
        
        all_results = []
        
        for filing in self.filings:
            print(f"\n{'='*40}")
            print(f"Processing: {filing['ticker']} - {filing['filing_date']}")
            print(f"{'='*40}")
            
            # A. Hybrid retrieval
            retrieved_chunks = self.retrieve_with_hybrid_search(filing['accession'], top_k=5)
            
            # B. Generate summary
            print("\nGenerating summary...")
            summary_result = self.generate_summary(retrieved_chunks, filing)
            
            print(f"Summary length: {len(summary_result['summary'])} chars")
            print(f"Tokens used: {summary_result['tokens_used']}")
            print(f"\nSummary preview:")
            print(summary_result['summary'][:300] + "...")
            
            # C. Compile result
            result = {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'filing_date': filing['filing_date'],
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct'],
                'retrieved_chunks': [
                    {
                        'chunk_id': c['chunk_id'],
                        'query_type': c['query_type'],
                        'combined_score': c['score'],
                        'semantic_score': c['semantic_score'],
                        'keyword_score': c['keyword_score'],
                        'text_preview': c['text'][:200]
                    }
                    for c in retrieved_chunks
                ],
                'summary': summary_result['summary'],
                'summary_tokens': summary_result['tokens_used'],
                'avg_retrieval_score': summary_result['avg_chunk_score']
            }
            
            all_results.append(result)
        
        # Save results
        output_file = self.output_dir / "summaries_and_retrievals.json"
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n{'='*60}")
        print(f"Saved all summaries to: {output_file}")
        
        # Also save just summaries for easy access
        summaries_only = [
            {
                'accession': r['accession'],
                'ticker': r['ticker'],
                'date': r['filing_date'],
                'summary': r['summary'],
                'true_label': r['true_label'],
                'true_return': r['true_return']
            }
            for r in all_results
        ]
        
        summaries_file = self.output_dir / "summaries_only.json"
        with open(summaries_file, 'w') as f:
            json.dump(summaries_only, f, indent=2)
        
        print(f"Saved summaries only to: {summaries_file}")
        
        return all_results
    
    def evaluate_summaries(self, results: List[Dict]):
        """Simple evaluation of summary quality"""
        print("\n" + "="*60)
        print("STEP 4: SUMMARY EVALUATION")
        print("="*60)
        
        print(f"\n{'Ticker':<8} {'Date':<12} {'Label':<8} {'Return':>8} {'Tokens':<8} {'Avg Score'}")
        print("-" * 60)
        
        for r in results:
            print(f"{r['ticker']:<8} {r['filing_date']:<12} {r['true_label']:<8} "
                  f"{r['true_return']:>7.2f}% {r['summary_tokens']:<8} {r['avg_retrieval_score']:.3f}")
        
        # Display summaries
        print("\n" + "="*60)
        print("GENERATED SUMMARIES")
        print("="*60)
        
        for r in results:
            print(f"\n{r['ticker']} ({r['filing_date']}) - True: {r['true_label']}")
            print("-" * 40)
            print(r['summary'])
            print()
    
    def run_complete_pipeline(self):
        """Run the complete pipeline"""
        print("\n" + "="*60)
        print("ENHANCED RAG PIPELINE WITH HYBRID SEARCH & SUMMARIZATION")
        print("="*60)
        
        # Step 1: Load and chunk
        self.load_and_chunk_filings(n=3)
        
        # Step 2: Setup hybrid search
        self.setup_chromadb_and_tfidf()
        
        # Step 3: Retrieve and summarize
        results = self.process_all_filings()
        
        # Step 4: Evaluate
        self.evaluate_summaries(results)
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}")
        print("="*60)
        
        print("\nSummaries are now ready for:")
        print("  - LLM-based prediction")
        print("  - Embedding-based classification")
        print("  - Traditional ML models")
        print("  - Fine-tuning datasets")
        
        return results


def main():
    pipeline = EnhancedRAGPipeline()
    results = pipeline.run_complete_pipeline()
    return results


if __name__ == "__main__":
    main()