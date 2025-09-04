"""
Complete RAG Pipeline with Improved Prompts
Full flow: Load → Chunk → Embed → Store in ChromaDB → Retrieve → Generate → Evaluate
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple
from datetime import datetime
import pandas as pd
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    OPENAI_API_KEY, SOURCE_DATA_DIR, CHROMADB_DIR,
    CHUNK_SIZE, CHUNK_OVERLAP, TOP_K_CHUNKS, RESULTS_DIR
)
from config.improved_prompts import (
    RETRIEVAL_QUERIES, GENERATION_PROMPT_TEMPLATE
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class ImprovedRAGPipeline:
    """Complete RAG pipeline with improved prompts and full persistence"""
    
    def __init__(self):
        self.run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = RESULTS_DIR / f"improved_pipeline_{self.run_id}"
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Pipeline run ID: {self.run_id}")
        print(f"Output directory: {self.output_dir}\n")
        
        self.filings = []
        self.chunks = []
        self.results = []
        self.collection = None
        
    def step1_load_sample_filings(self, n=3) -> List[Dict]:
        """Step 1: Load sample filings"""
        print("="*60)
        print("STEP 1: LOADING SAMPLE FILINGS")
        print("="*60)
        
        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"
        
        with open(texts_file, 'r') as f:
            all_texts = json.load(f)
        
        labels_df = pd.read_csv(labels_file)
        
        # Get first n filings with labels
        sample_filings = []
        for i, (accession, data) in enumerate(all_texts.items()):
            if i >= n:
                break
                
            label_row = labels_df[labels_df['accession'] == accession]
            if not label_row.empty:
                row = label_row.iloc[0]
                data['accession'] = accession
                data['signal'] = row['signal']
                data['adjusted_return_pct'] = row['adjusted_return_pct']
                data['vix_level'] = row.get('vix_level', 20.0)  # Default VIX
                sample_filings.append(data)
                
                print(f"\n{i+1}. Loaded filing:")
                print(f"   Ticker: {data['ticker']}")
                print(f"   Date: {data['filing_date']}")
                print(f"   Size: {len(data['text'])} characters")
                print(f"   True label: {data['signal']} ({data['adjusted_return_pct']:.2f}%)")
        
        self.filings = sample_filings
        
        # Save filings
        filings_file = self.output_dir / "1_filings.json"
        with open(filings_file, 'w') as f:
            json.dump(sample_filings, f, indent=2)
        print(f"\nSaved filings to: {filings_file}")
        
        return sample_filings
    
    def step2_chunk_filings(self) -> List[Dict]:
        """Step 2: Chunk filings into smaller segments"""
        print("\n" + "="*60)
        print("STEP 2: CHUNKING FILINGS")
        print("="*60)
        
        all_chunks = []
        
        for filing in self.filings:
            # Extract text sections if possible
            text = filing['text']
            
            # Create chunks with overlap
            filing_chunks = []
            for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
                chunk_text = text[i:i + CHUNK_SIZE]
                
                if len(chunk_text) < 100:  # Skip tiny chunks
                    continue
                
                chunk = {
                    'chunk_id': f"{filing['accession']}_{len(filing_chunks)}",
                    'accession': filing['accession'],
                    'ticker': filing['ticker'],
                    'filing_date': filing['filing_date'],
                    'chunk_text': chunk_text,
                    'chunk_position': len(filing_chunks),
                    'chunk_length': len(chunk_text),
                    'signal': filing['signal'],
                    'adjusted_return_pct': filing['adjusted_return_pct']
                }
                filing_chunks.append(chunk)
            
            all_chunks.extend(filing_chunks)
            print(f"\n{filing['ticker']}: Created {len(filing_chunks)} chunks")
        
        self.chunks = all_chunks
        print(f"\nTotal chunks created: {len(all_chunks)}")
        
        # Save chunks
        chunks_file = self.output_dir / "2_chunks.json"
        with open(chunks_file, 'w') as f:
            json.dump(all_chunks, f, indent=2)
        print(f"Saved chunks to: {chunks_file}")
        
        return all_chunks
    
    def step3_setup_chromadb(self):
        """Step 3: Set up ChromaDB and add chunks with embeddings"""
        print("\n" + "="*60)
        print("STEP 3: SETTING UP CHROMADB WITH EMBEDDINGS")
        print("="*60)
        
        # Initialize ChromaDB
        print("\nInitializing ChromaDB...")
        chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
        
        # Create OpenAI phase1_embedding function
        openai_ef = embedding_functions.OpenAIEmbeddingFunction(
            api_key=OPENAI_API_KEY,
            model_name="text-phase1_embedding-ada-002"
        )
        
        # Create unique collection name for this run
        collection_name = f"rag_test_{self.run_id}"
        
        # Delete if exists
        try:
            chroma_client.delete_collection(collection_name)
        except:
            pass
        
        # Create new collection
        self.collection = chroma_client.create_collection(
            name=collection_name,
            embedding_function=openai_ef,
            metadata={"hnsw:space": "cosine"}
        )
        print(f"Created collection: {collection_name}")
        
        # Add chunks to ChromaDB
        print("\nAdding chunks to ChromaDB with embeddings...")
        
        batch_size = 20
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Adding chunks"):
            batch = self.chunks[i:i + batch_size]
            
            ids = [chunk['chunk_id'] for chunk in batch]
            documents = [chunk['chunk_text'] for chunk in batch]
            metadatas = [
                {
                    'accession': chunk['accession'],
                    'ticker': chunk['ticker'],
                    'filing_date': chunk['filing_date'],
                    'chunk_position': chunk['chunk_position'],
                    'signal': chunk['signal'],
                    'return': float(chunk['adjusted_return_pct'])
                }
                for chunk in batch
            ]
            
            self.collection.add(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )
        
        print(f"Added {self.collection.count()} chunks to ChromaDB")
        
        # Save collection info
        collection_info = {
            'collection_name': collection_name,
            'num_chunks': self.collection.count(),
            'embedding_model': 'text-phase1_embedding-ada-002'
        }
        
        info_file = self.output_dir / "3_chromadb_info.json"
        with open(info_file, 'w') as f:
            json.dump(collection_info, f, indent=2)
        
        return self.collection
    
    def step4_retrieve_and_generate(self):
        """Step 4: Retrieve relevant chunks and generate predictions"""
        print("\n" + "="*60)
        print("STEP 4: RETRIEVAL AND GENERATION")
        print("="*60)
        
        all_retrievals = []
        all_generations = []
        
        for filing in self.filings:
            print(f"\n{'='*40}")
            print(f"Processing: {filing['ticker']} - {filing['filing_date']}")
            print(f"{'='*40}")
            
            # A. Retrieve relevant chunks using improved queries
            print("\nA. Retrieving relevant chunks...")
            retrieved_chunks = self.retrieve_with_improved_queries(filing['accession'])
            
            retrieval_info = {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'num_chunks_retrieved': len(retrieved_chunks),
                'chunks': retrieved_chunks
            }
            all_retrievals.append(retrieval_info)
            
            # B. Generate prediction with improved prompt
            print("\nB. Generating prediction...")
            analysis = self.generate_improved_prediction(retrieved_chunks, filing)
            
            generation_info = {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'analysis': analysis,
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct']
            }
            all_generations.append(generation_info)
            
            # C. Evaluate
            correct = analysis['prediction'] == filing['signal']
            
            print(f"\nC. Results:")
            print(f"   Prediction: {analysis['prediction']} (confidence: {analysis['confidence']:.2f})")
            print(f"   True label: {filing['signal']} ({filing['adjusted_return_pct']:.2f}%)")
            print(f"   Correct: {'✅ YES' if correct else '❌ NO'}")
            print(f"   Surprise level: {analysis.get('surprise_level', 'N/A')}")
            print(f"   Primary driver: {analysis.get('primary_driver', 'N/A')}")
            
            # Store result
            result = {
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'prediction': analysis['prediction'],
                'confidence': analysis['confidence'],
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct'],
                'correct': correct,
                'surprise_level': analysis.get('surprise_level', 'N/A'),
                'primary_driver': analysis.get('primary_driver', 'N/A'),
                'reasoning': analysis.get('reasoning', 'N/A')
            }
            self.results.append(result)
        
        # Save retrievals and generations
        retrievals_file = self.output_dir / "4_retrievals.json"
        with open(retrievals_file, 'w') as f:
            json.dump(all_retrievals, f, indent=2)
        
        generations_file = self.output_dir / "5_generations.json"
        with open(generations_file, 'w') as f:
            json.dump(all_generations, f, indent=2)
        
        print(f"\nSaved retrievals to: {retrievals_file}")
        print(f"Saved generations to: {generations_file}")
        
        return all_retrievals, all_generations
    
    def retrieve_with_improved_queries(self, accession: str) -> List[Dict]:
        """Retrieve chunks using improved queries"""
        all_retrieved = {}
        
        # Run each improved query
        for query_type, query_text in RETRIEVAL_QUERIES.items():
            results = self.collection.query(
                query_texts=[query_text],
                where={"accession": accession},
                n_results=3  # Get top 3 per query
            )
            
            if results['ids'][0]:
                for i, (chunk_id, doc, dist, meta) in enumerate(zip(
                    results['ids'][0],
                    results['documents'][0],
                    results['distances'][0],
                    results['metadatas'][0]
                )):
                    if chunk_id not in all_retrieved or all_retrieved[chunk_id]['distance'] > dist:
                        all_retrieved[chunk_id] = {
                            'chunk_id': chunk_id,
                            'text': doc[:500],  # Truncate for storage
                            'distance': dist,
                            'query_type': query_type,
                            'position': meta['chunk_position']
                        }
        
        # Sort by distance and take top chunks
        sorted_chunks = sorted(all_retrieved.values(), key=lambda x: x['distance'])[:TOP_K_CHUNKS]
        
        print(f"   Retrieved {len(sorted_chunks)} chunks")
        for chunk in sorted_chunks[:3]:
            print(f"   - Query '{chunk['query_type']}': distance {chunk['distance']:.3f}")
        
        return sorted_chunks
    
    def generate_improved_prediction(self, retrieved_chunks: List[Dict], filing: Dict) -> Dict:
        """Generate prediction using improved prompt"""
        
        # Format chunks for prompt
        chunks_text = "\n\n".join([
            f"[Excerpt {i+1} - matched '{chunk['query_type']}' query]\n{chunk['text']}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        # Determine VIX context
        vix = filing.get('vix_level', 20.0)
        vix_context = "high" if vix > 25 else "low" if vix < 15 else "moderate"
        
        # Format the improved prompt
        prompt = GENERATION_PROMPT_TEMPLATE.format(
            ticker=filing['ticker'],
            filing_date=filing['filing_date'],
            vix_context=vix_context,
            chunks_text=chunks_text
        )
        
        # Call GPT-3.5
        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert financial analyst specializing in event-driven trading."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=600
            )
            
            response_text = response.choices[0].message.content
            
            # Parse JSON response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            analysis = json.loads(response_text[start:end])
            
        except Exception as e:
            print(f"   Error in generation: {e}")
            analysis = {
                "surprise_level": "UNKNOWN",
                "prediction": "STAY",
                "confidence": 0.5,
                "primary_driver": "Error in analysis",
                "reasoning": "Failed to generate analysis"
            }
        
        return analysis
    
    def step5_evaluate_results(self):
        """Step 5: Evaluate and summarize results"""
        print("\n" + "="*60)
        print("STEP 5: EVALUATION AND SUMMARY")
        print("="*60)
        
        # Calculate metrics
        correct = sum(1 for r in self.results if r['correct'])
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0
        
        # Display results table
        print("\nResults Summary:")
        print(f"{'Ticker':<8} {'Date':<12} {'Predicted':<10} {'True':<10} {'Return':>8} {'Correct':<8} {'Driver'}")
        print("-" * 80)
        
        for r in self.results:
            status = "✅" if r['correct'] else "❌"
            print(f"{r['ticker']:<8} {r['date']:<12} {r['prediction']:<10} {r['true_label']:<10} "
                  f"{r['true_return']:>7.2f}% {status:<8} {r['primary_driver'][:30]}")
        
        print(f"\n{'='*40}")
        print(f"OVERALL ACCURACY: {accuracy:.1%} ({correct}/{total})")
        print(f"{'='*40}")
        
        # Analyze by surprise level
        surprise_analysis = {}
        for r in self.results:
            level = r.get('surprise_level', 'UNKNOWN')
            if level not in surprise_analysis:
                surprise_analysis[level] = {'correct': 0, 'total': 0}
            surprise_analysis[level]['total'] += 1
            if r['correct']:
                surprise_analysis[level]['correct'] += 1
        
        print("\nAccuracy by Surprise Level:")
        for level, stats in surprise_analysis.items():
            acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
            print(f"  {level}: {acc:.0%} ({stats['correct']}/{stats['total']})")
        
        # Save final results
        results_file = self.output_dir / "6_final_results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'summary': {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total,
                    'surprise_analysis': surprise_analysis
                },
                'details': self.results
            }, f, indent=2)
        
        print(f"\nSaved final results to: {results_file}")
        
        # Save complete pipeline output
        complete_output = {
            'run_id': self.run_id,
            'timestamp': datetime.now().isoformat(),
            'accuracy': accuracy,
            'results': self.results
        }
        
        complete_file = self.output_dir / "complete_pipeline_output.json"
        with open(complete_file, 'w') as f:
            json.dump(complete_output, f, indent=2)
        
        print(f"Saved complete output to: {complete_file}")
        
        return accuracy
    
    def run_complete_pipeline(self):
        """Run the complete pipeline from start to finish"""
        print("\n" + "="*60)
        print("IMPROVED RAG PIPELINE - COMPLETE RUN")
        print("="*60)
        
        # Step 1: Load filings
        self.step1_load_sample_filings(n=3)
        
        # Step 2: Chunk filings
        self.step2_chunk_filings()
        
        # Step 3: Set up ChromaDB with embeddings
        self.step3_setup_chromadb()
        
        # Step 4: Retrieve and generate predictions
        self.step4_retrieve_and_generate()
        
        # Step 5: Evaluate results
        accuracy = self.step5_evaluate_results()
        
        print("\n" + "="*60)
        print("PIPELINE COMPLETE!")
        print(f"All outputs saved to: {self.output_dir}")
        print("="*60)
        
        return accuracy


def main():
    """Run the improved pipeline"""
    pipeline = ImprovedRAGPipeline()
    accuracy = pipeline.run_complete_pipeline()
    
    print(f"\nFinal accuracy: {accuracy:.1%}")


if __name__ == "__main__":
    main()