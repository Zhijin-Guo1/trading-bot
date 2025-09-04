"""
Complete RAG pipeline test with 3 samples
Chunks -> OpenAI Embeddings -> Retrieval -> LLM Generation -> Results
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    OPENAI_API_KEY, QUERY_TEMPLATES, TOP_K_CHUNKS,
    SOURCE_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class CompletePipeline:
    """End-to-end RAG pipeline for testing"""
    
    def __init__(self):
        self.chunks = []
        self.embeddings = None
        self.results = []
    
    def load_sample_filings(self, n=3):
        """Load n sample filings from 2019 data"""
        print(f"\n=== Step 1: Loading {n} Sample Filings ===")
        
        # Load texts and labels
        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"
        
        with open(texts_file, 'r') as f:
            all_texts = json.load(f)
        
        labels_df = pd.read_csv(labels_file)
        
        # Take first n filings
        sample_filings = []
        for i, (accession, data) in enumerate(all_texts.items()):
            if i >= n:
                break
            
            # Get label data
            label_row = labels_df[labels_df['accession'] == accession]
            if not label_row.empty:
                data['signal'] = label_row.iloc[0]['signal']
                data['adjusted_return_pct'] = label_row.iloc[0]['adjusted_return_pct']
                data['accession'] = accession
                sample_filings.append(data)
                print(f"  {i+1}. {data['ticker']} - {data['filing_date']}: {len(data['text'])} chars")
        
        return sample_filings
    
    def chunk_filings(self, filings: List[Dict]):
        """Chunk the sample filings"""
        print(f"\n=== Step 2: Chunking Filings ===")
        
        all_chunks = []
        
        for filing in filings:
            text = filing['text']
            chunks = self.create_chunks(text, filing)
            all_chunks.extend(chunks)
            print(f"  {filing['ticker']}: {len(chunks)} chunks created")
        
        self.chunks = all_chunks
        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def create_chunks(self, text: str, filing: Dict) -> List[Dict]:
        """Simple chunking by character count with overlap"""
        chunks = []
        
        # Split into chunks
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            if len(chunk_text) < 100:  # Skip very small chunks
                continue
            
            chunk = {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'filing_date': filing['filing_date'],
                'chunk_text': chunk_text,
                'chunk_position': len(chunks),
                'signal': filing.get('signal', 'UNKNOWN'),
                'adjusted_return_pct': filing.get('adjusted_return_pct', 0.0)
            }
            chunks.append(chunk)
        
        return chunks
    
    def create_openai_embeddings(self):
        """Create embeddings using OpenAI API"""
        print(f"\n=== Step 3: Creating OpenAI Embeddings ===")
        
        embeddings = []
        batch_size = 20  # OpenAI allows up to 2048 tokens per request
        
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding chunks"):
            batch = self.chunks[i:i + batch_size]
            texts = [c['chunk_text'][:8000] for c in batch]  # Limit to 8000 chars
            
            # Call OpenAI embeddings API
            response = client.embeddings.create(
                model="text-phase1_embedding-ada-002",
                input=texts
            )
            
            # Extract embeddings
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
        
        self.embeddings = np.array(embeddings)
        print(f"Created embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def retrieve_relevant_chunks(self, filing_accession: str) -> List[Dict]:
        """Retrieve top-k chunks for a filing using similarity search"""
        
        # Get chunks for this filing
        filing_chunks = []
        filing_embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk['accession'] == filing_accession:
                filing_chunks.append(chunk)
                filing_embeddings.append(self.embeddings[i])
        
        if not filing_chunks:
            return []
        
        filing_embeddings = np.array(filing_embeddings)
        
        # Create query embeddings and score chunks
        all_scores = np.zeros(len(filing_chunks))
        
        for query_type, query_text in QUERY_TEMPLATES.items():
            # Get query phase1_embedding from OpenAI
            response = client.embeddings.create(
                model="text-phase1_embedding-ada-002",
                input=[query_text]
            )
            query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
            
            # Calculate cosine similarity
            similarities = cosine_similarity(query_embedding, filing_embeddings)[0]
            all_scores += similarities
        
        # Get top-k chunks
        top_indices = np.argsort(all_scores)[-TOP_K_CHUNKS:][::-1]
        
        retrieved = []
        for idx in top_indices:
            chunk = filing_chunks[idx].copy()
            chunk['retrieval_score'] = float(all_scores[idx])
            retrieved.append(chunk)
        
        return retrieved
    
    def generate_llm_prediction(self, retrieved_chunks: List[Dict], filing_info: Dict) -> Dict:
        """Generate prediction using GPT-3.5"""
        
        # Format chunks for prompt
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}] {chunk['chunk_text'][:500]}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""Analyze these excerpts from an 8-K filing for {filing_info['ticker']} on {filing_info['filing_date']}:

{chunks_text}

Based on these excerpts, predict the 5-day stock movement:
- UP: if likely to rise more than 1%
- DOWN: if likely to fall more than 1%
- STAY: if likely to stay within +/- 1%

Provide your analysis in JSON format:
{{
    "prediction": "UP/DOWN/STAY",
    "confidence": 0.0-1.0,
    "key_factors": ["factor1", "factor2"],
    "reasoning": "brief explanation"
}}"""

        # Call GPT-3.5
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst predicting stock movements from 8-K filings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=300
        )
        
        # Parse response
        response_text = response.choices[0].message.content
        
        try:
            # Extract JSON from response
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            analysis = json.loads(json_str)
        except:
            # Fallback
            analysis = {
                "prediction": "STAY",
                "confidence": 0.5,
                "key_factors": [],
                "reasoning": "Failed to parse response"
            }
        
        return analysis
    
    def run_complete_pipeline(self):
        """Run the complete pipeline on 3 samples"""
        
        # Load sample filings
        filings = self.load_sample_filings(n=3)
        
        # Chunk them
        self.chunk_filings(filings)
        
        # Create embeddings
        self.create_openai_embeddings()
        
        # Process each filing
        print(f"\n=== Step 4: Retrieval & Generation ===")
        
        for filing in filings:
            print(f"\n{filing['ticker']} - {filing['filing_date']}:")
            
            # Retrieve relevant chunks
            retrieved = self.retrieve_relevant_chunks(filing['accession'])
            print(f"  Retrieved {len(retrieved)} chunks")
            
            # Generate prediction
            analysis = self.generate_llm_prediction(retrieved, filing)
            print(f"  Prediction: {analysis['prediction']} (confidence: {analysis['confidence']:.2f})")
            print(f"  True label: {filing['signal']}")
            print(f"  Match: {'✓' if analysis['prediction'] == filing['signal'] else '✗'}")
            
            # Store result
            result = {
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'retrieved_chunks': len(retrieved),
                'total_chars': sum(len(c['chunk_text']) for c in retrieved),
                'prediction': analysis['prediction'],
                'confidence': analysis['confidence'],
                'reasoning': analysis['reasoning'],
                'key_factors': analysis['key_factors'],
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct'],
                'correct': analysis['prediction'] == filing['signal']
            }
            self.results.append(result)
        
        # Show final results
        self.show_results()
    
    def show_results(self):
        """Display final results"""
        print(f"\n=== Final Results ===")
        print(f"\n{'Ticker':<6} {'Date':<12} {'Predicted':<10} {'True':<10} {'Return':>8} {'Correct'}")
        print("-" * 60)
        
        for r in self.results:
            print(f"{r['ticker']:<6} {r['date']:<12} {r['prediction']:<10} {r['true_label']:<10} "
                  f"{r['true_return']:>7.2f}% {'✓' if r['correct'] else '✗'}")
        
        # Calculate accuracy
        correct = sum(1 for r in self.results if r['correct'])
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
        
        # Save results
        output_file = Path(__file__).parent.parent / "data" / "results" / "complete_pipeline_results.json"
        with open(output_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\nResults saved to: {output_file}")


def main():
    """Run the complete pipeline test"""
    print("=" * 60)
    print("COMPLETE RAG PIPELINE TEST - 3 SAMPLES")
    print("=" * 60)
    
    pipeline = CompletePipeline()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()