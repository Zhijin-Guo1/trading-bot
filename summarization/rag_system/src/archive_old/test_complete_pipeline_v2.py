"""
Complete RAG pipeline test with 3 samples - Version 2
Saves all intermediate outputs including LLM summaries
"""
import json
import numpy as np
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
import pandas as pd
from datetime import datetime
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    OPENAI_API_KEY, QUERY_TEMPLATES, TOP_K_CHUNKS,
    SOURCE_DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP,
    RESULTS_DIR
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class CompletePipelineV2:
    """End-to-end RAG pipeline with comprehensive saving"""
    
    def __init__(self, run_id=None):
        if run_id is None:
            run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        self.run_id = run_id
        self.output_dir = RESULTS_DIR / f"pipeline_run_{run_id}"
        self.output_dir.mkdir(exist_ok=True)
        
        self.chunks = []
        self.embeddings = None
        self.results = []
        self.all_outputs = {
            'run_id': run_id,
            'filings': [],
            'chunks': [],
            'embeddings': [],
            'retrievals': [],
            'generations': [],
            'final_results': []
        }
    
    def save_intermediate(self, data_type: str, data):
        """Save intermediate outputs"""
        output_file = self.output_dir / f"{data_type}.json"
        
        if data_type == "embeddings":
            # Save numpy array
            np_file = self.output_dir / f"{data_type}.npy"
            np.save(np_file, data)
            print(f"  Saved embeddings to: {np_file}")
        else:
            # Save JSON
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print(f"  Saved {data_type} to: {output_file}")
    
    def load_sample_filings(self, n=3):
        """Load n sample filings from 2019 data"""
        print(f"\n=== Step 1: Loading {n} Sample Filings ===")
        
        texts_file = SOURCE_DATA_DIR / "texts" / "merged_texts_2019.json"
        labels_file = SOURCE_DATA_DIR / "enhanced_dataset_2019_2019.csv"
        
        with open(texts_file, 'r') as f:
            all_texts = json.load(f)
        
        labels_df = pd.read_csv(labels_file)
        
        sample_filings = []
        for i, (accession, data) in enumerate(all_texts.items()):
            if i >= n:
                break
            
            label_row = labels_df[labels_df['accession'] == accession]
            if not label_row.empty:
                data['signal'] = label_row.iloc[0]['signal']
                data['adjusted_return_pct'] = label_row.iloc[0]['adjusted_return_pct']
                data['accession'] = accession
                sample_filings.append(data)
                print(f"  {i+1}. {data['ticker']} - {data['filing_date']}: {len(data['text'])} chars")
        
        # Save filings
        self.all_outputs['filings'] = sample_filings
        self.save_intermediate('filings', sample_filings)
        
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
        self.all_outputs['chunks'] = all_chunks
        self.save_intermediate('chunks', all_chunks)
        
        print(f"Total chunks: {len(all_chunks)}")
        return all_chunks
    
    def create_chunks(self, text: str, filing: Dict) -> List[Dict]:
        """Simple chunking by character count with overlap"""
        chunks = []
        
        for i in range(0, len(text), CHUNK_SIZE - CHUNK_OVERLAP):
            chunk_text = text[i:i + CHUNK_SIZE]
            if len(chunk_text) < 100:
                continue
            
            chunk = {
                'chunk_id': f"{filing['accession']}_{len(chunks)}",
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'filing_date': filing['filing_date'],
                'chunk_text': chunk_text,
                'chunk_position': len(chunks),
                'chunk_length': len(chunk_text),
                'signal': filing.get('signal', 'UNKNOWN'),
                'adjusted_return_pct': filing.get('adjusted_return_pct', 0.0)
            }
            chunks.append(chunk)
        
        return chunks
    
    def create_openai_embeddings(self):
        """Create embeddings using OpenAI API"""
        print(f"\n=== Step 3: Creating OpenAI Embeddings ===")
        
        embeddings = []
        embedding_metadata = []
        batch_size = 20
        
        for i in tqdm(range(0, len(self.chunks), batch_size), desc="Embedding chunks"):
            batch = self.chunks[i:i + batch_size]
            texts = [c['chunk_text'][:8000] for c in batch]
            
            response = client.embeddings.create(
                model="text-phase1_embedding-ada-002",
                input=texts
            )
            
            batch_embeddings = [e.embedding for e in response.data]
            embeddings.extend(batch_embeddings)
            
            # Save metadata
            for j, chunk in enumerate(batch):
                embedding_metadata.append({
                    'chunk_id': chunk['chunk_id'],
                    'embedding_index': i + j,
                    'accession': chunk['accession']
                })
        
        self.embeddings = np.array(embeddings)
        
        # Save embeddings and metadata
        self.save_intermediate('embeddings', self.embeddings)
        self.save_intermediate('embedding_metadata', embedding_metadata)
        
        print(f"Created embeddings: {self.embeddings.shape}")
        return self.embeddings
    
    def retrieve_relevant_chunks(self, filing_accession: str) -> List[Dict]:
        """Retrieve top-k chunks for a filing using similarity search"""
        
        filing_chunks = []
        filing_embeddings = []
        
        for i, chunk in enumerate(self.chunks):
            if chunk['accession'] == filing_accession:
                filing_chunks.append(chunk)
                filing_embeddings.append(self.embeddings[i])
        
        if not filing_chunks:
            return []
        
        filing_embeddings = np.array(filing_embeddings)
        
        # Score chunks against all queries
        all_scores = np.zeros(len(filing_chunks))
        query_scores_detail = {}
        
        for query_type, query_text in QUERY_TEMPLATES.items():
            response = client.embeddings.create(
                model="text-phase1_embedding-ada-002",
                input=[query_text]
            )
            query_embedding = np.array(response.data[0].embedding).reshape(1, -1)
            
            similarities = cosine_similarity(query_embedding, filing_embeddings)[0]
            all_scores += similarities
            query_scores_detail[query_type] = similarities.tolist()
        
        # Get top-k chunks
        top_indices = np.argsort(all_scores)[-TOP_K_CHUNKS:][::-1]
        
        retrieved = []
        for idx in top_indices:
            chunk = filing_chunks[idx].copy()
            chunk['retrieval_score'] = float(all_scores[idx])
            chunk['query_scores'] = {q: scores[idx] for q, scores in query_scores_detail.items()}
            retrieved.append(chunk)
        
        return retrieved
    
    def generate_llm_prediction(self, retrieved_chunks: List[Dict], filing_info: Dict) -> Dict:
        """Generate prediction using GPT-3.5 with comprehensive output"""
        
        chunks_text = "\n\n".join([
            f"[Chunk {i+1}] {chunk['chunk_text'][:500]}"
            for i, chunk in enumerate(retrieved_chunks)
        ])
        
        prompt = f"""Analyze these excerpts from an 8-K filing for {filing_info['ticker']} on {filing_info['filing_date']}:

{chunks_text}

Provide a comprehensive analysis:

1. SUMMARY: 2-3 sentences summarizing the key information
2. SENTIMENT: Overall sentiment (POSITIVE/NEGATIVE/NEUTRAL)
3. KEY_EVENTS: List the main events or announcements
4. FINANCIAL_METRICS: Any specific numbers or financial data mentioned
5. PREDICTION: 5-day stock movement (UP/DOWN/STAY)
6. CONFIDENCE: Your confidence level (0.0-1.0)
7. KEY_FACTORS: Factors driving your prediction
8. REASONING: Detailed explanation of your prediction

Format as JSON:
{{
    "summary": "...",
    "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
    "key_events": ["event1", "event2"],
    "financial_metrics": ["metric1", "metric2"],
    "prediction": "UP/DOWN/STAY",
    "confidence": 0.0-1.0,
    "key_factors": ["factor1", "factor2"],
    "reasoning": "..."
}}"""

        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a financial analyst predicting stock movements from 8-K filings."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=500
        )
        
        response_text = response.choices[0].message.content
        
        try:
            start = response_text.find('{')
            end = response_text.rfind('}') + 1
            json_str = response_text[start:end]
            analysis = json.loads(json_str)
        except:
            analysis = {
                "summary": "Failed to parse response",
                "sentiment": "NEUTRAL",
                "key_events": [],
                "financial_metrics": [],
                "prediction": "STAY",
                "confidence": 0.5,
                "key_factors": [],
                "reasoning": "Failed to parse response"
            }
        
        # Add token usage
        analysis['tokens_used'] = response.usage.total_tokens
        analysis['model'] = "gpt-3.5-turbo"
        
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
        
        all_retrievals = []
        all_generations = []
        
        for filing in filings:
            print(f"\n{filing['ticker']} - {filing['filing_date']}:")
            
            # Retrieve relevant chunks
            retrieved = self.retrieve_relevant_chunks(filing['accession'])
            print(f"  Retrieved {len(retrieved)} chunks")
            
            retrieval_record = {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'retrieved_chunks': retrieved,
                'total_retrieved_chars': sum(len(c['chunk_text']) for c in retrieved)
            }
            all_retrievals.append(retrieval_record)
            
            # Generate prediction
            analysis = self.generate_llm_prediction(retrieved, filing)
            print(f"  Prediction: {analysis['prediction']} (confidence: {analysis['confidence']:.2f})")
            print(f"  True label: {filing['signal']}")
            print(f"  Match: {'✓' if analysis['prediction'] == filing['signal'] else '✗'}")
            
            generation_record = {
                'accession': filing['accession'],
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'llm_analysis': analysis,
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct']
            }
            all_generations.append(generation_record)
            
            # Store result
            result = {
                'ticker': filing['ticker'],
                'date': filing['filing_date'],
                'retrieved_chunks': len(retrieved),
                'total_chars': sum(len(c['chunk_text']) for c in retrieved),
                'summary': analysis['summary'],
                'sentiment': analysis['sentiment'],
                'key_events': analysis['key_events'],
                'financial_metrics': analysis['financial_metrics'],
                'prediction': analysis['prediction'],
                'confidence': analysis['confidence'],
                'reasoning': analysis['reasoning'],
                'key_factors': analysis['key_factors'],
                'true_label': filing['signal'],
                'true_return': filing['adjusted_return_pct'],
                'correct': analysis['prediction'] == filing['signal'],
                'tokens_used': analysis['tokens_used']
            }
            self.results.append(result)
        
        # Save all outputs
        self.save_intermediate('retrievals', all_retrievals)
        self.save_intermediate('generations', all_generations)
        self.save_intermediate('final_results', self.results)
        
        # Save complete output
        self.all_outputs['retrievals'] = all_retrievals
        self.all_outputs['generations'] = all_generations
        self.all_outputs['final_results'] = self.results
        
        complete_output_file = self.output_dir / 'complete_pipeline_output.json'
        with open(complete_output_file, 'w') as f:
            json.dump(self.all_outputs, f, indent=2)
        
        print(f"\n=== All outputs saved to: {self.output_dir} ===")
        
        # Show results
        self.show_results()
    
    def show_results(self):
        """Display final results"""
        print(f"\n=== Final Results ===")
        print(f"\n{'Ticker':<6} {'Date':<12} {'Predicted':<10} {'True':<10} {'Return':>8} {'Correct'}")
        print("-" * 60)
        
        for r in self.results:
            print(f"{r['ticker']:<6} {r['date']:<12} {r['prediction']:<10} {r['true_label']:<10} "
                  f"{r['true_return']:>7.2f}% {'✓' if r['correct'] else '✗'}")
        
        correct = sum(1 for r in self.results if r['correct'])
        total = len(self.results)
        accuracy = correct / total if total > 0 else 0
        
        print(f"\nAccuracy: {accuracy:.1%} ({correct}/{total})")
        
        # Show summaries
        print(f"\n=== LLM Summaries ===")
        for r in self.results:
            print(f"\n{r['ticker']} ({r['date']}):")
            print(f"  Summary: {r['summary']}")
            print(f"  Key Events: {', '.join(r['key_events'])}")
            print(f"  Sentiment: {r['sentiment']}")


def main():
    """Run the complete pipeline test"""
    print("=" * 60)
    print("COMPLETE RAG PIPELINE V2 - WITH FULL SAVING")
    print("=" * 60)
    
    pipeline = CompletePipelineV2()
    pipeline.run_complete_pipeline()


if __name__ == "__main__":
    main()