"""
Test complete pipeline using ChromaDB for retrieval
"""
import json
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import OPENAI_API_KEY, CHROMADB_DIR, TOP_K_CHUNKS

client = OpenAI(api_key=OPENAI_API_KEY)


def test_chromadb_pipeline():
    """Test retrieval and generation using ChromaDB"""
    
    print("=== Testing RAG Pipeline with ChromaDB ===\n")
    
    # 1. Connect to ChromaDB
    print("1. Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path=str(CHROMADB_DIR))
    
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key=OPENAI_API_KEY,
        model_name="text-phase1_embedding-ada-002"
    )
    
    collection = chroma_client.get_collection(
        name="eight_k_chunks",
        embedding_function=openai_ef
    )
    print(f"   Connected to collection with {collection.count()} documents\n")
    
    # 2. Get unique filings in collection
    print("2. Getting unique filings...")
    all_docs = collection.get()
    unique_filings = {}
    
    for i, metadata in enumerate(all_docs['metadatas']):
        accession = metadata['accession']
        if accession not in unique_filings:
            unique_filings[accession] = {
                'ticker': metadata['ticker'],
                'date': metadata['filing_date'],
                'signal': metadata.get('signal', 'UNKNOWN'),
                'return': metadata.get('return', 0.0)
            }
    
    print(f"   Found {len(unique_filings)} unique filings\n")
    
    # 3. Process each filing
    results = []
    
    for accession, filing_info in unique_filings.items():
        print(f"\n{filing_info['ticker']} - {filing_info['date']}:")
        
        # Retrieve relevant chunks using multiple queries
        queries = [
            "quarterly earnings revenue profit guidance forecast",
            "CEO CFO resignation departure management change",
            "merger acquisition partnership strategic agreement"
        ]
        
        all_chunks = []
        all_distances = []
        
        for query in queries:
            query_results = collection.query(
                query_texts=[query],
                where={"accession": accession},
                n_results=TOP_K_CHUNKS
            )
            
            if query_results['ids'][0]:
                for doc, dist, meta in zip(
                    query_results['documents'][0],
                    query_results['distances'][0],
                    query_results['metadatas'][0]
                ):
                    all_chunks.append(doc)
                    all_distances.append(dist)
        
        # Deduplicate and take top chunks
        unique_chunks = []
        seen = set()
        for chunk, dist in sorted(zip(all_chunks, all_distances), key=lambda x: x[1]):
            if chunk[:100] not in seen:
                seen.add(chunk[:100])
                unique_chunks.append(chunk)
                if len(unique_chunks) >= TOP_K_CHUNKS:
                    break
        
        print(f"  Retrieved {len(unique_chunks)} unique chunks")
        
        # Generate prediction
        if unique_chunks:
            chunks_text = "\n\n".join([
                f"[Chunk {i+1}] {chunk[:400]}"
                for i, chunk in enumerate(unique_chunks)
            ])
            
            prompt = f"""Analyze these excerpts from an 8-K filing for {filing_info['ticker']} on {filing_info['date']}:

{chunks_text}

Predict the 5-day stock movement and provide analysis:

{{
    "prediction": "UP/DOWN/STAY",
    "confidence": 0.0-1.0,
    "summary": "brief summary",
    "key_factors": ["factor1", "factor2"]
}}"""

            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are a financial analyst."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=300
            )
            
            try:
                response_text = response.choices[0].message.content
                start = response_text.find('{')
                end = response_text.rfind('}') + 1
                analysis = json.loads(response_text[start:end])
            except:
                analysis = {
                    "prediction": "STAY",
                    "confidence": 0.5,
                    "summary": "Failed to parse",
                    "key_factors": []
                }
            
            correct = analysis['prediction'] == filing_info['signal']
            
            print(f"  Prediction: {analysis['prediction']} (confidence: {analysis['confidence']:.2f})")
            print(f"  True label: {filing_info['signal']}")
            print(f"  Match: {'✓' if correct else '✗'}")
            print(f"  Summary: {analysis['summary'][:100]}...")
            
            results.append({
                'ticker': filing_info['ticker'],
                'date': filing_info['date'],
                'prediction': analysis['prediction'],
                'true_label': filing_info['signal'],
                'true_return': filing_info['return'],
                'correct': correct,
                'confidence': analysis['confidence'],
                'summary': analysis['summary'],
                'key_factors': analysis['key_factors'],
                'chunks_retrieved': len(unique_chunks)
            })
    
    # 4. Show final results
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    
    correct = sum(1 for r in results if r['correct'])
    total = len(results)
    
    print(f"\nAccuracy: {correct}/{total} = {correct/total:.1%}")
    
    for r in results:
        print(f"\n{r['ticker']} ({r['date']}):")
        print(f"  Predicted: {r['prediction']} | True: {r['true_label']} | {r['correct'] and '✓' or '✗'}")
        print(f"  Key factors: {', '.join(r['key_factors'][:3])}")
    
    # Save results
    output_file = Path(__file__).parent.parent / "data" / "results" / "chromadb_test_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    test_chromadb_pipeline()