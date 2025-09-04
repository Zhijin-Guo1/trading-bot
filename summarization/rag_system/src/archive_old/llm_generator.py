"""
LLM generation component using GPT-3.5 for chunk analysis
"""
import json
import os
from pathlib import Path
from typing import List, Dict
from openai import OpenAI
from dotenv import load_dotenv
import sys
sys.path.append(str(Path(__file__).parent.parent))

from config.settings import (
    OPENAI_API_KEY, LLM_MODEL, LLM_TEMPERATURE, MAX_TOKENS
)

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)


class LLMGenerator:
    """Generate analysis and predictions using GPT-3.5"""
    
    def __init__(self, model=LLM_MODEL, temperature=LLM_TEMPERATURE):
        self.model = model
        self.temperature = temperature
        
        if not OPENAI_API_KEY:
            raise ValueError("OpenAI API key not found in environment variables")
    
    def format_chunks_for_prompt(self, chunks: List[Dict]) -> str:
        """Format retrieved chunks for the prompt"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks, 1):
            item = chunk.get('item_number', 'UNKNOWN')
            text = chunk.get('chunk_text', '')
            score = chunk.get('retrieval_score', 0)
            
            formatted_chunks.append(f"[Chunk {i} - Item {item} (relevance: {score:.2f})]")
            formatted_chunks.append(text[:500])  # Limit each chunk to 500 chars
            formatted_chunks.append("")
        
        return "\n".join(formatted_chunks)
    
    def generate_analysis(self, retrieved_data: Dict) -> Dict:
        """Generate analysis from retrieved chunks"""
        
        # Format the chunks
        chunks_text = self.format_chunks_for_prompt(retrieved_data['retrieved_chunks'])
        
        # Create the prompt
        prompt = f"""Analyze these excerpts from an 8-K filing for {retrieved_data['ticker']} on {retrieved_data['filing_date']}:

{chunks_text}

Based on these excerpts, provide a structured analysis:

1. SUMMARY: Write 2-3 sentences summarizing the most important information that would impact stock price.

2. KEY FACTORS: List the main price-moving elements (e.g., "earnings_beat", "guidance_cut", "ceo_departure", "merger_announced"). Maximum 5 factors.

3. SENTIMENT: Classify the overall sentiment as POSITIVE, NEGATIVE, or NEUTRAL.

4. PREDICTION: Predict the 5-day stock movement:
   - UP: if likely to rise more than 1%
   - DOWN: if likely to fall more than 1%  
   - STAY: if likely to stay within +/- 1%

5. CONFIDENCE: Rate your confidence in the prediction from 0.0 to 1.0.

6. REASONING: Explain in 1-2 sentences why you made this prediction.

Format your response as JSON:
{{
    "summary": "...",
    "key_factors": ["factor1", "factor2", ...],
    "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
    "prediction": "UP/DOWN/STAY",
    "confidence": 0.85,
    "reasoning": "..."
}}"""

        try:
            # Call GPT-3.5
            response = client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a financial analyst expert at analyzing 8-K filings to predict stock movements."},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=MAX_TOKENS
            )
            
            # Extract the response
            response_text = response.choices[0].message.content
            
            # Try to parse as JSON
            try:
                # Find JSON in the response
                start_idx = response_text.find('{')
                end_idx = response_text.rfind('}') + 1
                if start_idx >= 0 and end_idx > start_idx:
                    json_str = response_text[start_idx:end_idx]
                    analysis = json.loads(json_str)
                else:
                    # Fallback if JSON not found
                    analysis = self.parse_text_response(response_text)
            except json.JSONDecodeError:
                # Fallback parsing
                analysis = self.parse_text_response(response_text)
            
            # Add metadata
            analysis['model'] = self.model
            analysis['tokens_used'] = response.usage.total_tokens
            
            return analysis
            
        except Exception as e:
            print(f"Error generating analysis: {e}")
            return self.get_default_analysis()
    
    def parse_text_response(self, text: str) -> Dict:
        """Parse non-JSON response"""
        # Simple parsing logic for fallback
        analysis = {
            "summary": "Unable to parse response",
            "key_factors": [],
            "sentiment": "NEUTRAL",
            "prediction": "STAY",
            "confidence": 0.5,
            "reasoning": "Parsing error"
        }
        
        # Try to extract prediction
        if "UP" in text.upper():
            analysis["prediction"] = "UP"
        elif "DOWN" in text.upper():
            analysis["prediction"] = "DOWN"
        
        # Try to extract sentiment
        if "POSITIVE" in text.upper():
            analysis["sentiment"] = "POSITIVE"
        elif "NEGATIVE" in text.upper():
            analysis["sentiment"] = "NEGATIVE"
        
        return analysis
    
    def get_default_analysis(self) -> Dict:
        """Return default analysis in case of error"""
        return {
            "summary": "Analysis failed",
            "key_factors": [],
            "sentiment": "NEUTRAL",
            "prediction": "STAY",
            "confidence": 0.0,
            "reasoning": "Error in analysis generation"
        }
    
    def batch_generate(self, retrieved_results: List[Dict]) -> List[Dict]:
        """Generate analysis for multiple filings"""
        results = []
        
        for data in retrieved_results:
            print(f"Generating analysis for {data['ticker']} - {data['filing_date']}")
            
            # Generate analysis
            analysis = self.generate_analysis(data)
            
            # Combine with retrieved data
            result = {
                **data,  # Include all retrieved data
                'llm_analysis': analysis,
                'llm_summary': analysis['summary'],
                'llm_sentiment': analysis['sentiment'],
                'llm_key_factors': analysis['key_factors'],
                'llm_prediction': analysis['prediction'],
                'llm_confidence': analysis['confidence'],
                'llm_reasoning': analysis['reasoning']
            }
            
            results.append(result)
        
        return results


def main():
    """Test LLM generation with sample retrieved chunks"""
    from pathlib import Path
    import json
    
    # Check if we have retrieved chunks
    chunks_file = Path(__file__).parent.parent / "data" / "chunks" / "retrieved_chunks_2019.json"
    
    if not chunks_file.exists():
        print("No retrieved chunks found. Please run rag_retriever.py first.")
        return
    
    # Load retrieved chunks
    with open(chunks_file, 'r') as f:
        retrieved_data = json.load(f)
    
    # Test with first filing
    if retrieved_data:
        generator = LLMGenerator()
        
        # Test with single filing
        test_data = retrieved_data[0]
        print(f"\nTesting with filing: {test_data['ticker']} - {test_data['filing_date']}")
        
        analysis = generator.generate_analysis(test_data)
        
        print("\nGenerated Analysis:")
        print(json.dumps(analysis, indent=2))
        
        # Save test result
        output_file = Path(__file__).parent.parent / "data" / "results" / "test_llm_analysis.json"
        with open(output_file, 'w') as f:
            json.dump(analysis, f, indent=2)
        
        print(f"\nAnalysis saved to: {output_file}")


if __name__ == "__main__":
    main()