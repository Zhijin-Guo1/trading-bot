"""
Final LLM Feature Extraction Pipeline for 8-K Filings
Simplified version without previous filing dependencies
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import time
from tqdm import tqdm
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
import os
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import prompts
from prompts import (
    get_subtopic_prompt,
    get_novelty_prompt,
    get_salience_prompt,
    get_tone_prompt,
    get_risk_extraction_prompt,
    get_volatility_prompt
)

# Configure logging
log_file = Path(__file__).parent / 'extraction_final.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class LLMFeatureExtractor:
    """Extract analytical features from 8-K filings using GPT-5-mini"""
    
    def __init__(self, use_mock=False):
        """Initialize the feature extractor"""
        self.use_mock = use_mock
        
        if not use_mock:
            from openai import OpenAI
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable must be set")
            self.client = OpenAI(api_key=api_key)
        else:
            self.client = None
            
        self.total_cost = 0
        self.total_tokens = 0
        self.success_count = 0
        self.error_count = 0
        
        # GPT-5-mini pricing (per 1M tokens)
        self.price_per_million = 0.25
        
    def parse_gpt5_response(self, response) -> str:
        """Parse GPT-5-mini response format"""
        try:
            if hasattr(response, 'output'):
                if isinstance(response.output, list) and len(response.output) > 0:
                    for item in response.output:
                        if hasattr(item, 'content') and item.content:
                            if isinstance(item.content, list) and len(item.content) > 0:
                                return item.content[0].text if hasattr(item.content[0], 'text') else str(item.content[0])
                            else:
                                return str(item.content)
                    return str(response.output)
                else:
                    return str(response.output)
            else:
                return str(response)
        except Exception as e:
            logger.error(f"Error parsing GPT-5 response: {e}")
            return str(response)
    
    def generate_mock_response(self, prompt: str) -> str:
        """Generate realistic mock responses for testing"""
        import random
        
        if "classify into the most specific sub-topic" in prompt:
            topics = ["earnings_beat", "guidance_raised", "acquisition", "ceo_change", "restructuring"]
            return json.dumps({
                "sub_topic": random.choice(topics),
                "confidence": round(random.uniform(0.6, 0.95), 2)
            })
        elif "novelty" in prompt.lower():
            return json.dumps({
                "novelty_score": round(random.uniform(0.3, 0.8), 2),
                "change_tags": random.choice([["earnings_update"], ["strategic_shift"], ["routine_disclosure"]]),
                "is_material": random.choice([True, False]),
                "surprise_level": random.choice(["low", "medium", "high"])
            })
        elif "salience" in prompt or "materiality" in prompt:
            return json.dumps({
                "salience_score": round(random.uniform(0.4, 0.9), 2),
                "impact_magnitude": random.choice(["minimal", "moderate", "significant"]),
                "time_horizon": random.choice(["immediate", "short_term", "medium_term"]),
                "uncertainty_level": random.choice(["low", "medium", "high"]),
                "business_impact": random.choice(["core", "peripheral"])
            })
        elif "tone" in prompt.lower():
            return json.dumps({
                "tone_score": round(random.uniform(-0.8, 0.8), 2),
                "confidence": round(random.uniform(0.6, 0.9), 2),
                "tone_drivers": ["key metric", "guidance change"],
                "quantitative_support": random.choice([True, False])
            })
        elif "risk" in prompt.lower():
            return json.dumps({
                "risk_factors": random.choice([["liquidity"], ["competition"], []]),
                "positive_catalysts": random.choice([["growth"], ["efficiency"], []]),
                "net_risk_assessment": random.choice(["high_risk", "balanced", "high_opportunity"]),
                "key_terms": {"metric": "context"},
                "action_orientation": random.choice(["defensive", "neutral", "offensive"])
            })
        elif "volatility" in prompt.lower():
            return json.dumps({
                "volatility_score": round(random.uniform(0.3, 0.8), 2),
                "surprise_factor": random.choice(["low", "medium", "high"]),
                "outcome_clarity": random.choice(["clear", "developing", "uncertain"]),
                "event_type": random.choice(["binary", "continuous", "mixed"]),
                "guidance_impact": random.choice(["none", "confirms", "changes"]),
                "expected_reaction": random.choice(["muted", "moderate", "significant"])
            })
        else:
            return "{}"
    
    def call_gpt5_mini(self, prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """Call GPT-5-mini API with retry logic"""
        if self.use_mock:
            mock_response = self.generate_mock_response(prompt)
            return {"status": "success", "response": mock_response}
            
        for attempt in range(max_retries):
            try:
                response = self.client.responses.create(
                    model="gpt-5-mini",
                    input=prompt,
                    reasoning={
                        "effort": "minimal"
                    }
                )
                
                output_text = self.parse_gpt5_response(response)
                tokens = response.usage.total_tokens if hasattr(response, 'usage') else len(prompt) // 4
                cost = tokens * self.price_per_million / 1000000
                
                self.total_tokens += tokens
                self.total_cost += cost
                
                return {
                    "status": "success",
                    "response": output_text,
                    "tokens": tokens,
                    "cost": cost
                }
                
            except Exception as e:
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1}: {str(e)[:100]}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed after {max_retries} attempts: {e}")
                    return {"status": "error", "response": str(e)[:200]}
    
    def parse_json_response(self, response_text: str, default_value: Dict) -> Dict:
        """Parse JSON from LLM response with fallback"""
        try:
            response_text = response_text.strip()
            
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]
            
            return json.loads(response_text)
        except Exception as e:
            logger.warning(f"Failed to parse JSON: {e}. Response: {response_text[:200]}")
            return default_value
    
    def task1_subtopic(self, context: Dict) -> Dict:
        """Extract sub-topic classification"""
        prompt = get_subtopic_prompt(
            context['items_present'],
            context['sector'],
            context['industry'],
            context['summary']
        )
        
        result = self.call_gpt5_mini(prompt)
        
        if result['status'] == 'success':
            parsed = self.parse_json_response(
                result['response'],
                {"sub_topic": "routine_filing", "confidence": 0.5}
            )
            return {
                'sub_topic': parsed.get('sub_topic', 'routine_filing'),
                'sub_topic_confidence': float(parsed.get('confidence', 0.5))
            }
        else:
            return {'sub_topic': 'error', 'sub_topic_confidence': 0.0}
    
    def task2_novelty(self, context: Dict) -> Dict:
        """Extract novelty score and significance"""
        prompt = get_novelty_prompt(
            context['ticker'],
            context['filing_date'],
            context['summary']
        )
        
        result = self.call_gpt5_mini(prompt)
        
        if result['status'] == 'success':
            parsed = self.parse_json_response(
                result['response'],
                {
                    "novelty_score": 0.5,
                    "change_tags": [],
                    "is_material": False,
                    "surprise_level": "medium"
                }
            )
            return {
                'novelty_score': float(parsed.get('novelty_score', 0.5)),
                'change_tags': json.dumps(parsed.get('change_tags', [])),
                'is_material': parsed.get('is_material', False),
                'surprise_level': parsed.get('surprise_level', 'medium')
            }
        else:
            return {
                'novelty_score': 0.5,
                'change_tags': '[]',
                'is_material': False,
                'surprise_level': 'error'
            }
    
    def task3_salience(self, context: Dict) -> Dict:
        """Extract event salience and impact assessment"""
        prompt = get_salience_prompt(
            context['sector'],
            context['industry'],
            context['summary']
        )
        
        result = self.call_gpt5_mini(prompt)
        
        if result['status'] == 'success':
            parsed = self.parse_json_response(
                result['response'],
                {
                    "salience_score": 0.5,
                    "impact_magnitude": "moderate",
                    "time_horizon": "short_term",
                    "uncertainty_level": "medium",
                    "business_impact": "unclear"
                }
            )
            return {
                'salience_score': float(parsed.get('salience_score', 0.5)),
                'impact_magnitude': parsed.get('impact_magnitude', 'moderate'),
                'time_horizon': parsed.get('time_horizon', 'short_term'),
                'uncertainty_level': parsed.get('uncertainty_level', 'medium'),
                'business_impact': parsed.get('business_impact', 'unclear')
            }
        else:
            return {
                'salience_score': 0.5,
                'impact_magnitude': 'error',
                'time_horizon': 'error',
                'uncertainty_level': 'high',
                'business_impact': 'error'
            }
    
    def task4_tone(self, context: Dict) -> Dict:
        """Extract financial tone analysis - continuous score"""
        prompt = get_tone_prompt(
            context['summary'],
            context['sector']
        )
        
        result = self.call_gpt5_mini(prompt)
        
        if result['status'] == 'success':
            parsed = self.parse_json_response(
                result['response'],
                {
                    "tone_score": 0.0,
                    "confidence": 0.5,
                    "tone_drivers": [],
                    "quantitative_support": False
                }
            )
            tone_score = float(parsed.get('tone_score', 0.0))
            tone_score = max(-1.0, min(1.0, tone_score))
            
            return {
                'tone_score': tone_score,
                'tone_confidence': float(parsed.get('confidence', 0.5)),
                'tone_drivers': json.dumps(parsed.get('tone_drivers', [])),
                'quantitative_support': parsed.get('quantitative_support', False)
            }
        else:
            return {
                'tone_score': 0.0,
                'tone_confidence': 0.0,
                'tone_drivers': '[]',
                'quantitative_support': False
            }
    
    def task5_risks(self, context: Dict) -> Dict:
        """Extract risk factors and opportunities"""
        prompt = get_risk_extraction_prompt(
            context['industry'],
            context['summary']
        )
        
        result = self.call_gpt5_mini(prompt)
        
        if result['status'] == 'success':
            parsed = self.parse_json_response(
                result['response'],
                {
                    "risk_factors": [],
                    "positive_catalysts": [],
                    "net_risk_assessment": "balanced",
                    "key_terms": {},
                    "action_orientation": "neutral"
                }
            )
            return {
                'risk_factors': json.dumps(parsed.get('risk_factors', [])),
                'positive_catalysts': json.dumps(parsed.get('positive_catalysts', [])),
                'net_risk_assessment': parsed.get('net_risk_assessment', 'balanced'),
                'action_orientation': parsed.get('action_orientation', 'neutral')
            }
        else:
            return {
                'risk_factors': '[]',
                'positive_catalysts': '[]',
                'net_risk_assessment': 'error',
                'action_orientation': 'error'
            }
    
    def task6_volatility(self, context: Dict) -> Dict:
        """Extract volatility signal"""
        prompt = get_volatility_prompt(
            context['sector'],
            context['summary']
        )
        
        result = self.call_gpt5_mini(prompt)
        
        if result['status'] == 'success':
            parsed = self.parse_json_response(
                result['response'],
                {
                    "volatility_score": 0.5,
                    "surprise_factor": "medium",
                    "outcome_clarity": "developing",
                    "event_type": "continuous",
                    "guidance_impact": "none",
                    "expected_reaction": "moderate"
                }
            )
            return {
                'volatility_score': float(parsed.get('volatility_score', 0.5)),
                'surprise_factor': parsed.get('surprise_factor', 'medium'),
                'outcome_clarity': parsed.get('outcome_clarity', 'developing'),
                'event_type': parsed.get('event_type', 'continuous'),
                'expected_reaction': parsed.get('expected_reaction', 'moderate')
            }
        else:
            return {
                'volatility_score': 0.5,
                'surprise_factor': 'error',
                'outcome_clarity': 'error',
                'event_type': 'error',
                'expected_reaction': 'error'
            }
    
    def extract_all_features(self, row: pd.Series) -> Dict:
        """Extract all 6 feature sets for a single filing"""
        
        context = {
            'ticker': row['ticker'],
            'filing_date': row['filing_date'],
            'summary': row['summary'],
            'items_present': row['items_present'],
            'sector': row['sector'],
            'industry': row['industry']
        }
        
        # Run all 6 tasks in parallel
        with ThreadPoolExecutor(max_workers=12) as executor:
            futures = {
                executor.submit(self.task1_subtopic, context): 'task1',
                executor.submit(self.task2_novelty, context): 'task2',
                executor.submit(self.task3_salience, context): 'task3',
                executor.submit(self.task4_tone, context): 'task4',
                executor.submit(self.task5_risks, context): 'task5',
                executor.submit(self.task6_volatility, context): 'task6'
            }
            
            results = {}
            for future in as_completed(futures):
                task_name = futures[future]
                try:
                    results[task_name] = future.result()
                    self.success_count += 1
                except Exception as e:
                    logger.error(f"Task {task_name} failed: {e}")
                    self.error_count += 1
                    results[task_name] = self.get_default_values(task_name)
        
        # Combine all results with original row data
        combined = row.to_dict()
        for task_results in results.values():
            combined.update(task_results)
        
        return combined
    
    def get_default_values(self, task_name: str) -> Dict:
        """Get default values for failed tasks"""
        defaults = {
            'task1': {'sub_topic': 'error', 'sub_topic_confidence': 0.0},
            'task2': {'novelty_score': 0.5, 'change_tags': '[]', 'is_material': False, 'surprise_level': 'error'},
            'task3': {'salience_score': 0.5, 'impact_magnitude': 'error', 'time_horizon': 'error', 
                     'uncertainty_level': 'high', 'business_impact': 'error'},
            'task4': {'tone_score': 0.0, 'tone_confidence': 0.0, 'tone_drivers': '[]', 'quantitative_support': False},
            'task5': {'risk_factors': '[]', 'positive_catalysts': '[]', 'net_risk_assessment': 'error', 
                     'action_orientation': 'error'},
            'task6': {'volatility_score': 0.5, 'surprise_factor': 'error', 'outcome_clarity': 'error',
                     'event_type': 'error', 'expected_reaction': 'error'}
        }
        return defaults.get(task_name, {})
    
    def process_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Process a batch of filings in parallel"""
        results = []
        
        with ThreadPoolExecutor(max_workers=20) as executor:
            futures = {
                executor.submit(self.extract_all_features, row): idx 
                for idx, row in df.iterrows()
            }
            
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing batch"):
                idx = futures[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed processing row {idx}: {e}")
                    error_row = df.loc[idx].to_dict()
                    for task_num in range(1, 7):
                        error_row.update(self.get_default_values(f'task{task_num}'))
                    results.append(error_row)
        
        return pd.DataFrame(results)


def main():
    """Main execution function"""
    logger.info("Starting LLM feature extraction pipeline (FINAL)")
    
    # Load filtered data
    data_path = Path('modeling/phase1_ml/data/train.csv')
    if not data_path.exists():
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info(f"Loading data from {data_path}")
    train_df = pd.read_csv(data_path)
    logger.info(f"Loaded {len(train_df)} rows")
    
    # NO NEED FOR PREVIOUS FILING DATA
    
    # Initialize extractor
    extractor = LLMFeatureExtractor(use_mock=False)
    
    # Process in batches
    batch_size = 50
    all_results = []
    
    output_dir = Path('modeling/llm_features')
    output_dir.mkdir(exist_ok=True)
    
    logger.info(f"Processing {len(train_df)} filings in batches of {batch_size}")
    
    for i in range(0, len(train_df), batch_size):
        batch_num = i // batch_size + 1
        logger.info(f"Processing batch {batch_num}/{(len(train_df) + batch_size - 1) // batch_size}")
        
        batch = train_df.iloc[i:i+batch_size]
        batch_results = extractor.process_batch(batch)
        all_results.append(batch_results)
        
        # Save checkpoint every 500 rows
        if i % 500 == 0 and i > 0:
            checkpoint_df = pd.concat(all_results, ignore_index=True)
            checkpoint_path = output_dir / 'checkpoint_train.csv'
            checkpoint_df.to_csv(checkpoint_path, index=False)
            logger.info(f"Checkpoint saved: {len(checkpoint_df)} rows processed")
            logger.info(f"Stats - Success: {extractor.success_count}, Errors: {extractor.error_count}")
            logger.info(f"Cost so far: ${extractor.total_cost:.2f}, Tokens: {extractor.total_tokens:,}")
    
    # Combine and save final results
    logger.info("Combining all results")
    final_df = pd.concat(all_results, ignore_index=True)
    
    output_path = output_dir / 'enhanced_train_final.csv'
    final_df.to_csv(output_path, index=False)
    logger.info(f"Final results saved to {output_path}")
    
    # Save extraction metrics
    metrics = {
        'total_rows': len(final_df),
        'total_cost': extractor.total_cost,
        'total_tokens': extractor.total_tokens,
        'success_count': extractor.success_count,
        'error_count': extractor.error_count,
        'success_rate': extractor.success_count / (extractor.success_count + extractor.error_count) if (extractor.success_count + extractor.error_count) > 0 else 0,
        'timestamp': datetime.now().isoformat()
    }
    
    metrics_path = output_dir / 'extraction_metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.info("=" * 50)
    logger.info("Extraction complete!")
    logger.info(f"Total rows processed: {metrics['total_rows']}")
    logger.info(f"Total cost: ${metrics['total_cost']:.2f}")
    logger.info(f"Total tokens: {metrics['total_tokens']:,}")
    logger.info(f"Success rate: {metrics['success_rate']:.1%}")
    logger.info(f"Results saved to: {output_path}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()