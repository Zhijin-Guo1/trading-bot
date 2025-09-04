"""
Improved prompts for RAG retrieval and generation
Focused on market-moving signals and surprises
"""

# More targeted retrieval queries focusing on deviations and surprises
RETRIEVAL_QUERIES = {
    "earnings_surprise": "earnings beat miss exceed fell short consensus expectations surprise disappointing better worse than expected revenue guidance",
    
    "guidance_change": "revised guidance raised lowered withdrawn suspended updated outlook forecast changing increasing decreasing accelerating slowing",
    
    "unexpected_events": "unexpected surprised unusual extraordinary one-time charge impairment restructuring discontinued acquisition terminated cancelled delayed",
    
    "management_turmoil": "resigned fired terminated abruptly suddenly departure CEO CFO investigation scandal fraud restatement weakness",
    
    "material_contracts": "material agreement contract partnership strategic alliance joint venture licensing deal billion million value worth",
    
    "regulatory_issues": "SEC FDA investigation subpoena warning letter violation penalty fine sanctions regulatory compliance failure",
    
    "competitive_dynamics": "market share gain loss competition competitive advantage disadvantage pricing pressure margin compression expansion",
    
    "operational_metrics": "backlog bookings orders pipeline conversion utilization capacity efficiency productivity yield defect recall"
}

# Context-aware generation prompt
GENERATION_PROMPT_TEMPLATE = """You are an expert financial analyst specializing in event-driven trading. Analyze these excerpts from an 8-K filing to predict short-term stock price movement.

Company: {ticker}
Filing Date: {filing_date}
Current Market Context: VIX level indicates {vix_context} volatility

Key Excerpts from 8-K Filing:
{chunks_text}

CRITICAL ANALYSIS FRAMEWORK:

1. SURPRISE FACTOR: Are these events expected or surprising?
   - Compare actual numbers to typical quarterly changes
   - Identify any language suggesting surprise ("unexpected", "revised", "terminated")
   - Note if guidance is being changed

2. MAGNITUDE ASSESSMENT: How significant are these events?
   - Quantify financial impact if possible (% revenue, % earnings)
   - Assess if events are one-time or recurring
   - Determine if events affect core business or periphery

3. TIMING CONSIDERATION: When will impact be felt?
   - Immediate impact (current quarter)
   - Future impact (next quarters)
   - Already priced in (old news)

4. MARKET REACTION PREDICTION:
   Based on the above analysis, predict 5-day price movement:
   - UP: Positive surprise or better-than-expected news likely to drive buying
   - DOWN: Negative surprise or worse-than-expected news likely to drive selling  
   - STAY: Information is neutral, expected, or already priced in

Provide your analysis in this JSON format:
{{
    "surprise_level": "HIGH/MEDIUM/LOW",
    "surprise_factors": ["specific surprising elements"],
    
    "magnitude": "LARGE/MODERATE/SMALL",
    "financial_impact": "quantified impact if available",
    
    "sentiment": "POSITIVE/NEGATIVE/NEUTRAL",
    "key_positives": ["positive factors"],
    "key_negatives": ["negative factors"],
    
    "prediction": "UP/DOWN/STAY",
    "confidence": 0.0-1.0,
    "primary_driver": "main reason for prediction",
    
    "reasoning": "2-3 sentences explaining the prediction based on surprise and magnitude",
    
    "risk_factors": ["what could make this prediction wrong"]
}}"""

# Short-form prompt for quick predictions
QUICK_PREDICTION_PROMPT = """Analyze this 8-K filing excerpt for {ticker} on {filing_date}:

{chunks_text}

Focus on SURPRISES and DEVIATIONS from normal business:
1. What's unexpected here?
2. How big is the impact?
3. Will it move the stock?

Predict 5-day movement (UP >1%, DOWN <-1%, STAY within 1%):
{{
    "prediction": "UP/DOWN/STAY",
    "confidence": 0.0-1.0,
    "key_surprise": "most surprising element",
    "impact_size": "LARGE/MEDIUM/SMALL"
}}"""