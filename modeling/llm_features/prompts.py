"""
Refined prompts for LLM feature extraction from 8-K filings
All prompts use only filing content and sector/industry context - no market data
"""

def get_subtopic_prompt(items_present, sector, industry, summary):
    """Task 1: Granular sub-topic classification beyond basic 8-K items"""
    return f"""Analyze this 8-K filing and classify into the most specific sub-topic.

8-K Items Reported: {items_present}
Company Sector: {sector}
Industry: {industry}
Filing Summary: {summary}

Categories:
EARNINGS & GUIDANCE:
- earnings_beat: Results exceeded consensus expectations
- earnings_miss: Results below consensus expectations  
- earnings_inline: Results met expectations
- guidance_raised: Increased forward outlook
- guidance_lowered: Decreased forward outlook
- guidance_withdrawn: Removed previous guidance

CORPORATE ACTIONS:
- acquisition: M&A announcement or completion
- divestiture: Asset sale or spinoff
- share_buyback: Stock repurchase program
- dividend_change: Dividend increase/decrease/suspension
- debt_restructuring: Refinancing or covenant changes

LEADERSHIP & GOVERNANCE:
- ceo_change: CEO departure or appointment
- cfo_change: CFO departure or appointment
- board_changes: Director additions or departures
- compensation_change: Executive comp modifications

OPERATIONAL:
- restructuring: Cost cutting or reorganization
- expansion: New facilities, markets, or products
- contract_win: Major customer or partnership
- regulatory_update: Approval, violation, or investigation

SECTOR-SPECIFIC:
- clinical_trial: (Biotech) Trial results or updates
- fda_action: (Biotech) FDA approval or rejection
- store_metrics: (Retail) Same-store sales or closures
- production_update: (Manufacturing) Output changes

OTHER:
- litigation: Legal proceedings or settlements
- accounting_change: Restatement or policy change
- cybersecurity: Data breach or security incident
- routine_filing: Administrative or non-material

Classify based on the PRIMARY material impact, not just form item.

Output JSON only:
{{"sub_topic": "...", "confidence": 0.0-1.0}}"""


def get_novelty_prompt(ticker, filing_date, current_summary):
    """Task 2: Assess novelty and significance of filing"""
    
    return f"""Analyze this 8-K filing to assess its novelty and significance.

Filing Date: {filing_date}
Ticker: {ticker}
Summary: {current_summary}

Questions to assess:
1. Is this filing reporting routine information or something unexpected?
2. Does this represent a significant change for the company?
3. How surprising would this be to investors following the company?
4. What are the key new developments or changes announced?

Rate novelty from 0 (completely routine/expected) to 1 (major surprise).

Output JSON only:
{{
  "novelty_score": 0.0-1.0,
  "change_tags": ["list", "of", "key", "developments"],
  "is_material": true/false,
  "surprise_level": "low|medium|high"
}}"""


def get_salience_prompt(sector, industry, summary):
    """Task 3: Event salience and impact assessment"""
    return f"""Assess the materiality and market impact potential of this 8-K filing.

Company Sector: {sector}
Industry: {industry}
Filing Summary: {summary}

Evaluate WITHOUT using market data:
1. How material is this event to the company's business model?
2. Does this affect core operations or is it peripheral?
3. Is this a one-time event or ongoing development?
4. How clear vs ambiguous is the outcome?

Sector-specific considerations:
- Tech: Product/platform changes, competitive positioning
- Healthcare: Clinical/regulatory milestones, reimbursement
- Financial: Credit quality, regulatory capital, loan losses
- Retail: Inventory, store footprint, e-commerce shift
- Energy: Production, reserves, commodity exposure
- Industrial: Capacity, backlog, supply chain

Output JSON only:
{{
  "salience_score": 0.0-1.0,
  "impact_magnitude": "minimal|moderate|significant|transformational",
  "time_horizon": "immediate|short_term|medium_term|long_term",
  "uncertainty_level": "low|medium|high",
  "business_impact": "core|peripheral|unclear"
}}"""


def get_tone_prompt(summary, sector):
    """Task 4: Financial tone analysis - continuous score"""
    return f"""Analyze the tone and sentiment of this 8-K filing from a financial perspective.

Filing Summary: {summary}
Sector Context: {sector}

Assess the tone considering:
1. Concrete metrics vs vague language
2. Management confidence vs defensiveness
3. Forward-looking optimism vs caution
4. Acknowledgment of challenges vs denial
5. Specificity of plans and timelines

Provide a SPECIFIC NUMERIC tone score between -1.0 and 1.0:
- -1.0: Extremely negative (bankruptcy, major crisis, severe guidance cut)
- -0.5: Negative (missed expectations, challenges, guidance lowered)
- 0.0: Neutral (routine disclosure, mixed signals, no clear direction)
- 0.5: Positive (met expectations, growth, guidance raised)
- 1.0: Extremely positive (major beat, transformational deal, huge growth)

Be precise with the score. Examples:
- Earnings beat by 10%: 0.35
- Guidance raised slightly: 0.25
- CEO resignation: -0.4
- Routine dividend: 0.05
- Major acquisition: 0.6

Output JSON only with a SPECIFIC DECIMAL NUMBER for tone_score:
{{
  "tone_score": <specific decimal between -1.0 and 1.0>,
  "confidence": <decimal between 0.0 and 1.0>,
  "tone_drivers": ["specific", "phrases", "from", "text"],
  "quantitative_support": true/false
}}"""


def get_risk_extraction_prompt(industry, summary):
    """Task 5: Risk and opportunity extraction"""
    return f"""Extract risk factors and positive catalysts from this 8-K filing.

Industry: {industry}
Filing Summary: {summary}

Identify IN CONTEXT (not just keywords):

RISK INDICATORS:
- Liquidity: cash concerns, covenant pressure, refinancing needs
- Operational: disruptions, cost overruns, margin pressure
- Competitive: market share loss, pricing pressure, disruption
- Regulatory: investigations, violations, pending actions
- Legal: litigation, settlements, liability exposure
- Macro: input costs, demand weakness, FX headwinds

OPPORTUNITY INDICATORS:  
- Growth: new markets, products, customers, expansion
- Efficiency: cost savings, synergies, productivity
- Pricing: power, increases, favorable mix
- Innovation: R&D success, patents, technology edge
- Market: share gains, competitive wins, consolidation

Context matters: "restructuring" could be defensive (risk) or offensive (opportunity).

Output JSON only:
{{
  "risk_factors": ["specific_risks_identified"],
  "positive_catalysts": ["specific_opportunities"],
  "net_risk_assessment": "high_risk|balanced|high_opportunity",
  "key_terms": {{"term": "contextual_meaning"}},
  "action_orientation": "defensive|neutral|offensive"
}}"""


def get_volatility_prompt(sector, summary):
    """Task 6: Volatility signal without market data"""
    return f"""Based solely on this 8-K filing content, assess the likelihood of significant stock price movement.

Sector: {sector}
Filing Summary: {summary}

Evaluate factors that typically drive volatility:
1. Surprise factor - how unexpected is this disclosure?
2. Magnitude - how big are the numbers or changes mentioned?
3. Clarity - is the outcome clear or uncertain?
4. Binary nature - is this a yes/no event or gradual development?
5. Guidance impact - does this change forward expectations?

Industry-specific volatility drivers:
- Biotech: Clinical trial binary events, FDA decisions
- Tech: Product launches, platform changes, user metrics
- Retail: Same-store sales surprises, guidance changes
- Financial: Credit losses, regulatory actions
- Energy: Production surprises, reserve changes

Estimate probability this news drives a large (>3%) stock move.

Output JSON only:
{{
  "volatility_score": 0.0-1.0,
  "surprise_factor": "low|medium|high",
  "outcome_clarity": "clear|developing|uncertain",
  "event_type": "binary|continuous|mixed",
  "guidance_impact": "none|confirms|changes",
  "expected_reaction": "muted|moderate|significant"
}}"""