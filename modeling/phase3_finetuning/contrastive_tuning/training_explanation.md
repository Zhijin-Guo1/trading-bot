# How Contrastive Fine-Tuning Actually Works

## The Training Process: Next Token Prediction

### 1. Training Data Format

```json
{
  "instruction": "Compare Filing A... Filing B... Which is better?",
  "input": "",
  "output": "Filing A resulted in better performance..."
}
```

### 2. What the Model Actually Sees

The tokenizer concatenates this into a single sequence:

```
[INST] Compare Filing A (earnings beat) and Filing B (earnings miss). Which is better? [/INST] Filing A resulted in better performance. Analysis: Filing A shows stronger positive signals...<EOS>
```

### 3. The Actual Training Task

For each token position, the model:

```python
# Position 1: [INST] Compare Filing A...Which is better? [/INST] Filing
# Target: "A"  
# Loss: CrossEntropy(predicted_token, "A")

# Position 2: [INST] Compare Filing A...Which is better? [/INST] Filing A
# Target: "resulted"
# Loss: CrossEntropy(predicted_token, "resulted")

# Position 3: [INST] Compare Filing A...Which is better? [/INST] Filing A resulted
# Target: "in"
# Loss: CrossEntropy(predicted_token, "in")

# And so on for every token in the output...
```

### 4. What Makes Fine-Tuning Different from Pre-Training

**Pre-training**: Learn from any text
```
"The stock market closed higher today" → predict "today"
```

**Fine-tuning**: Learn specific pattern
```
"Filing with earnings beat" → predict "Filing A performs better"
```

### 5. LoRA (Low-Rank Adaptation) Optimization

Instead of updating all 7B parameters, LoRA only updates ~0.1% of them:

```python
# Original model weights (frozen)
W_original = [7B parameters]  # Not updated

# LoRA adaptation (trained)
W_lora = W_original + (A × B)  # A and B are small matrices
# Only A and B are trained (few million parameters)
```

## Why This Works for Our Task

### The Model Learns Associations

Through next-token prediction on our examples, the model learns:

```
"earnings beat" in context → predict "better performance"
"missed expectations" in context → predict "worse performance"
"convertible debt" in context → predict "negative reaction"
```

### Token-Level Learning Examples

**Example 1: Learning Positive Signals**
```
Input: "revenue exceeded guidance by 10%"
Next tokens learned: "strong" → "positive" → "performance"
```

**Example 2: Learning Comparison**
```
Input: "Filing A has earnings beat, Filing B has earnings miss"
Next tokens learned: "Filing" → "A" → "better"
```

### Loss Calculation

The training loss is calculated only on the output tokens:

```python
def training_step(batch):
    # Get model predictions for all positions
    logits = model(batch['input_ids'])
    
    # Calculate loss only on output portion
    output_start = batch['instruction_length']
    loss = cross_entropy(
        logits[output_start:],  # Predictions
        batch['labels'][output_start:]  # True next tokens
    )
    
    # Backpropagate and update LoRA weights
    loss.backward()
    optimizer.step()
```

## Training Dynamics for Our Contrastive Task

### Epoch 1: Learn Basic Patterns
- "beat" → positive
- "miss" → negative
- "Filing A" vs "Filing B" structure

### Epoch 2: Learn Nuanced Patterns
- "beat by 10%" → "strong positive"
- "slight miss" → "mild negative"
- Confidence calibration

### Epoch 3: Learn Domain-Specific Patterns
- Event type matters (7.01 vs 2.02)
- Convertible debt → usually negative
- CEO departure → context-dependent

## What the Model Actually Optimizes

The objective function:
```
Loss = -Σ log P(token_i | all_previous_tokens)
```

For our contrastive examples:
```
Loss = -log P("Filing" | context) 
       -log P("A" | context + "Filing")
       -log P("resulted" | context + "Filing A")
       ... (for all output tokens)
```

## Memory Efficiency with Gradient Checkpointing

Since we're training on long sequences (2048 tokens):

```python
# Without gradient checkpointing: Store all 32 layer activations
Memory = 32 layers × 2048 tokens × hidden_dim × batch_size  # ~20GB

# With gradient checkpointing: Recompute during backward pass
Memory = checkpoint_layers × 2048 × hidden_dim × batch_size  # ~5GB
```

## The Final Result

After training, when we prompt:
```
"Compare Filing A (new earnings) vs Filing B (baseline)"
```

The model's next-token predictions will be biased toward:
1. Identifying key signals it learned
2. Predicting the correct filing choice
3. Generating appropriate confidence levels
4. Providing learned reasoning patterns

This is why contrastive learning works well - we're teaching the model to predict specific token sequences that encode our domain knowledge about what makes one filing better than another.