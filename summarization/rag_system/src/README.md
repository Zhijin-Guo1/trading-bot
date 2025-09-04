# RAG System Source Code

## Current Production Files (3 Essential Files)

### 1. `pipeline_with_summarization.py` ⭐ **[LATEST VERSION]**
- **Purpose**: Complete RAG pipeline with hybrid search and LLM summarization
- **Features**:
  - Hybrid search (70% semantic + 30% keyword)
  - TF-IDF for keyword matching
  - LLM summarization to 200-300 words
  - Saves summaries for flexible prediction
- **Run**: `python pipeline_with_summarization.py`

### 2. `complete_pipeline_improved.py`
- **Purpose**: Full pipeline with improved prompts (no hybrid search)
- **Features**:
  - 8 specialized retrieval queries
  - Surprise-focused prompts
  - Direct prediction generation
- **Run**: `python complete_pipeline_improved.py`

### 3. `setup_chromadb_openai.py`
- **Purpose**: Set up ChromaDB with OpenAI embeddings
- **Features**:
  - Creates vector database
  - Uses OpenAI text-embedding-ada-002
  - Cosine similarity search
- **Run**: `python setup_chromadb_openai.py`

## Workflow

```
1. Run pipeline_with_summarization.py
   ├── Loads 3 sample filings
   ├── Chunks into ~44 segments
   ├── Creates ChromaDB + TF-IDF indices
   ├── Hybrid retrieval (semantic + keyword)
   ├── Generates focused summaries
   └── Saves summaries for prediction

2. Use summaries for prediction:
   - Feed to GPT-4/Claude for prediction
   - Create embeddings for ML models
   - Extract features for traditional models
```

## Output Location

All results saved to:
```
/rag_system/data/results/enhanced_pipeline_[timestamp]/
├── chunks.json               # Chunked text
├── summaries_only.json      # Clean summaries
└── summaries_and_retrievals.json # Full details
```

## Archived Files

Old versions moved to `archive_old/` directory for reference.

## Key Improvements in Latest Version

1. **Hybrid Search**: Combines semantic (ChromaDB) + keyword (TF-IDF)
2. **LLM Summarization**: Reduces ~10k chars to ~1k focused summary
3. **Flexible Output**: Summaries ready for any prediction method
4. **Better Scoring**: Shows both semantic and keyword scores