# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an LLM Router Testing Suite that demonstrates and compares two intelligent query routing approaches:
- **Semantic Router**: Ultra-fast routing using vector similarity with OpenAI embeddings
- **RouteLLM**: Cost-optimized routing between different LLM models based on query complexity

## Development Commands

### Environment Setup
```bash
# Activate virtual environment (recommended)
source venv/roenv/bin/activate

# Install dependencies (if not using venv)
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your OpenAI API key
```

### Running Tests/Demos
```bash
# Test Semantic Router (OpenAI)
python src/semantic_router_demo.py

# Test Semantic Router (Azure OpenAI)
python src/semantic_router_azure.py

# Test RouteLLM (simulated)
python src/routellm_demo.py

# Run comparison between both approaches (simulated)
python src/combined_comparison.py

# NEW: Direct OpenAI API Integration
# Test Semantic Router with direct OpenAI API
python src/semantic_router_oai.py

# Test RouteLLM with real OpenAI API calls and cost optimization
python src/routellm_oai.py

# Compare both approaches with real OpenAI API integration
python src/combined_comparison_oai.py

# Simple test without interactive mode
python src/test_semantic_router_oai.py

# Run individual test queries
python src/test_queries.py
```

## Architecture

### Core Components

#### Original Implementations
1. **Semantic Router** (`semantic_router_demo.py`, `semantic_router_azure.py`)
   - Uses OpenAI embeddings for vector similarity routing
   - Routes queries to predefined categories (product_documentation, data_analytics, customer_support, general_conversation)
   - Ultra-low latency (20-50ms) with minimal cost (~$0.02 per 1K queries)

2. **RouteLLM** (`routellm_demo.py`)
   - Routes between strong (GPT-4) and weak (GPT-3.5) models based on query complexity
   - Uses keyword analysis and query length to determine complexity score
   - Higher latency (100-500ms) but adaptive to any query type (simulated responses)

3. **Comparison Suite** (`combined_comparison.py`)
   - Direct performance comparison between both approaches
   - Provides metrics on latency, cost, and routing accuracy
   - Includes recommendations for when to use each approach

#### NEW: Direct OpenAI API Implementations
4. **Enhanced Semantic Router** (`semantic_router_oai.py`)
   - Direct OpenAI API integration with real-time embedding generation
   - 6 comprehensive route categories (technical, analytics, customer_service, content, research, general)
   - Performance tracking with detailed statistics
   - Batch processing capabilities and interactive testing mode
   - ~400-800ms latency (includes OpenAI API calls)

5. **Enhanced RouteLLM** (`routellm_oai.py`)
   - Real OpenAI API calls with actual cost optimization
   - Routes between GPT-4o-mini (strong) and GPT-4o-nano (weak) models
   - Advanced complexity analysis with multiple factors (keywords, length, technical terms)
   - Accurate token counting with tiktoken
   - Real cost tracking and savings analysis
   - ~1-3 second latency (full LLM inference)

6. **Comprehensive Comparison** (`combined_comparison_oai.py`)
   - Side-by-side comparison using real OpenAI API calls
   - Detailed performance metrics, cost analysis, and model usage statistics
   - Interactive comparison mode for testing custom queries
   - Real-world performance benchmarks

7. **Metrics Utilities** (`src/utils/metrics.py`)
   - `PerformanceTimer`: Context manager for timing operations
   - `MetricsCollector`: Comprehensive routing metrics collection and analysis

### Configuration

Environment variables are managed through `.env` file:
- **OpenAI**: `OPENAI_API_KEY`, `OPENAI_ORG_ID`
- **Azure OpenAI**: `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_EMBEDDING_DEPLOYMENT`
- **RouteLLM**: `ROUTELLM_STRONG_MODEL`, `ROUTELLM_WEAK_MODEL`, `ROUTELLM_THRESHOLD`
- **Optional**: `REDIS_URL`, `ENABLE_CACHE`

## Key Design Patterns

### Route Definition Pattern (Semantic Router)
Routes are defined with example utterances for training:
```python
Route(
    name="category_name",
    utterances=[
        "Example query 1",
        "Example query 2",
        # 10-15 diverse examples per route
    ]
)
```

### Complexity Classification (RouteLLM)
Queries are classified by complexity using:
- Keyword analysis (high/medium/low complexity indicators)
- Query length factor
- Threshold-based model selection

### Performance Testing Pattern
All demos follow a consistent pattern:
1. Initialize router/classifier
2. Process test queries with timing
3. Collect metrics (latency, cost, route distribution)
4. Display results with colored output
5. Provide interactive mode for custom queries

## Dependencies

Core routing libraries:
- `semantic-router==0.0.20`
- `routellm==0.0.5`
- `openai>=1.12.0`

Vector operations and ML:
- `numpy`, `scikit-learn`, `sentence-transformers`, `torch`

Optional performance optimizations:
- `faiss-cpu` for optimized vector search
- `redis` for caching