# LLM Router Testing Suite

This repository contains example scripts for testing and comparing Semantic Router and RouteLLM for intelligent query routing.

## Features

- 🚀 **Semantic Router**: Ultra-fast routing using vector similarity
- 💰 **RouteLLM**: Cost-optimized routing between different LLM models
- 📊 **Performance Comparison**: Side-by-side comparison of both approaches
- 🔧 **Azure OpenAI & OpenAI Support**: Works with both providers

## Prerequisites

- Python 3.8+
- Azure OpenAI API access OR OpenAI API access
- ~100MB for model caching

## Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd router-testing
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your API keys
```

## Quick Start

### Test Semantic Router (OpenAI)
```bash
python semantic_router_demo.py
```

### Test Semantic Router (Azure OpenAI)
```bash
python semantic_router_azure.py
```

### Test RouteLLM
```bash
python routellm_demo.py
```

### Run Comparison
```bash
python combined_comparison.py
```

## Configuration

Edit `.env` file with your credentials:
```
# OpenAI
OPENAI_API_KEY=sk-...

# Azure OpenAI
AZURE_OPENAI_API_KEY=...
AZURE_OPENAI_ENDPOINT=https://...openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## Performance Results

Typical results on standard hardware:

| Method | Latency | Cost per 1K queries | Accuracy |
|--------|---------|---------------------|----------|
| Semantic Router | 20-50ms | $0.02 | 92-96% |
| RouteLLM | 100-500ms | $0.10-$2.00 | 95-99% |
| Direct GPT-4 | 1000-3000ms | $10-30 | 99% |

## Architecture

### Semantic Router Flow:
```
Query → Embedding → Cosine Similarity → Route Decision
         (20ms)        (<1ms)            (<1ms)
```

### RouteLLM Flow:
```
Query → Router Model → Model Selection → Response
         (100ms)        (10ms)           (varies)
```

## Troubleshooting

### Common Issues:

1. **Import Error for semantic-router**:
   - Solution: `pip install semantic-router --upgrade`

2. **Azure OpenAI Authentication Failed**:
   - Check endpoint format (should end with `/`)
   - Verify API key and deployment names

3. **RouteLLM Model Download**:
   - First run downloads ~50MB model
   - Ensure stable internet connection

## License

MIT

## Contributing

Pull requests welcome! Please check existing issues first.