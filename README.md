# Plexure API Search

A powerful semantic search engine for API documentation with real-time progress tracking and advanced embedding capabilities.

## Features

### Core Functionality
- Semantic search over API documentation using modern embedding models
- Real-time progress tracking with WebSocket updates
- Advanced vector embedding generation with fallback models
- LLM-powered API documentation enrichment
- Caching system for embeddings and search results
- Support for multiple vector stores (Pinecone)

### Search Capabilities
- Semantic similarity search
- Context-aware query expansion
- Dynamic result ranking and boosting
- Cross-encoder reranking
- Multi-language support
- Fuzzy matching

### Monitoring & Progress
- Real-time progress visualization
- Event-driven architecture
- Detailed progress tracking
- WebSocket-based updates
- Component status monitoring
- Error tracking and reporting

### Model Features
- Multiple embedding models support
- Automatic model fallback
- Configurable model settings
- Hugging Face integration
- Cross-encoder reranking
- Multi-language support

### Performance
- Efficient vector operations
- Caching at multiple levels
- Batch processing support
- Optimized embedding generation
- Configurable performance settings

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/plexure-api-search.git
cd plexure-api-search
```

2. Install dependencies using Poetry:
```bash
poetry install
```

3. Copy the example environment file and configure your settings:
```bash
cp .env.sample .env
```

4. Configure the required environment variables in `.env`:
```env
# API Directory
API_DIR=assets/apis

# Vector Settings
VECTOR_DIMENSION=384

# Pinecone Settings
PINECONE_API_KEY=your_api_key
PINECONE_ENVIRONMENT=your_environment
PINECONE_INDEX_NAME=your_index_name

# Model Settings
HUGGINGFACE_TOKEN=your_token
BI_ENCODER_MODEL=sentence-transformers/all-MiniLM-L6-v2
CROSS_ENCODER_MODEL=sentence-transformers/all-mpnet-base-v2

# OpenRouter Settings (Optional)
OPENROUTER_API_KEY=your_api_key
```

## Usage

### Indexing API Documentation

Index your API documentation:
```bash
poetry run python -m plexure_api_search index
```

Force reindexing of all documents:
```bash
poetry run python -m plexure_api_search index --force
```

### Running Searches

Search API documentation:
```bash
poetry run python -m plexure_api_search search "your query here"
```

### Monitoring Progress

1. Start the web server:
```bash
poetry run python -m plexure_api_search serve
```

2. Open your browser and navigate to:
```
http://localhost:8000/progress
```

## Configuration

The system can be configured through environment variables in the `.env` file:

### Vector Store Settings
- `VECTOR_DIMENSION`: Dimension of embedding vectors
- `FAISS_INDEX_TYPE`: Type of FAISS index
- `FAISS_NLIST`: Number of clusters for IVF indices

### Model Settings
- `BI_ENCODER_MODEL`: Primary bi-encoder model
- `BI_ENCODER_FALLBACK`: Fallback bi-encoder model
- `CROSS_ENCODER_MODEL`: Model for reranking
- `MULTILINGUAL_MODEL`: Model for multi-language support

### Cache Settings
- `CACHE_TTL`: Cache time-to-live in seconds
- `EMBEDDING_CACHE_TTL`: Embedding cache TTL
- `REDIS_CACHE_ENABLED`: Enable Redis caching
- `REDIS_ENABLED`: Enable Redis for other features

### Performance Settings
- `NORMALIZE_EMBEDDINGS`: Whether to normalize vectors
- `POOLING_STRATEGY`: Token pooling strategy
- `MAX_SEQ_LENGTH`: Maximum sequence length

## Development

### Project Structure
- `plexure_api_search/`: Main package directory
  - `embedding/`: Embedding generation
  - `indexing/`: API indexing
  - `search/`: Search functionality
  - `monitoring/`: Progress tracking
  - `web/`: Web interface
  - `config.py`: Configuration management
  - `cli.py`: Command-line interface

### Running Tests
```bash
poetry run pytest
```

### Code Style
The project follows PEP 8 guidelines. Format code using:
```bash
poetry run black .
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request
