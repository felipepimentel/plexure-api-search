# Plexure API Search

A powerful semantic search engine for API endpoints, designed to help developers quickly find and understand API endpoints across multiple OpenAPI specifications.

## Features

### Core Features
- 🔍 Semantic Search: Find API endpoints using natural language queries
- 📚 Multi-API Support: Index and search across multiple OpenAPI specifications
- 🎯 Accurate Results: Advanced vector-based search with relevance scoring
- 🔄 Real-time Monitoring: Live monitoring of indexing and search operations
- 🚀 High Performance: Optimized for speed with caching and parallel processing

### Search Capabilities
- Natural language queries
- Method-based filtering (GET, POST, PUT, DELETE, PATCH)
- Score-based relevance ranking
- Query expansion and enhancement
- Multi-language support
- Fallback model support

### Monitoring
- Real-time event tracking
- Process-based event grouping
- Interactive TUI (Terminal User Interface)
- Event history with pagination
- Cross-process communication via ZeroMQ
- Log-only mode for debugging

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

3. Configure environment variables:
```bash
cp .env.sample .env
# Edit .env with your settings
```

## Usage

### Indexing APIs
```bash
poetry run python -m plexure_api_search index
```

### Searching Endpoints
```bash
poetry run python -m plexure_api_search search "find user by id"
```

### Monitoring Operations
```bash
# With TUI
poetry run python -m plexure_api_search monitor

# Log-only mode
poetry run python -m plexure_api_search monitor --no-tui
```

## Configuration

### Environment Variables
- `API_DIR`: Directory containing API specifications (default: assets/apis)
- `VECTOR_DIMENSION`: Embedding vector dimension (default: 384)
- `PINECONE_*`: Pinecone vector database settings
- `BI_ENCODER_MODEL`: Model for semantic encoding
- `CROSS_ENCODER_MODEL`: Model for relevance scoring
- `MULTILINGUAL_MODEL`: Model for multi-language support
- `CACHE_TTL`: Cache time-to-live in seconds
- `LOG_LEVEL`: Logging verbosity level

### Supported API Formats
- OpenAPI 3.0+ (YAML/JSON)
- Swagger 2.0 (YAML/JSON)

## Project Structure

```
plexure_api_search/
├── cli/                    # Command-line interface
│   ├── commands/          # CLI commands
│   └── ui/               # TUI components
├── monitoring/            # Event monitoring system
├── search/               # Search engine core
├── indexing/             # API indexing and processing
├── embedding/            # Vector embedding generation
└── utils/               # Utility functions
```

## Development

### Running Tests
```bash
poetry run pytest
```

### Code Style
```bash
poetry run black .
poetry run isort .
poetry run flake8
```

### Hot Reload (Development Mode)
```bash
poetry run python -m plexure_api_search monitor --hot-reload
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Built with [Poetry](https://python-poetry.org/)
- Vector search powered by [Pinecone](https://www.pinecone.io/)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- TUI powered by [Rich](https://rich.readthedocs.io/)
- IPC using [ZeroMQ](https://zeromq.org/)
