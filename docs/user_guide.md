# User Guide

## Overview

Plexure API Search is a tool for searching and exploring API endpoints using natural language queries. This guide will help you get started with using the tool effectively.

## Installation

### Prerequisites

1. Python 3.9 or higher
2. Poetry package manager
3. Git (for source installation)
4. FAISS with AVX2 support

### Quick Install

1. Install using pip:

```bash
pip install plexure-api-search
```

2. Install using Poetry:

```bash
git clone https://github.com/yourusername/plexure-api-search.git
cd plexure-api-search
poetry install
```

### Configuration

1. Create environment file:

```bash
cp .env.sample .env
```

2. Configure settings:

```bash
# Required settings
ENVIRONMENT=development
API_DIR=assets/apis
CACHE_DIR=.cache/default
MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2

# Optional settings
DEBUG=false
LOG_LEVEL=INFO
MODEL_BATCH_SIZE=32
```

## Basic Usage

### Searching Endpoints

1. Simple search:

```bash
poetry run python -m plexure_api_search search "find authentication endpoints"
```

2. Advanced search:

```bash
poetry run python -m plexure_api_search search \
    --query "find user management endpoints" \
    --limit 5 \
    --min-score 0.5
```

### Managing API Contracts

1. Index contracts:

```bash
poetry run python -m plexure_api_search index --clear
```

2. Update index:

```bash
poetry run python -m plexure_api_search index --update
```

3. Check index status:

```bash
poetry run python -m plexure_api_search index --status
```

## Advanced Features

### Query Optimization

1. Query expansion:

```bash
export EXPAND_QUERY=true
poetry run python -m plexure_api_search search "auth"
```

2. Result reranking:

```bash
export RERANK_RESULTS=true
poetry run python -m plexure_api_search search "users"
```

### Performance Tuning

1. Batch processing:

```bash
export MODEL_BATCH_SIZE=64
poetry run python -m plexure_api_search index
```

2. Cache management:

```bash
export CACHE_DIR=.cache/production
poetry run python -m plexure_api_search search "endpoints"
```

## Common Tasks

### Finding Endpoints

1. Authentication endpoints:

```bash
poetry run python -m plexure_api_search search "authentication and authorization"
```

2. User management:

```bash
poetry run python -m plexure_api_search search "user creation and management"
```

3. Data operations:

```bash
poetry run python -m plexure_api_search search "data retrieval and updates"
```

### Managing Contracts

1. Add new contract:

```bash
cp new-api.yaml assets/apis/
poetry run python -m plexure_api_search index --update
```

2. Remove contract:

```bash
rm assets/apis/old-api.yaml
poetry run python -m plexure_api_search index --clear
```

3. Update contract:

```bash
cp updated-api.yaml assets/apis/
poetry run python -m plexure_api_search index --update
```

## Best Practices

### Effective Searching

1. Use descriptive queries:

- Good: "find user authentication endpoints"
- Bad: "auth"

2. Include context:

- Good: "create new user account with email"
- Bad: "create user"

3. Be specific:

- Good: "update user profile information"
- Bad: "user update"

### Contract Management

1. Organize contracts:

```
assets/apis/
├── auth/
│   ├── authentication.yaml
│   └── authorization.yaml
├── users/
│   ├── profiles.yaml
│   └── management.yaml
└── data/
    ├── queries.yaml
    └── mutations.yaml
```

2. Use consistent formatting:

```yaml
paths:
  /users:
    post:
      summary: Create new user
      description: Detailed description of the endpoint
      parameters:
        - name: email
          description: User's email address
```

3. Include metadata:

```yaml
info:
  title: User Management API
  version: 1.0.0
  description: API for managing user accounts
```

## Troubleshooting

### Common Issues

1. No results found:

- Check if contracts are indexed
- Verify query format
- Lower similarity threshold

```bash
export MIN_SCORE=0.1
```

2. Poor search results:

- Use more specific queries
- Enable query expansion
- Check contract descriptions

```bash
export EXPAND_QUERY=true
```

3. Indexing fails:

- Check file permissions
- Verify contract format
- Monitor storage space

```bash
df -h
du -sh .cache/
```

### Error Messages

1. "No contracts found":

```bash
# Check API directory
ls -la assets/apis/

# Verify contract format
poetry run python -m plexure_api_search validate
```

2. "Model loading failed":

```bash
# Clear model cache
rm -rf .models/*

# Verify model name
echo $MODEL_NAME
```

3. "Search failed":

```bash
# Check logs
tail -f logs/app.log

# Verify configuration
poetry run python -m plexure_api_search config show
```

## Configuration Reference

### Environment Variables

| Variable           | Description             | Default                                | Required |
| ------------------ | ----------------------- | -------------------------------------- | -------- |
| `ENVIRONMENT`      | Environment name        | development                            | Yes      |
| `API_DIR`          | API contracts directory | assets/apis                            | Yes      |
| `CACHE_DIR`        | Cache directory         | .cache/default                         | Yes      |
| `MODEL_NAME`       | Embedding model         | sentence-transformers/all-MiniLM-L6-v2 | Yes      |
| `DEBUG`            | Debug mode              | false                                  | No       |
| `LOG_LEVEL`        | Logging level           | INFO                                   | No       |
| `MODEL_BATCH_SIZE` | Batch size              | 32                                     | No       |
| `MIN_SCORE`        | Minimum score           | 0.1                                    | No       |
| `TOP_K`            | Result limit            | 10                                     | No       |

### Command Line Options

1. Search options:

```bash
poetry run python -m plexure_api_search search --help
```

2. Index options:

```bash
poetry run python -m plexure_api_search index --help
```

3. Config options:

```bash
poetry run python -m plexure_api_search config --help
```

## Examples

### Basic Examples

1. Simple search:

```bash
poetry run python -m plexure_api_search search "login endpoints"
```

2. Index contracts:

```bash
poetry run python -m plexure_api_search index
```

3. Show config:

```bash
poetry run python -m plexure_api_search config show
```

### Advanced Examples

1. Complex search:

```bash
poetry run python -m plexure_api_search search \
    "find endpoints for managing user preferences" \
    --limit 10 \
    --min-score 0.3 \
    --expand-query
```

2. Selective indexing:

```bash
poetry run python -m plexure_api_search index \
    --include "auth/*.yaml" \
    --exclude "internal/*.yaml"
```

3. Performance monitoring:

```bash
poetry run python -m plexure_api_search search \
    "authentication" \
    --profile
```

## Support

### Getting Help

1. Documentation:

- Read the docs
- API reference
- Examples

2. Community:

- GitHub issues
- Discussions
- Stack Overflow

### Reporting Issues

1. Bug reports:

- Clear description
- Steps to reproduce
- System information
- Error messages

2. Feature requests:

- Use case
- Expected behavior
- Example usage
