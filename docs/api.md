# API Documentation

## Overview

The Plexure API Search provides a command-line interface and a set of Python modules for searching and managing API endpoints. This document describes the available commands, modules, and their usage.

## Command Line Interface

### Search Command

Search for API endpoints using natural language queries.

```bash
poetry run python -m plexure_api_search search [OPTIONS] QUERY
```

#### Options:

- `--limit INTEGER`: Maximum number of results (default: 10)
- `--min-score FLOAT`: Minimum similarity score (default: 0.1)
- `--expand-query`: Enable query expansion
- `--rerank`: Enable result reranking
- `--format [text|json|yaml]`: Output format (default: text)
- `--output FILE`: Output file (default: stdout)
- `--profile`: Enable performance profiling

#### Examples:

```bash
# Basic search
poetry run python -m plexure_api_search search "authentication endpoints"

# Advanced search with options
poetry run python -m plexure_api_search search \
    "user management" \
    --limit 5 \
    --min-score 0.3 \
    --expand-query \
    --format json

# Search with output file
poetry run python -m plexure_api_search search \
    "data operations" \
    --output results.json
```

### Index Command

Manage the API endpoint index.

```bash
poetry run python -m plexure_api_search index [OPTIONS]
```

#### Options:

- `--clear`: Clear existing index
- `--update`: Update existing index
- `--status`: Show index status
- `--include PATTERN`: Include file pattern
- `--exclude PATTERN`: Exclude file pattern
- `--batch-size INTEGER`: Processing batch size
- `--async`: Enable async processing

#### Examples:

```bash
# Clear and rebuild index
poetry run python -m plexure_api_search index --clear

# Update existing index
poetry run python -m plexure_api_search index --update

# Show index status
poetry run python -m plexure_api_search index --status

# Selective indexing
poetry run python -m plexure_api_search index \
    --include "auth/*.yaml" \
    --exclude "internal/*.yaml"
```

### Config Command

Manage configuration settings.

```bash
poetry run python -m plexure_api_search config [OPTIONS] COMMAND
```

#### Commands:

- `show`: Show current configuration
- `validate`: Validate configuration
- `set KEY VALUE`: Set configuration value
- `get KEY`: Get configuration value
- `list`: List all settings

#### Examples:

```bash
# Show configuration
poetry run python -m plexure_api_search config show

# Validate configuration
poetry run python -m plexure_api_search config validate

# Set configuration value
poetry run python -m plexure_api_search config set MODEL_BATCH_SIZE 64

# Get configuration value
poetry run python -m plexure_api_search config get MODEL_NAME
```

## Python API

### Search Module

```python
from plexure_api_search.search import Searcher

# Initialize searcher
searcher = Searcher()

# Simple search
results = searcher.search("authentication endpoints")

# Advanced search
results = searcher.search(
    query="user management",
    limit=5,
    min_score=0.3,
    expand_query=True,
    rerank=True
)

# Process results
for result in results:
    print(f"Endpoint: {result.endpoint}")
    print(f"Score: {result.score}")
    print(f"Metadata: {result.metadata}")
```

### Index Module

```python
from plexure_api_search.indexing import Indexer

# Initialize indexer
indexer = Indexer()

# Clear and rebuild index
indexer.clear()
indexer.index_directory("assets/apis")

# Update index
indexer.update()

# Get index status
status = indexer.status()
print(f"Total endpoints: {status.total_endpoints}")
print(f"Index size: {status.index_size}")
```

### Config Module

```python
from plexure_api_search.config import Config

# Get configuration
config = Config()

# Access settings
model_name = config.model_name
batch_size = config.model_batch_size

# Update settings
config.update({
    "model_batch_size": 64,
    "min_score": 0.3
})

# Validate configuration
config.validate()
```

## Data Models

### SearchResult

```python
class SearchResult:
    """Search result model."""

    endpoint: str  # API endpoint path
    score: float  # Similarity score
    metadata: dict  # Additional metadata
    method: str  # HTTP method
    description: str  # Endpoint description
    parameters: List[dict]  # Endpoint parameters
```

### EndpointData

```python
class EndpointData:
    """API endpoint data model."""

    path: str  # Endpoint path
    method: str  # HTTP method
    summary: str  # Short description
    description: str  # Full description
    parameters: List[dict]  # Parameters
    responses: dict  # Response schemas
    metadata: dict  # Additional metadata
```

### IndexStatus

```python
class IndexStatus:
    """Index status model."""

    total_endpoints: int  # Total indexed endpoints
    index_size: int  # Size in bytes
    last_updated: datetime  # Last update time
    metadata_size: int  # Metadata size
    vector_dimension: int  # Vector dimension
```

## Events

### Event Types

1. Search Events:

```python
{
    "type": "search",
    "timestamp": "2024-01-20T10:30:00Z",
    "query": str,
    "results": int,
    "duration_ms": float,
    "success": bool
}
```

2. Index Events:

```python
{
    "type": "index",
    "timestamp": "2024-01-20T10:30:00Z",
    "operation": str,
    "endpoints": int,
    "duration_ms": float,
    "success": bool
}
```

3. Error Events:

```python
{
    "type": "error",
    "timestamp": "2024-01-20T10:30:00Z",
    "component": str,
    "message": str,
    "traceback": str
}
```

## Metrics

### Available Metrics

1. Counter Metrics:

- `embeddings_generated_total`
- `embedding_errors_total`
- `searches_performed_total`
- `search_errors_total`
- `contract_errors_total`

2. Gauge Metrics:

- `index_size`
- `metadata_size`

3. Histogram Metrics:

- `search_latency_seconds`
- `embedding_latency_seconds`

### Accessing Metrics

```python
from plexure_api_search.monitoring import metrics

# Get metric value
total_searches = metrics.get_counter("searches_performed_total")

# Observe value
metrics.observe_value("search_latency_seconds", 0.125)

# Export metrics
metrics_data = metrics.export()
```

## Error Handling

### Exception Types

```python
class SearchError(Exception):
    """Base class for search errors."""
    pass

class IndexError(Exception):
    """Base class for index errors."""
    pass

class ConfigError(Exception):
    """Base class for configuration errors."""
    pass

class ModelError(Exception):
    """Base class for model errors."""
    pass
```

### Error Handling Example

```python
from plexure_api_search.exceptions import SearchError

try:
    results = searcher.search("query")
except SearchError as e:
    logger.error(f"Search failed: {e}")
    # Handle error
```

## Best Practices

### Search Optimization

1. Query Construction:

```python
# Good
results = searcher.search("find user authentication endpoints")

# Bad
results = searcher.search("auth")
```

2. Result Processing:

```python
# Process all results
for result in results:
    if result.score >= 0.5:
        process_high_confidence(result)
    else:
        process_low_confidence(result)
```

### Performance Optimization

1. Batch Processing:

```python
# Process in batches
indexer.index_directory(
    "assets/apis",
    batch_size=64,
    async_processing=True
)
```

2. Caching:

```python
# Enable result caching
searcher = Searcher(
    cache_enabled=True,
    cache_ttl=3600
)
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

1. Bug Reports:

- Clear description
- Reproduction steps
- System information
- Error messages

2. Feature Requests:

- Use case
- Expected behavior
- Example usage
