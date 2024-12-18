# Developer Guide

## Architecture Overview

### Core Components

1. Search Engine
- Semantic search using SentenceTransformers
- Hybrid search combining vector and keyword search
- Query understanding and expansion
- Result ranking and reranking

2. Indexing System
- API contract parsing
- Vector generation
- Incremental indexing
- Transaction handling

3. Vector Store
- Pinecone integration
- Connection pooling
- Batch operations
- Sharding support

4. Service Layer
- Dependency injection
- Event-driven architecture
- Circuit breaker pattern
- Rate limiting

### Directory Structure

```
plexure_api_search/
├── cli/                    # Command-line interface
│   ├── commands/          # CLI commands
│   └── ui/                # TUI components
├── config/                # Configuration management
├── di/                    # Dependency injection
├── embedding/             # Vector embeddings
├── indexing/              # API indexing
├── integrations/          # External integrations
├── monitoring/            # System monitoring
├── plugins/               # Plugin system
├── search/                # Search functionality
├── services/              # Service layer
└── utils/                 # Utility functions
```

## Development Setup

1. Clone repository:
```bash
git clone https://github.com/plexure/plexure-api-search.git
cd plexure-api-search
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Install dependencies:
```bash
pip install -e ".[dev]"
```

4. Set up pre-commit hooks:
```bash
pre-commit install
```

## Development Workflow

### Code Style

1. Follow PEP 8 guidelines
2. Use type hints
3. Maximum line length: 100 characters
4. Use docstrings for all public functions/classes
5. Keep functions focused and single-purpose

Example:
```python
from typing import List, Optional

def process_query(
    query: str,
    filters: Optional[dict] = None,
    limit: int = 10,
) -> List[dict]:
    """Process search query with filters.

    Args:
        query: Search query string
        filters: Optional search filters
        limit: Maximum number of results

    Returns:
        List of search results
    """
    # Implementation
```

### Testing

1. Write unit tests:
```python
def test_process_query():
    # Arrange
    query = "test query"
    filters = {"method": "GET"}
    
    # Act
    results = process_query(query, filters)
    
    # Assert
    assert len(results) <= 10
    assert all(r["method"] == "GET" for r in results)
```

2. Run tests:
```bash
pytest tests/
pytest tests/ -v                 # Verbose
pytest tests/ -k test_search     # Filter tests
pytest tests/ --cov=plexure_api_search  # Coverage
```

3. Test coverage requirements:
- Minimum coverage: 80%
- Critical paths: 100%
- Edge cases covered

### Documentation

1. Code Documentation
- Clear docstrings
- Type hints
- Inline comments for complex logic
- Examples in docstrings

2. API Documentation
- OpenAPI/Swagger specs
- Example requests/responses
- Error handling
- Authentication

3. Architecture Documentation
- Component diagrams
- Sequence diagrams
- Data flow diagrams
- Decision records

### Dependency Management

1. Adding dependencies:
```bash
poetry add package-name
poetry add package-name --dev  # Dev dependency
```

2. Updating dependencies:
```bash
poetry update
poetry update package-name
```

3. Dependency guidelines:
- Pin versions
- Regular updates
- Security checks
- Minimize dependencies

### Version Control

1. Branch naming:
- feature/description
- fix/description
- refactor/description
- docs/description

2. Commit messages:
```
type(scope): description

- type: feat, fix, docs, style, refactor, test, chore
- scope: component affected
- description: clear, concise change description
```

3. Pull requests:
- Clear title and description
- Link related issues
- Tests included
- Documentation updated

### Monitoring and Profiling

1. Performance monitoring:
```python
from plexure_api_search.monitoring import profiler

@profiler.profile
async def search_endpoint(query: str) -> dict:
    # Implementation
```

2. Metrics collection:
```python
from plexure_api_search.monitoring import metrics

metrics.increment("search_requests")
metrics.timing("search_latency", duration)
```

3. Error tracking:
```python
from plexure_api_search.monitoring import logger

try:
    result = await process_query(query)
except Exception as e:
    logger.error(f"Search failed: {e}", exc_info=True)
    raise
```

### Deployment

1. Build package:
```bash
poetry build
```

2. Run checks:
```bash
poetry run pytest
poetry run mypy .
poetry run black --check .
poetry run isort --check .
```

3. Deploy steps:
- Version bump
- Changelog update
- Tag release
- Build package
- Run checks
- Deploy package
- Verify deployment

## Component Details

### Search Engine

1. Query Processing
```python
async def process_query(query: str) -> str:
    # Spell checking
    query = await spell_checker.check(query)
    
    # Query expansion
    query = await query_expander.expand(query)
    
    # Intent detection
    intent = await intent_detector.detect(query)
    
    return query
```

2. Vector Search
```python
async def vector_search(
    query: str,
    limit: int = 10,
) -> List[dict]:
    # Generate query vector
    vector = await embedder.encode(query)
    
    # Search vector store
    results = await vector_store.search(
        vector=vector,
        limit=limit,
    )
    
    return results
```

3. Result Ranking
```python
async def rank_results(
    results: List[dict],
    query: str,
) -> List[dict]:
    # Calculate relevance scores
    scores = await ranker.score(results, query)
    
    # Sort by score
    ranked = sorted(
        results,
        key=lambda x: scores[x["id"]],
        reverse=True,
    )
    
    return ranked
```

### Indexing System

1. Contract Parsing
```python
async def parse_contract(path: str) -> dict:
    # Read file
    content = await file_utils.read_file(path)
    
    # Parse format
    if path.endswith(".yaml"):
        contract = yaml.safe_load(content)
    else:
        contract = json.loads(content)
    
    # Validate schema
    validator.validate(contract)
    
    return contract
```

2. Vector Generation
```python
async def generate_vectors(
    contracts: List[dict],
) -> List[np.ndarray]:
    # Extract text
    texts = [
        extract_text(contract)
        for contract in contracts
    ]
    
    # Generate embeddings
    vectors = await embedder.encode_batch(texts)
    
    return vectors
```

3. Index Management
```python
async def update_index(
    contracts: List[dict],
    vectors: List[np.ndarray],
) -> None:
    # Start transaction
    async with vector_store.transaction():
        # Delete old vectors
        await vector_store.delete([
            contract["id"]
            for contract in contracts
        ])
        
        # Insert new vectors
        await vector_store.insert(
            ids=[contract["id"] for contract in contracts],
            vectors=vectors,
            metadata=contracts,
        )
```

### Service Layer

1. Dependency Injection
```python
from dependency_injector import containers, providers

class Container(containers.DeclarativeContainer):
    config = providers.Singleton(Config)
    
    vector_store = providers.Singleton(
        VectorStore,
        config=config,
    )
    
    search_service = providers.Singleton(
        SearchService,
        vector_store=vector_store,
        config=config,
    )
```

2. Event Publishing
```python
async def publish_event(event: Event) -> None:
    # Serialize event
    message = event.to_json()
    
    # Publish to topics
    await publisher.publish(
        topic="events",
        message=message,
    )
```

3. Circuit Breaker
```python
async def execute_with_breaker(
    command: Callable,
    *args,
    **kwargs,
) -> Any:
    # Check circuit state
    if breaker.is_open:
        raise CircuitBreakerError()
    
    try:
        # Execute command
        result = await command(*args, **kwargs)
        breaker.success()
        return result
    except Exception as e:
        breaker.failure()
        raise
```

## Plugin Development

1. Create plugin:
```python
from plexure_api_search.plugins import SearchPlugin

class CustomPlugin(SearchPlugin):
    def __init__(self, config: Config):
        self.config = config
    
    async def process_query(
        self,
        query: str,
    ) -> str:
        # Custom processing
        return query
```

2. Register plugin:
```python
# plugin.py
from plexure_api_search.plugins import registry

registry.register(CustomPlugin)
```

3. Use plugin:
```python
# Load plugins
plugins = registry.load_plugins()

# Process query
for plugin in plugins:
    query = await plugin.process_query(query)
```

## Performance Optimization

1. Caching
```python
from plexure_api_search.cache import cache

@cache(ttl=3600)
async def expensive_operation(
    key: str,
) -> dict:
    # Expensive computation
    return result
```

2. Batch Processing
```python
async def process_batch(
    items: List[dict],
    batch_size: int = 100,
) -> List[dict]:
    results = []
    for i in range(0, len(items), batch_size):
        batch = items[i:i + batch_size]
        batch_results = await process_items(batch)
        results.extend(batch_results)
    return results
```

3. Connection Pooling
```python
async def get_connection():
    async with pool.acquire() as conn:
        return conn
```

## Error Handling

1. Custom Exceptions
```python
class SearchError(Exception):
    """Base class for search errors."""
    pass

class QueryError(SearchError):
    """Invalid query error."""
    pass

class IndexError(SearchError):
    """Index operation error."""
    pass
```

2. Error Recovery
```python
async def with_retry(
    operation: Callable,
    max_retries: int = 3,
    delay: float = 1.0,
) -> Any:
    for attempt in range(max_retries):
        try:
            return await operation()
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            await asyncio.sleep(delay * (attempt + 1))
```

3. Graceful Degradation
```python
async def search_with_fallback(
    query: str,
) -> List[dict]:
    try:
        # Try vector search
        results = await vector_search(query)
    except Exception as e:
        # Fall back to keyword search
        results = await keyword_search(query)
    return results
```

## Security

1. Input Validation
```python
from pydantic import BaseModel, validator

class SearchRequest(BaseModel):
    query: str
    filters: Optional[dict] = None
    
    @validator("query")
    def validate_query(cls, v):
        if len(v) > 1000:
            raise ValueError("Query too long")
        return v
```

2. Rate Limiting
```python
from plexure_api_search.security import rate_limit

@rate_limit(
    requests=100,
    window=60,
)
async def search_endpoint(
    request: Request,
) -> Response:
    # Handle request
```

3. Authentication
```python
from plexure_api_search.security import require_auth

@require_auth
async def protected_endpoint(
    request: Request,
) -> Response:
    # Handle request
```

## Monitoring

1. Health Checks
```python
async def check_health() -> dict:
    return {
        "status": "healthy",
        "components": {
            "search": await search_health(),
            "index": await index_health(),
            "store": await store_health(),
        },
    }
```

2. Metrics Collection
```python
async def collect_metrics() -> dict:
    return {
        "search": {
            "requests": counter.get("search_requests"),
            "latency": histogram.get("search_latency"),
            "errors": counter.get("search_errors"),
        },
    }
```

3. Alerting
```python
async def check_alerts() -> None:
    metrics = await collect_metrics()
    if metrics["errors"] > threshold:
        await send_alert(
            level="error",
            message="High error rate detected",
            metrics=metrics,
        )
```

## Contributing

1. Issue Guidelines
- Use issue templates
- Clear reproduction steps
- System information
- Logs/error messages

2. Pull Request Guidelines
- Reference issues
- Clean commit history
- Tests included
- Documentation updated

3. Review Guidelines
- Code quality
- Test coverage
- Performance impact
- Security implications

## Resources

1. Documentation
- [API Documentation](docs/api.md)
- [User Guide](docs/user_guide.md)
- [Architecture](docs/architecture.md)

2. Examples
- [Basic Usage](examples/basic.py)
- [Advanced Features](examples/advanced.py)
- [Plugin Development](examples/plugin.py)

3. Tools
- [Development Tools](tools/README.md)
- [Benchmarking](tools/benchmark.py)
- [Migration Scripts](tools/migrate.py) 