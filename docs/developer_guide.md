# Developer Guide

## Overview

This guide provides information for developers working on the Plexure API Search project. It covers the project structure, development setup, coding standards, and contribution guidelines.

## Project Structure

```
plexure_api_search/
├── plexure_api_search/     # Main package
│   ├── __init__.py        # Package initialization
│   ├── __main__.py        # CLI entry point
│   ├── config.py          # Configuration management
│   ├── cli.py             # Command-line interface
│   ├── indexing/          # Indexing components
│   │   ├── __init__.py
│   │   ├── indexer.py     # API contract indexing
│   │   └── vectorizer.py  # Vector generation
│   ├── search/            # Search components
│   │   ├── __init__.py
│   │   ├── searcher.py    # Search engine
│   │   └── storage.py     # Data storage
│   ├── services/          # Core services
│   │   ├── __init__.py
│   │   ├── events.py      # Event handling
│   │   ├── models.py      # Model management
│   │   └── vector_store.py # Vector storage
│   └── utils/             # Utility functions
│       ├── __init__.py
│       ├── logging.py     # Logging setup
│       └── metrics.py     # Metrics collection
├── tests/                 # Test suite
│   ├── __init__.py
│   ├── conftest.py       # Test configuration
│   ├── test_indexing.py  # Indexing tests
│   └── test_search.py    # Search tests
├── docs/                 # Documentation
├── assets/              # Static assets
└── scripts/            # Development scripts
```

## Development Setup

### Prerequisites

1. Python 3.9+
2. Poetry
3. Git
4. FAISS with AVX2 support

### Installation

1. Clone repository:

```bash
git clone https://github.com/yourusername/plexure-api-search.git
cd plexure-api-search
```

2. Install dependencies:

```bash
poetry install
```

3. Set up pre-commit hooks:

```bash
poetry run pre-commit install
```

4. Create environment file:

```bash
cp .env.sample .env
```

### Development Environment

1. Activate environment:

```bash
poetry shell
```

2. Run tests:

```bash
pytest
```

3. Run linting:

```bash
flake8
black .
isort .
```

## Code Style

### Python Style Guide

1. Follow PEP 8
2. Maximum line length: 100 characters
3. Use type hints
4. Write docstrings
5. Use meaningful names

### Example Code

```python
from typing import List, Optional

def process_endpoints(
    endpoints: List[str],
    batch_size: int = 32,
    normalize: bool = True
) -> Optional[List[float]]:
    """Process API endpoints and generate vectors.

    Args:
        endpoints: List of endpoint strings
        batch_size: Size of processing batches
        normalize: Whether to normalize vectors

    Returns:
        List of vectors or None if processing fails
    """
    try:
        # Process endpoints
        vectors = []
        for batch in chunks(endpoints, batch_size):
            batch_vectors = generate_vectors(batch)
            if normalize:
                batch_vectors = normalize_vectors(batch_vectors)
            vectors.extend(batch_vectors)
        return vectors
    except Exception as e:
        logger.error(f"Failed to process endpoints: {e}")
        return None
```

### Docstring Format

```python
def function_name(arg1: type1, arg2: type2) -> return_type:
    """Short description.

    Longer description if needed.

    Args:
        arg1: Description of arg1
        arg2: Description of arg2

    Returns:
        Description of return value

    Raises:
        ErrorType: Description of error condition
    """
```

## Testing

### Test Structure

1. Unit Tests:

```python
def test_vector_generation():
    """Test vector generation for endpoints."""
    # Arrange
    endpoints = ["GET /api/v1/users", "POST /api/v1/auth"]

    # Act
    vectors = generate_vectors(endpoints)

    # Assert
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)
```

2. Integration Tests:

```python
def test_search_workflow():
    """Test complete search workflow."""
    # Setup
    index_endpoints(test_contracts)

    # Execute
    results = search("find user endpoints")

    # Verify
    assert len(results) > 0
    assert all(r.score >= 0.5 for r in results)
```

### Running Tests

1. All tests:

```bash
pytest
```

2. Specific tests:

```bash
pytest tests/test_search.py
```

3. With coverage:

```bash
pytest --cov=plexure_api_search
```

## Development Workflow

### Feature Development

1. Create branch:

```bash
git checkout -b feature/new-feature
```

2. Make changes:

- Write tests first
- Implement feature
- Update documentation

3. Run checks:

```bash
# Run tests
pytest

# Check style
flake8
black .
isort .

# Check types
mypy .
```

4. Commit changes:

```bash
git add .
git commit -m "feat: add new feature"
```

### Code Review

1. Pull Request Template:

```markdown
## Description

Brief description of changes

## Changes

- Change 1
- Change 2

## Testing

- [ ] Unit tests
- [ ] Integration tests
- [ ] Manual testing

## Documentation

- [ ] Updated docstrings
- [ ] Updated README
- [ ] Updated guides
```

2. Review Checklist:

- Code style compliance
- Test coverage
- Documentation updates
- Performance impact
- Security considerations

## Debugging

### Logging

1. Configure logging:

```python
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
```

2. Log levels:

```python
logger.debug("Detailed information")
logger.info("General information")
logger.warning("Warning messages")
logger.error("Error messages")
logger.critical("Critical errors")
```

### Debugging Tools

1. Debug configuration:

```python
if settings.DEBUG:
    breakpoint()
```

2. Performance profiling:

```python
import cProfile

def profile_function():
    profiler = cProfile.Profile()
    profiler.enable()
    # Code to profile
    profiler.disable()
    profiler.print_stats()
```

## API Documentation

### OpenAPI Schema

1. Endpoint documentation:

```yaml
paths:
  /search:
    post:
      summary: Search API endpoints
      parameters:
        - name: query
          in: body
          required: true
          schema:
            type: string
      responses:
        200:
          description: Search results
          content:
            application/json:
              schema:
                $ref: "#/components/schemas/SearchResults"
```

2. Model documentation:

```yaml
components:
  schemas:
    SearchResult:
      type: object
      properties:
        endpoint:
          type: string
        score:
          type: number
        metadata:
          type: object
```

## Performance Optimization

### Profiling

1. CPU profiling:

```python
@profile
def expensive_function():
    # Code to profile
    pass
```

2. Memory profiling:

```python
from memory_profiler import profile

@profile
def memory_intensive():
    # Code to profile
    pass
```

### Optimization Tips

1. Vectorization:

```python
# Instead of loops
vectors = []
for text in texts:
    vector = model.encode(text)
    vectors.append(vector)

# Use batch processing
vectors = model.encode(texts, batch_size=32)
```

2. Caching:

```python
from functools import lru_cache

@lru_cache(maxsize=1000)
def expensive_computation(input_data):
    # Expensive operation
    pass
```

## Security

### Best Practices

1. Input validation:

```python
def validate_input(query: str) -> bool:
    """Validate search query."""
    if not query:
        raise ValueError("Empty query")
    if len(query) > 1000:
        raise ValueError("Query too long")
    return True
```

2. Error handling:

```python
def secure_operation():
    try:
        # Sensitive operation
        pass
    except Exception as e:
        logger.error(f"Operation failed: {e}")
        raise SecurityError("Operation failed")
```

### Security Checks

1. Dependency scanning:

```bash
poetry run safety check
```

2. Static analysis:

```bash
poetry run bandit -r .
```

## Contribution Guidelines

### Commit Messages

1. Format:

```
type(scope): description

[optional body]

[optional footer]
```

2. Types:

- feat: New feature
- fix: Bug fix
- docs: Documentation
- style: Formatting
- refactor: Code restructuring
- test: Testing
- chore: Maintenance

### Pull Requests

1. Branch naming:

- feature/description
- fix/description
- docs/description

2. Review process:

- Create PR
- Pass CI checks
- Code review
- Address feedback
- Merge

## Release Process

### Version Management

1. Update version:

```bash
poetry version patch  # or minor/major
```

2. Update changelog:

```markdown
## [1.0.1] - 2024-01-20

### Added

- New feature X

### Fixed

- Bug in Y
```

### Release Steps

1. Create release:

```bash
git tag -a v1.0.1 -m "Release v1.0.1"
git push origin v1.0.1
```

2. Build package:

```bash
poetry build
```

3. Publish:

```bash
poetry publish
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
- Reproduction steps
- System information
- Logs

2. Feature requests:

- Use case
- Expected behavior
- Implementation ideas
