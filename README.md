# Plexure API Search

Advanced semantic search engine for API endpoints with enhanced quality features.

## Features

### Core Search
- Semantic search using triple vector embeddings
- Hybrid search combining vector and keyword matching
- Query expansion with semantic similarity
- Intent detection and query understanding
- Template-based query matching
- Spell checking and correction
- Query reformulation with multiple strategies
- Distributed search with load balancing
- Sharding for large indices

### Search Quality
- Cross-encoder reranking for improved relevance
- Result clustering for better organization
- Result diversification within clusters
- Fuzzy matching for endpoints
- Category-based result enhancement
- Business context boosting
- Personalized ranking based on user preferences
- Context-aware boosting based on business rules
- Feedback loop for continuous improvement
- Semantic filtering with multiple strategies

### Performance
- Connection pooling for Pinecone
- Batch processing for embeddings
- Optimized vector operations
- Caching layer for search results
- Async processing for non-blocking operations
- Efficient search pipeline
- Distributed search across multiple nodes
- Load balancing with multiple strategies
- Health monitoring and failover
- Concurrent request handling
- Sharding with consistent hashing
- Automatic rebalancing
- Replica management

### Monitoring
- Comprehensive event logging
- Performance metrics tracking
- Search quality metrics
- Real-time monitoring dashboard
- Health checks for all services

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

3. Install the spaCy language model:
```bash
python -m spacy download en_core_web_sm
```

4. Create a `.env` file with your configuration:
```bash
cp .env.sample .env
# Edit .env with your settings
```

## Usage

### Basic Search
```python
from plexure_api_search.search import APISearcher

# Create searcher instance
searcher = APISearcher(top_k=10)

# Perform search
results = await searcher.search(
    query="create user with email",
    user_id="user123",  # Optional: Enable personalization
    domain="users",     # Optional: Enable context boosting
)

# Print results
for result in results:
    print(f"{result['method']} {result['endpoint']} - Score: {result['score']}")
    print(f"Cluster: {result['cluster']['label']} - {result['cluster']['description']}")
```

### Advanced Features

#### Template Matching
```python
# Search with template matching
results = await searcher.search("list orders by status")
# Returns endpoints matching the list template pattern
```

#### Spell Checking
```python
# Search with automatic spell correction
results = await searcher.search("creat usre")
# Automatically corrects to "create user"
```

#### Result Clustering
```python
# Get clustered results
results = await searcher.search("user management")
# Returns results organized by clusters (e.g., CRUD operations, authentication, etc.)
```

#### Result Diversification
```python
# Get diverse results within clusters
results = await searcher.search("product api")
# Returns diverse endpoints avoiding similar ones
```

#### Personalization
```python
# Update user preferences
searcher.update_user_profile(
    user_id="user123",
    preferences={
        "users": 0.8,
        "authentication": 0.6,
        "products": 0.4,
    },
    tags={"auth", "admin", "api"},
)

# Search with personalization
results = await searcher.search(
    query="user management",
    user_id="user123",
)
```

#### Business Context
```python
# Update business context
searcher.update_business_context(
    domain="users",
    priority=0.8,
    rules={
        "authentication": 0.9,
        "authorization": 0.8,
        "validation": 0.7,
    },
    features={
        "mfa": True,
        "rate_limit": 100,
    },
    constraints={
        "max_users": 1000,
        "restricted": False,
    },
)

# Search with context boosting
results = await searcher.search(
    query="user authentication",
    domain="users",
)
```

#### Feedback Loop
```python
# Add feedback for a result
searcher.add_feedback(
    user_id="user123",
    query="create user",
    result_id="POST_/users",
    feedback_type="click",  # click, relevant, irrelevant, bookmark, report
    metadata={
        "session_id": "abc123",
        "time_spent": 30,
        "converted": True,
    },
)

# Search with feedback-based ranking
results = await searcher.search("user management")
# Results are automatically ranked based on feedback
```

#### Semantic Filtering
```python
# Create semantic filters
filters = [
    # Filter by semantic similarity
    searcher.create_semantic_filter(
        type="semantic",
        conditions=[
            {
                "field": "description",
                "operator": "similar",
                "value": "authentication and authorization",
                "threshold": 0.7,
            }
        ],
    ),
    # Filter by category
    searcher.create_semantic_filter(
        type="category",
        conditions=[
            {
                "field": "category",
                "operator": "contains",
                "value": "security",
            }
        ],
    ),
    # Filter by attributes
    searcher.create_semantic_filter(
        type="attribute",
        conditions=[
            {
                "field": "method",
                "operator": "eq",
                "value": "POST",
            },
            {
                "field": "endpoint",
                "operator": "contains",
                "value": "auth",
            }
        ],
        operator="and",  # Both conditions must match
    ),
    # Filter by constraints
    searcher.create_semantic_filter(
        type="constraint",
        conditions=[
            {
                "field": "rate_limit",
                "operator": "lt",
                "value": 100,
            }
        ],
    ),
    # Composite filter
    searcher.create_semantic_filter(
        type="composite",
        conditions=[
            {
                "field": "semantic",
                "operator": "similar",
                "value": "user management",
                "threshold": 0.7,
            },
            {
                "field": "category",
                "operator": "contains",
                "value": "users",
            }
        ],
        operator="or",  # Either condition can match
    ),
]

# Search with semantic filters
results = await searcher.search(
    query="secure user authentication",
    semantic_filters=filters,
)
```

#### Query Reformulation
```python
# Search with query reformulation
results = await searcher.search(
    query="create usre acount",  # Misspelled query
    reformulation_types=[
        "spelling",      # Fix spelling errors
        "template",      # Match query templates
        "intent",        # Detect query intent
        "expansion",     # Expand with similar terms
    ],
    min_confidence=0.5,  # Minimum confidence for reformulations
)

# Results will include matches for:
# - "create user account" (spelling correction)
# - "create new user" (template matching)
# - "register user" (intent detection)
# - "create user profile" (query expansion)
```

#### Distributed Search
```python
# Search with distributed nodes
results = await searcher.search(
    query="user authentication",
    use_distributed=True,  # Enable distributed search
)

# Configure distributed search
distributed_config = {
    "min_nodes": 2,        # Minimum number of nodes
    "max_nodes": 10,       # Maximum number of nodes
    "min_healthy": 0.5,    # Minimum healthy node ratio
    "strategy": "round_robin",  # Load balancing strategy
    "timeout": 10.0,       # Request timeout in seconds
    "retry_count": 3,      # Number of retries
    "retry_delay": 1.0,    # Delay between retries
    "health_interval": 60.0,  # Health check interval
}

# Configure node
node_config = {
    "node_id": "node_1",
    "host": "localhost",
    "port": 8001,
    "weight": 1.0,         # Node weight for load balancing
    "max_concurrent": 10,  # Maximum concurrent requests
    "timeout": 5.0,       # Node timeout in seconds
    "retry_count": 3,     # Node retry count
    "retry_delay": 1.0,   # Node retry delay
    "health_interval": 60.0,  # Node health check interval
}

# Create distributed searcher
distributed_searcher = APISearcher(
    distributed_config=distributed_config,
    node_configs=[node_config],
)

# Search with custom configuration
results = await distributed_searcher.search(
    query="user authentication",
    use_distributed=True,
)
```

#### Sharding
```python
# Search with sharded index
results = await searcher.search(
    query="user authentication",
    use_sharding=True,  # Enable sharding
)

# Configure sharding
sharding_config = {
    "min_shards": 2,        # Minimum number of shards
    "max_shards": 10,       # Maximum number of shards
    "min_healthy": 0.5,     # Minimum healthy shard ratio
    "strategy": "consistent_hashing",  # Sharding strategy
    "timeout": 10.0,        # Request timeout in seconds
    "retry_count": 3,       # Number of retries
    "retry_delay": 1.0,     # Delay between retries
    "health_interval": 60.0,  # Health check interval
    "rebalance_threshold": 0.2,  # Rebalance when imbalance > 20%
    "rebalance_interval": 3600.0,  # Rebalance every hour
}

# Configure shard
shard_config = {
    "shard_id": "shard_1",
    "host": "localhost",
    "port": 8001,
    "weight": 1.0,          # Shard weight for load balancing
    "max_concurrent": 10,   # Maximum concurrent requests
    "timeout": 5.0,        # Shard timeout in seconds
    "retry_count": 3,      # Shard retry count
    "retry_delay": 1.0,    # Shard retry delay
    "health_interval": 60.0,  # Shard health check interval
    "size_limit": 1000000,  # Maximum vectors per shard
    "replication_factor": 2,  # Number of replicas
}

# Create sharded searcher
sharded_searcher = APISearcher(
    sharding_config=sharding_config,
    shard_configs=[shard_config],
)

# Search with custom configuration
results = await sharded_searcher.search(
    query="user authentication",
    use_sharding=True,
)

# Add vectors with sharding
success = await sharded_searcher.add_vectors(
    vectors=vectors,
    metadata=metadata,
    use_sharding=True,
)
```

## Configuration

### Environment Variables
- `API_DIR`: Directory containing API specifications
- `CACHE_TTL`: Cache time-to-live in seconds
- `MODEL_NAME`: Name of the sentence transformer model
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_ENV`: Pinecone environment
- `ZMQ_PUB_PORT`: ZeroMQ publisher port
- `ZMQ_SUB_PORT`: ZeroMQ subscriber port

### Search Configuration
- `top_k`: Number of results to return (default: 10)
- `min_score`: Minimum similarity score (default: 0.5)
- `use_cache`: Whether to use caching (default: True)
- `enhance_results`: Whether to enhance results (default: True)

### Quality Configuration
- `min_clusters`: Minimum number of clusters (default: 2)
- `max_clusters`: Maximum number of clusters (default: 5)
- `min_diversity_score`: Minimum diversity score (default: 0.3)
- `max_similarity`: Maximum allowed similarity (default: 0.8)

### Personalization Configuration
- `history_weight`: Weight for historical interactions (default: 0.3)
- `preference_weight`: Weight for category preferences (default: 0.3)
- `tag_weight`: Weight for tag preferences (default: 0.2)
- `recency_weight`: Weight for recency bias (default: 0.2)

### Business Context Configuration
- `business_weight`: Weight for business rules (default: 0.4)
- `feature_weight`: Weight for feature flags (default: 0.3)
- `constraint_weight`: Weight for constraints (default: 0.3)
- `max_boost`: Maximum boost factor (default: 2.0)

### Feedback Configuration
- `click_weight`: Weight for click feedback (default: 0.3)
- `relevance_weight`: Weight for relevance feedback (default: 0.4)
- `bookmark_weight`: Weight for bookmark feedback (default: 0.2)
- `report_weight`: Weight for report feedback (default: 0.1)
- `min_feedback`: Minimum feedback entries (default: 5)
- `max_feedback`: Maximum feedback entries (default: 1000)
- `learning_rate`: Learning rate for updates (default: 0.1)
- `decay_factor`: Time decay factor (default: 0.01)

### Filter Configuration
- `semantic_weight`: Weight for semantic filters (default: 0.4)
- `category_weight`: Weight for category filters (default: 0.3)
- `attribute_weight`: Weight for attribute filters (default: 0.2)
- `constraint_weight`: Weight for constraint filters (default: 0.1)
- `min_threshold`: Minimum similarity threshold (default: 0.5)
- `max_threshold`: Maximum similarity threshold (default: 0.9)
- `cache_embeddings`: Whether to cache embeddings (default: True)
- `max_cache_size`: Maximum cache size (default: 10000)

### Reformulation Configuration
- `expansion_weight`: Weight for query expansion (default: 0.4)
- `template_weight`: Weight for template matching (default: 0.3)
- `intent_weight`: Weight for intent detection (default: 0.2)
- `spelling_weight`: Weight for spell checking (default: 0.1)
- `min_confidence`: Minimum confidence threshold (default: 0.5)
- `max_expansions`: Maximum number of expansions (default: 3)
- `max_templates`: Maximum number of templates (default: 2)
- `max_intents`: Maximum number of intents (default: 2)
- `cache_embeddings`: Whether to cache embeddings (default: True)
- `max_cache_size`: Maximum cache size (default: 10000)

### Distributed Configuration
- `min_nodes`: Minimum number of nodes (default: 2)
- `max_nodes`: Maximum number of nodes (default: 10)
- `min_healthy`: Minimum healthy node ratio (default: 0.5)
- `strategy`: Load balancing strategy (default: "round_robin")
- `timeout`: Request timeout in seconds (default: 10.0)
- `retry_count`: Number of retries (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 1.0)
- `health_interval`: Health check interval in seconds (default: 60.0)

### Node Configuration
- `node_id`: Node identifier
- `host`: Node hostname (default: "localhost")
- `port`: Node port (default: 8000)
- `weight`: Node weight for load balancing (default: 1.0)
- `max_concurrent`: Maximum concurrent requests (default: 10)
- `timeout`: Node timeout in seconds (default: 5.0)
- `retry_count`: Node retry count (default: 3)
- `retry_delay`: Node retry delay in seconds (default: 1.0)
- `health_interval`: Node health check interval in seconds (default: 60.0)

### Sharding Configuration
- `min_shards`: Minimum number of shards (default: 2)
- `max_shards`: Maximum number of shards (default: 10)
- `min_healthy`: Minimum healthy shard ratio (default: 0.5)
- `strategy`: Sharding strategy (default: "consistent_hashing")
- `timeout`: Request timeout in seconds (default: 10.0)
- `retry_count`: Number of retries (default: 3)
- `retry_delay`: Delay between retries in seconds (default: 1.0)
- `health_interval`: Health check interval in seconds (default: 60.0)
- `rebalance_threshold`: Rebalance threshold (default: 0.2)
- `rebalance_interval`: Rebalance interval in seconds (default: 3600.0)

### Shard Configuration
- `shard_id`: Shard identifier
- `host`: Shard hostname (default: "localhost")
- `port`: Shard port (default: 8000)
- `weight`: Shard weight for load balancing (default: 1.0)
- `max_concurrent`: Maximum concurrent requests (default: 10)
- `timeout`: Shard timeout in seconds (default: 5.0)
- `retry_count`: Shard retry count (default: 3)
- `retry_delay`: Shard retry delay in seconds (default: 1.0)
- `health_interval`: Shard health check interval in seconds (default: 60.0)
- `size_limit`: Maximum vectors per shard (default: 1000000)
- `replication_factor`: Number of replicas (default: 2)

## Development

### Running Tests
```bash
poetry run pytest
```

### Code Style
```bash
poetry run black .
poetry run isort .
poetry run mypy .
poetry run pylint plexure_api_search
```

### Pre-commit Hooks
```bash
poetry run pre-commit install
```

## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Built with [Poetry](https://python-poetry.org/)
- Vector search powered by [Pinecone](https://www.pinecone.io/)
- Embeddings by [Sentence Transformers](https://www.sbert.net/)
- TUI powered by [Rich](https://rich.readthedocs.io/)
- IPC using [ZeroMQ](https://zeromq.org/)
