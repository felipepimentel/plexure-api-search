# User Guide

## Introduction

Welcome to the Plexure API Search user guide. This document will help you get started with using the system and explain its main features.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Advanced Features](#advanced-features)
4. [Best Practices](#best-practices)
5. [Troubleshooting](#troubleshooting)

## Getting Started

### Prerequisites

Before using Plexure API Search, ensure you have:

- Python 3.8 or higher
- Access to a Pinecone instance
- API contract files in YAML or JSON format

### Installation

1. Install using pip:

```bash
pip install plexure-api-search
```

2. Set up environment variables:

```bash
# Core settings
export API_DIR=/path/to/apis
export MODEL_NAME=all-MiniLM-L6-v2

# Pinecone settings
export PINECONE_API_KEY=your_api_key
export PINECONE_ENV=your_environment

# Optional settings
export CACHE_TTL=3600
export LOG_LEVEL=INFO
```

3. Initialize the system:

```bash
plexure-api-search init
```

## Basic Usage

### Indexing API Files

1. Index your API files:

```bash
plexure-api-search index
```

This will:
- Scan your API directory
- Parse API contracts
- Generate embeddings
- Store vectors in Pinecone

Monitor indexing progress:

```bash
plexure-api-search monitor
```

### Searching APIs

1. Basic search:

```bash
plexure-api-search search "create user account"
```

2. Search with filters:

```bash
plexure-api-search search "create user" --method POST --tags user,account
```

3. View detailed results:

```bash
plexure-api-search search "payment process" --verbose
```

### Managing the Index

1. Update index:

```bash
plexure-api-search index --update
```

2. Clear index:

```bash
plexure-api-search index --clear
```

3. Check index status:

```bash
plexure-api-search index --status
```

## Advanced Features

### Semantic Search

The system uses advanced semantic search capabilities:

1. Query Understanding
- Natural language processing
- Intent detection
- Query expansion
- Spell checking

2. Contextual Search
- Business domain awareness
- User context
- Feature flags

Example:

```bash
plexure-api-search search "user signup" \
  --domain retail \
  --context '{"features": ["premium"]}'
```

### Personalization

Customize search results based on:

1. User Preferences
- Search history
- Frequently used APIs
- Favorite endpoints

2. Business Rules
- Access permissions
- Usage quotas
- API versions

Example:

```bash
plexure-api-search search "order status" \
  --user-id user123 \
  --preferences '{"versions": ["v2"]}'
```

### Analytics

Monitor search performance:

1. Usage Analytics
```bash
plexure-api-search analytics --type usage
```

2. Performance Metrics
```bash
plexure-api-search analytics --type performance
```

3. Quality Metrics
```bash
plexure-api-search analytics --type quality
```

### A/B Testing

Run experiments:

1. Create experiment:
```bash
plexure-api-search experiment create \
  --name "ranking_algorithm" \
  --variants "default,improved" \
  --duration 7d
```

2. View results:
```bash
plexure-api-search experiment results ranking_algorithm
```

## Best Practices

### Search Optimization

1. Query Construction
- Use natural language
- Be specific but concise
- Include relevant context

2. Filter Usage
- Use filters to narrow results
- Combine multiple filters
- Order filters by importance

3. Result Processing
- Check relevance scores
- Review multiple results
- Use detailed view for important queries

### Performance Optimization

1. Indexing
- Regular updates
- Incremental indexing
- Optimal batch size

2. Caching
- Enable result caching
- Set appropriate TTL
- Monitor cache hit rate

3. Resource Usage
- Monitor memory usage
- Scale vector store
- Use async operations

### Monitoring

1. Health Checks
```bash
plexure-api-search monitor --health
```

2. Performance Monitoring
```bash
plexure-api-search monitor --performance
```

3. Error Tracking
```bash
plexure-api-search monitor --errors
```

## Troubleshooting

### Common Issues

1. Search Quality
- Problem: Poor search results
- Solution: Check query construction, update index
- Prevention: Regular quality monitoring

2. Performance
- Problem: Slow response time
- Solution: Enable caching, optimize queries
- Prevention: Performance monitoring

3. Indexing
- Problem: Failed indexing
- Solution: Check file formats, permissions
- Prevention: Validate files before indexing

### Error Messages

Common error messages and solutions:

1. "Invalid API format"
- Check file syntax
- Validate against schema
- Use correct format (YAML/JSON)

2. "Connection failed"
- Check network connectivity
- Verify credentials
- Check service status

3. "Resource limit exceeded"
- Scale resources
- Optimize usage
- Check quotas

### Getting Help

1. Documentation
- API documentation
- Developer guide
- Deployment guide

2. Support
- GitHub issues
- Support email
- Community forum

3. Updates
- Release notes
- Change log
- Migration guides

## Configuration Reference

### Core Settings

```yaml
# config.yaml
api:
  dir: /path/to/apis
  formats: [yaml, json]
  recursive: true

model:
  name: all-MiniLM-L6-v2
  batch_size: 32
  cache_dir: .cache

vector_store:
  type: pinecone
  dimension: 384
  metric: cosine
  namespace: api-search

search:
  limit: 10
  min_score: 0.5
  timeout: 5.0
  cache_ttl: 3600
```

### Feature Flags

```yaml
features:
  spell_check: true
  query_expansion: true
  personalization: true
  analytics: true
  caching: true
  monitoring: true
```

### Monitoring Settings

```yaml
monitoring:
  health_check_interval: 60
  metrics_interval: 300
  log_level: INFO
  export_metrics: true
  alert_on_errors: true
```

## Command Reference

### Core Commands

```bash
# Initialize system
plexure-api-search init [--config path/to/config.yaml]

# Index APIs
plexure-api-search index [--path dir] [--force] [--async]

# Search APIs
plexure-api-search search <query> [--filters json] [--limit n]

# Monitor system
plexure-api-search monitor [--type health|metrics|logs]
```

### Management Commands

```bash
# Manage cache
plexure-api-search cache [clear|status]

# Manage experiments
plexure-api-search experiment [create|list|results]

# Export data
plexure-api-search export [analytics|metrics|logs]
```

### Utility Commands

```bash
# Validate configuration
plexure-api-search validate [--config file]

# Check version
plexure-api-search version

# Show help
plexure-api-search --help
``` 