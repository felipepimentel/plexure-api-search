# Deployment Guide

## Overview

This guide provides instructions for deploying the Plexure API Search system in various environments.

## System Requirements

### Hardware Requirements

1. **Minimum Requirements**

   - CPU: 2 cores
   - RAM: 4GB
   - Storage: 10GB

2. **Recommended Requirements**
   - CPU: 4+ cores
   - RAM: 8GB+
   - Storage: 20GB+
   - AVX2 support for FAISS optimization

### Software Requirements

1. **Base Requirements**

   - Python 3.9+
   - Poetry
   - Git
   - FAISS with AVX2 support

2. **Optional Requirements**
   - Docker
   - Prometheus (for metrics)
   - Grafana (for monitoring)

## Installation

### From Source

1. Clone the repository:

```bash
git clone https://github.com/yourusername/plexure-api-search.git
cd plexure-api-search
```

2. Install dependencies:

```bash
poetry install
```

3. Set up environment:

```bash
cp .env.sample .env
```

### Using Docker

1. Build image:

```bash
docker build -t plexure-api-search .
```

2. Run container:

```bash
docker run -p 5555:5555 plexure-api-search
```

## Configuration

### Environment Variables

| Variable                     | Description                                  | Default                                | Required |
| ---------------------------- | -------------------------------------------- | -------------------------------------- | -------- |
| `ENVIRONMENT`                | Environment (development/staging/production) | development                            | Yes      |
| `DEBUG`                      | Enable debug mode                            | false                                  | No       |
| `API_DIR`                    | Directory containing API contracts           | assets/apis                            | Yes      |
| `CACHE_DIR`                  | Cache directory                              | .cache/default                         | Yes      |
| `METRICS_DIR`                | Metrics directory                            | .cache/metrics                         | Yes      |
| `MODEL_NAME`                 | Embedding model name                         | sentence-transformers/all-MiniLM-L6-v2 | Yes      |
| `MODEL_DIMENSION`            | Embedding dimension                          | 384                                    | Yes      |
| `MODEL_BATCH_SIZE`           | Model batch size                             | 32                                     | No       |
| `MODEL_NORMALIZE_EMBEDDINGS` | Normalize embeddings                         | true                                   | No       |
| `ENABLE_TELEMETRY`           | Enable metrics collection                    | true                                   | No       |
| `METRICS_BACKEND`            | Metrics backend (prometheus)                 | prometheus                             | No       |
| `PUBLISHER_PORT`             | Event publisher port                         | 5555                                   | No       |
| `MIN_SCORE`                  | Minimum similarity score                     | 0.1                                    | No       |
| `TOP_K`                      | Default number of results                    | 10                                     | No       |
| `EXPAND_QUERY`               | Enable query expansion                       | true                                   | No       |
| `RERANK_RESULTS`             | Enable result reranking                      | true                                   | No       |
| `LOG_LEVEL`                  | Logging level                                | INFO                                   | No       |

### Configuration Files

1. **Environment Files**

   - `.env`: Environment variables
   - `.env.sample`: Example environment variables

2. **Configuration Files**
   - `config.yaml`: Default configuration
   - `config.dev.yaml`: Development configuration
   - `config.prod.yaml`: Production configuration
   - `config.test.yaml`: Test configuration

## Directory Structure

```
plexure_api_search/
├── assets/           # Static assets
│   └── apis/        # API contracts
├── .cache/          # Cache directory
│   ├── default/     # Default cache
│   └── metrics/     # Metrics cache
├── .models/         # Model cache
└── logs/           # Log files
```

## Deployment Steps

### 1. Prepare Environment

1. Create directories:

```bash
mkdir -p assets/apis .cache/default .cache/metrics .models logs
```

2. Set permissions:

```bash
chmod 755 assets/apis .cache/default .cache/metrics .models logs
```

### 2. Configure System

1. Set environment variables:

```bash
export ENVIRONMENT=production
export LOG_LEVEL=INFO
export DEBUG=false
```

2. Update configuration:

```bash
cp config.prod.yaml config.yaml
```

### 3. Initialize System

1. Install package:

```bash
poetry install --no-dev
```

2. Verify installation:

```bash
poetry run python -m plexure_api_search --version
```

### 4. Index API Contracts

1. Add API contracts:

```bash
cp /path/to/contracts/* assets/apis/
```

2. Run indexing:

```bash
poetry run python -m plexure_api_search index --clear
```

## Monitoring Setup

### Prometheus Integration

1. Configure Prometheus:

```yaml
scrape_configs:
  - job_name: "plexure-api-search"
    static_configs:
      - targets: ["localhost:5555"]
```

2. Available metrics:
   - `embeddings_generated_total`
   - `embedding_errors_total`
   - `searches_performed_total`
   - `search_errors_total`
   - `contract_errors_total`
   - `index_size`
   - `metadata_size`
   - `search_latency_seconds`
   - `embedding_latency_seconds`

### Logging

1. Log files:

   - `logs/app.log`: Application logs
   - `logs/error.log`: Error logs
   - `logs/access.log`: Access logs

2. Log format:

```
%(asctime)s - %(name)s - %(levelname)s - %(message)s
```

## Security

### File Permissions

1. Set ownership:

```bash
chown -R service_user:service_group plexure-api-search/
```

2. Set permissions:

```bash
find plexure-api-search/ -type d -exec chmod 755 {} \;
find plexure-api-search/ -type f -exec chmod 644 {} \;
```

### Network Security

1. Firewall rules:

```bash
# Allow metrics port
ufw allow 5555/tcp
```

2. Rate limiting:
   - Default: 100 requests per minute
   - Configurable through environment variables

## Maintenance

### Backup

1. Backup directories:

   - `assets/apis/`: API contracts
   - `.cache/`: Cache data
   - `.models/`: Model cache

2. Backup command:

```bash
tar -czf backup.tar.gz assets/apis/ .cache/ .models/
```

### Updates

1. Update package:

```bash
git pull
poetry install
```

2. Rebuild index:

```bash
poetry run python -m plexure_api_search index --clear
```

## Troubleshooting

### Common Issues

1. **Search Returns No Results**

   - Check if contracts are indexed
   - Verify query format
   - Check similarity threshold
   - Review contract descriptions

2. **Indexing Fails**

   - Validate contract format
   - Check file permissions
   - Verify storage space
   - Review error messages

3. **Performance Issues**
   - Monitor memory usage
   - Check batch sizes
   - Review cache settings
   - Profile critical paths

### Logs

1. Check application logs:

```bash
tail -f logs/app.log
```

2. Check error logs:

```bash
tail -f logs/error.log
```

### Metrics

1. Check metrics endpoint:

```bash
curl http://localhost:5555/metrics
```

2. Monitor system metrics:

```bash
top -p $(pgrep -f plexure-api-search)
```

## Health Checks

### System Health

1. Check process:

```bash
systemctl status plexure-api-search
```

2. Check ports:

```bash
netstat -tulpn | grep 5555
```

### Application Health

1. Check metrics:

```bash
curl http://localhost:5555/metrics
```

2. Run search test:

```bash
poetry run python -m plexure_api_search search "test"
```

## Performance Tuning

### Model Settings

1. Batch size:

```bash
export MODEL_BATCH_SIZE=64
```

2. Vector dimension:

```bash
export MODEL_DIMENSION=384
```

### Cache Settings

1. Cache directories:

```bash
export CACHE_DIR=.cache/production
```

2. Cache cleanup:

```bash
find .cache/ -type f -mtime +30 -delete
```

## Scaling

### Vertical Scaling

1. Increase resources:

   - CPU cores
   - RAM
   - Storage

2. Update settings:
   - Batch size
   - Cache size
   - Thread count

### Horizontal Scaling

1. Load balancing:

   - Multiple instances
   - Shared storage
   - Distributed cache

2. Monitoring:
   - Instance health
   - Resource usage
   - Error rates
