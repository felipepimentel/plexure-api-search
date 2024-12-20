# Troubleshooting Guide

## Common Issues

### Search Issues

#### No Results Found

**Symptoms:**

- Search returns empty results
- Search returns fewer results than expected
- Results have low relevance scores

**Possible Causes:**

1. API contracts not indexed
2. Query format issues
3. Similarity threshold too high
4. Poor contract descriptions
5. Model mismatch

**Solutions:**

1. Check indexing:

```bash
poetry run python -m plexure_api_search index --status
```

2. Verify query format:

- Use clear, descriptive queries
- Include relevant keywords
- Check for typos

3. Adjust similarity threshold:

```bash
export MIN_SCORE=0.1  # Lower threshold
```

4. Review contract descriptions:

- Ensure descriptions are clear
- Add relevant keywords
- Use consistent terminology

5. Check model configuration:

```bash
poetry run python -m plexure_api_search config show
```

#### Poor Search Results

**Symptoms:**

- Irrelevant results
- Missing obvious matches
- Inconsistent rankings

**Possible Causes:**

1. Query expansion issues
2. Reranking problems
3. Model configuration
4. Contract quality

**Solutions:**

1. Configure query expansion:

```bash
export EXPAND_QUERY=true
```

2. Enable reranking:

```bash
export RERANK_RESULTS=true
```

3. Update model settings:

```bash
export MODEL_NORMALIZE_EMBEDDINGS=true
```

4. Improve contracts:

- Add detailed descriptions
- Use consistent formatting
- Include example usage

### Indexing Issues

#### Indexing Fails

**Symptoms:**

- Index command fails
- Partial indexing
- Missing endpoints

**Possible Causes:**

1. Invalid contract format
2. File permissions
3. Storage space
4. Memory issues

**Solutions:**

1. Validate contracts:

```bash
poetry run python -m plexure_api_search validate
```

2. Check permissions:

```bash
ls -la assets/apis/
chmod -R 644 assets/apis/*
```

3. Verify storage:

```bash
df -h
du -sh .cache/
```

4. Monitor memory:

```bash
free -h
top -p $(pgrep -f plexure-api-search)
```

#### Slow Indexing

**Symptoms:**

- Long indexing times
- High CPU usage
- Memory warnings

**Possible Causes:**

1. Large contracts
2. Small batch size
3. Resource constraints
4. Cache issues

**Solutions:**

1. Split contracts:

- Break into smaller files
- Use modular structure

2. Adjust batch size:

```bash
export MODEL_BATCH_SIZE=64
```

3. Increase resources:

- Add CPU cores
- Increase RAM
- Use SSD storage

4. Clear cache:

```bash
rm -rf .cache/*
```

### Performance Issues

#### High Latency

**Symptoms:**

- Slow search responses
- Timeouts
- High CPU usage

**Possible Causes:**

1. Resource constraints
2. Large index size
3. Cache misses
4. Network issues

**Solutions:**

1. Monitor resources:

```bash
top -p $(pgrep -f plexure-api-search)
```

2. Optimize index:

```bash
poetry run python -m plexure_api_search index --optimize
```

3. Configure cache:

```bash
export CACHE_DIR=.cache/production
```

4. Check network:

```bash
netstat -tulpn | grep 5555
```

#### Memory Issues

**Symptoms:**

- Out of memory errors
- Swap usage
- Process crashes

**Possible Causes:**

1. Large model
2. Memory leaks
3. Cache size
4. Concurrent requests

**Solutions:**

1. Use smaller model:

```bash
export MODEL_NAME="sentence-transformers/all-MiniLM-L6-v2"
```

2. Monitor memory:

```bash
watch -n 1 'ps -o pid,ppid,%mem,rss,cmd -p $(pgrep -f plexure-api-search)'
```

3. Limit cache:

```bash
find .cache/ -type f -mtime +7 -delete
```

4. Rate limiting:

```bash
export MAX_CONCURRENT_REQUESTS=10
```

### Configuration Issues

#### Environment Variables

**Symptoms:**

- Missing configuration
- Default values used
- Unexpected behavior

**Possible Causes:**

1. Unset variables
2. Invalid values
3. Permission issues
4. Shell configuration

**Solutions:**

1. Check variables:

```bash
env | grep PLEXURE
```

2. Validate values:

```bash
poetry run python -m plexure_api_search config validate
```

3. Set permissions:

```bash
chmod 600 .env
```

4. Update shell:

```bash
source .env
```

#### File Paths

**Symptoms:**

- File not found errors
- Permission denied
- Path resolution issues

**Possible Causes:**

1. Invalid paths
2. Missing directories
3. Permission issues
4. Relative paths

**Solutions:**

1. Check paths:

```bash
ls -la $(poetry run python -m plexure_api_search config paths)
```

2. Create directories:

```bash
mkdir -p assets/apis .cache/default .cache/metrics
```

3. Set permissions:

```bash
chmod -R 755 assets/apis/
```

4. Use absolute paths:

```bash
export API_DIR=$(pwd)/assets/apis
```

### Model Issues

#### Model Loading

**Symptoms:**

- Model load failures
- Initialization errors
- Missing files

**Possible Causes:**

1. Download issues
2. Corruption
3. Version mismatch
4. Cache problems

**Solutions:**

1. Clear cache:

```bash
rm -rf .models/*
```

2. Verify files:

```bash
poetry run python -m plexure_api_search model verify
```

3. Check version:

```bash
poetry run python -m plexure_api_search model info
```

4. Update cache:

```bash
poetry run python -m plexure_api_search model update
```

#### Embedding Issues

**Symptoms:**

- Embedding errors
- Dimension mismatch
- Quality issues

**Possible Causes:**

1. Model configuration
2. Input format
3. Normalization
4. Batch size

**Solutions:**

1. Check config:

```bash
poetry run python -m plexure_api_search model config
```

2. Validate input:

```bash
poetry run python -m plexure_api_search validate --input
```

3. Enable normalization:

```bash
export MODEL_NORMALIZE_EMBEDDINGS=true
```

4. Adjust batch:

```bash
export MODEL_BATCH_SIZE=32
```

### Logging and Monitoring

#### Missing Logs

**Symptoms:**

- No log output
- Incomplete logs
- Wrong log level

**Possible Causes:**

1. Log configuration
2. File permissions
3. Disk space
4. Log rotation

**Solutions:**

1. Set log level:

```bash
export LOG_LEVEL=DEBUG
```

2. Check permissions:

```bash
chmod 644 logs/*.log
```

3. Check space:

```bash
df -h logs/
```

4. Configure rotation:

```bash
find logs/ -type f -mtime +7 -delete
```

#### Metrics Issues

**Symptoms:**

- Missing metrics
- Invalid values
- Collection errors

**Possible Causes:**

1. Telemetry disabled
2. Port conflicts
3. Backend issues
4. Collection errors

**Solutions:**

1. Enable telemetry:

```bash
export ENABLE_TELEMETRY=true
```

2. Check port:

```bash
netstat -tulpn | grep 5555
```

3. Verify backend:

```bash
curl http://localhost:5555/metrics
```

4. Reset metrics:

```bash
rm -rf .cache/metrics/*
```

## Debugging Tools

### Log Analysis

1. View recent logs:

```bash
tail -f logs/app.log
```

2. Search errors:

```bash
grep ERROR logs/app.log
```

3. Count occurrences:

```bash
grep -c "pattern" logs/app.log
```

### Performance Analysis

1. CPU profiling:

```bash
poetry run python -m cProfile -o profile.stats main.py
```

2. Memory profiling:

```bash
poetry run python -m memory_profiler main.py
```

3. System monitoring:

```bash
htop -p $(pgrep -f plexure-api-search)
```

### Network Analysis

1. Port scanning:

```bash
netstat -tulpn
```

2. Traffic monitoring:

```bash
tcpdump -i any port 5555
```

3. Connection testing:

```bash
curl -v http://localhost:5555/health
```

## Support Resources

### Documentation

1. View docs:

```bash
poetry run python -m plexure_api_search docs
```

2. API reference:

```bash
poetry run python -m plexure_api_search docs api
```

### Community Support

1. GitHub Issues:

- Report bugs
- Request features
- Share feedback

2. Discussion Forums:

- Ask questions
- Share experiences
- Get help

### Contact Support

1. Email Support:

- support@plexure.com
- Include logs
- Describe steps

2. Emergency Contact:

- emergency@plexure.com
- 24/7 support
- Critical issues
