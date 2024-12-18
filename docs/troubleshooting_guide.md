# Troubleshooting Guide

## Common Issues

### Installation Issues

1. Package Installation Fails
```
Problem: pip install fails with dependency conflicts
Solution:
- Create fresh virtual environment
- Upgrade pip: pip install --upgrade pip
- Install with extras: pip install "plexure-api-search[all]"
```

2. CUDA Installation
```
Problem: GPU support not working
Solution:
- Verify CUDA installation: nvidia-smi
- Install correct CUDA version: cuda-11.0+
- Set CUDA_VISIBLE_DEVICES
```

### Configuration Issues

1. Invalid Configuration
```
Problem: Service fails to start with configuration error
Solution:
- Validate config: plexure-api-search validate
- Check environment variables
- Verify file permissions
```

2. Missing API Directory
```
Problem: No API files found
Solution:
- Verify API_DIR path exists
- Check file permissions
- Ensure correct file formats (YAML/JSON)
```

### Search Issues

1. Poor Search Results
```
Problem: Irrelevant or missing results
Solution:
- Update index: plexure-api-search index --force
- Check query construction
- Adjust min_score in config
- Verify API contract quality
```

2. Slow Search Performance
```
Problem: High search latency
Solution:
- Enable caching
- Optimize batch size
- Scale vector store
- Monitor resource usage
```

### Indexing Issues

1. Failed Indexing
```
Problem: Indexing process fails
Solution:
- Check API file formats
- Verify Pinecone connection
- Monitor memory usage
- Use incremental indexing
```

2. Missing Documents
```
Problem: Documents not appearing in search
Solution:
- Check indexing status
- Verify document format
- Update index manually
- Check error logs
```

### Vector Store Issues

1. Connection Errors
```
Problem: Cannot connect to Pinecone
Solution:
- Verify API key and environment
- Check network connectivity
- Test with pinecone-cli
- Monitor rate limits
```

2. Out of Memory
```
Problem: Vector store operations fail with OOM
Solution:
- Reduce batch size
- Implement sharding
- Scale resources
- Clean unused vectors
```

### Performance Issues

1. High CPU Usage
```
Problem: Service using excessive CPU
Solution:
- Profile with: plexure-api-search profile
- Optimize batch processing
- Scale horizontally
- Monitor worker processes
```

2. Memory Leaks
```
Problem: Growing memory usage
Solution:
- Enable memory profiling
- Check resource cleanup
- Monitor memory patterns
- Implement circuit breakers
```

### Monitoring Issues

1. Missing Metrics
```
Problem: Metrics not appearing
Solution:
- Check Prometheus config
- Verify metrics endpoint
- Enable debug logging
- Check collector status
```

2. False Alerts
```
Problem: Receiving incorrect alerts
Solution:
- Adjust thresholds
- Verify alert rules
- Check metric values
- Update alert conditions
```

## Debugging Tools

### Log Analysis

1. Enable Debug Logging
```bash
# Set log level
export LOG_LEVEL=DEBUG

# Run with debug output
plexure-api-search serve --debug

# Check logs
tail -f /var/log/plexure/debug.log
```

2. Log Patterns
```
# Error patterns to watch:
- "Failed to connect" -> Network issues
- "Out of memory" -> Resource issues
- "Timeout exceeded" -> Performance issues
- "Invalid format" -> Data issues
```

### Performance Profiling

1. CPU Profiling
```bash
# Run with profiler
plexure-api-search profile cpu

# Analyze results
plexure-api-search profile analyze cpu.prof
```

2. Memory Profiling
```bash
# Track memory usage
plexure-api-search profile memory

# Find leaks
plexure-api-search profile leaks
```

### Network Diagnostics

1. Connection Testing
```bash
# Test Pinecone
plexure-api-search test pinecone

# Test endpoints
plexure-api-search test endpoints

# Monitor latency
plexure-api-search monitor latency
```

2. Traffic Analysis
```bash
# Monitor requests
plexure-api-search monitor requests

# Check bandwidth
plexure-api-search monitor bandwidth
```

## Recovery Procedures

### Service Recovery

1. Service Fails to Start
```bash
# Check status
systemctl status plexure

# View logs
journalctl -u plexure -n 100

# Restart service
systemctl restart plexure
```

2. Service Becomes Unresponsive
```bash
# Check health
curl http://localhost:8000/health

# Kill stuck process
pkill -f plexure-api-search

# Restart with monitoring
plexure-api-search serve --monitor
```

### Data Recovery

1. Index Corruption
```bash
# Backup current state
plexure-api-search backup

# Clear index
plexure-api-search index --clear

# Rebuild index
plexure-api-search index --force
```

2. Lost Vector Data
```bash
# Restore from backup
plexure-api-search restore --latest

# Verify restoration
plexure-api-search verify

# Rebuild if needed
plexure-api-search rebuild
```

### System Recovery

1. Resource Exhaustion
```bash
# Free resources
plexure-api-search cleanup

# Scale resources
plexure-api-search scale up

# Monitor usage
plexure-api-search monitor resources
```

2. Network Issues
```bash
# Test connectivity
plexure-api-search test network

# Reset connections
plexure-api-search reset connections

# Enable fallback
plexure-api-search enable fallback
```

## Prevention Measures

### Monitoring Setup

1. Health Checks
```yaml
monitoring:
  health_check:
    enabled: true
    interval: 60
    timeout: 10
    endpoints:
      - /health
      - /metrics
```

2. Alert Configuration
```yaml
alerts:
  rules:
    - name: high_error_rate
      condition: error_rate > 0.1
      duration: 5m
      severity: critical
```

### Backup Strategy

1. Regular Backups
```yaml
backup:
  schedule: "0 2 * * *"
  retention: 7
  type: full
  destination: s3://backup
```

2. Verification
```yaml
verify:
  schedule: "0 3 * * *"
  checks:
    - backup_integrity
    - data_consistency
```

### Resource Management

1. Resource Limits
```yaml
resources:
  limits:
    cpu: 4
    memory: 8Gi
    storage: 100Gi
```

2. Scaling Rules
```yaml
scaling:
  rules:
    - metric: cpu_usage
      threshold: 80
      action: scale_up
```

## Best Practices

### Development

1. Code Quality
- Use linters
- Run tests
- Follow style guide
- Document changes

2. Testing
- Unit tests
- Integration tests
- Performance tests
- Security tests

### Deployment

1. Release Process
- Version control
- Change log
- Backup first
- Rollback plan

2. Configuration
- Use version control
- Validate changes
- Test in staging
- Monitor deployment

### Operations

1. Monitoring
- Watch metrics
- Check logs
- Set alerts
- Regular audits

2. Maintenance
- Regular updates
- Security patches
- Performance tuning
- Capacity planning

## Getting Help

### Support Channels

1. Community Support
- GitHub Issues
- Stack Overflow
- Discord Channel
- Documentation

2. Commercial Support
- Email Support
- Phone Support
- SLA Coverage
- Priority Response

### Reporting Issues

1. Issue Template
```markdown
## Problem Description
[Describe the issue]

## Steps to Reproduce
1. [First Step]
2. [Second Step]
3. [Additional Steps...]

## Expected Behavior
[What you expected to happen]

## Actual Behavior
[What actually happened]

## System Information
- Version: [e.g. 1.0.0]
- OS: [e.g. Ubuntu 20.04]
- Python: [e.g. 3.8.5]

## Logs
```[logs]
[Relevant log output]
```
```

2. Debug Information
```bash
# Gather debug info
plexure-api-search debug info > debug.log

# Create report
plexure-api-search report create
``` 