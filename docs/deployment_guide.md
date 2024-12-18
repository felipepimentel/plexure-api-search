# Deployment Guide

## System Requirements

### Hardware Requirements

1. Minimum Requirements
- CPU: 4 cores
- RAM: 8GB
- Storage: 50GB SSD
- Network: 100Mbps

2. Recommended Requirements
- CPU: 8+ cores
- RAM: 16GB+
- Storage: 100GB+ SSD
- Network: 1Gbps+

3. Production Requirements
- CPU: 16+ cores
- RAM: 32GB+
- Storage: 500GB+ SSD
- Network: 10Gbps+

### Software Requirements

1. Operating System
- Linux (Ubuntu 20.04+ recommended)
- Container support
- Systemd or similar init system

2. Python Environment
- Python 3.8+
- pip/poetry
- virtualenv

3. Dependencies
- Docker 20.10+
- Docker Compose 2.0+
- NVIDIA drivers (if using GPU)
- CUDA 11.0+ (if using GPU)

## Installation

### Package Installation

1. Install system dependencies:
```bash
apt-get update
apt-get install -y \
    python3.8 \
    python3.8-venv \
    python3.8-dev \
    build-essential \
    git
```

2. Create service user:
```bash
useradd -m -s /bin/bash plexure
```

3. Install package:
```bash
python3.8 -m pip install plexure-api-search
```

### Configuration

1. Create configuration directory:
```bash
mkdir -p /etc/plexure
chown plexure:plexure /etc/plexure
```

2. Create configuration file:
```bash
cat > /etc/plexure/config.yaml << EOF
api:
  dir: /var/lib/plexure/apis
  formats: [yaml, json]
  recursive: true

model:
  name: all-MiniLM-L6-v2
  batch_size: 32
  cache_dir: /var/cache/plexure

vector_store:
  type: pinecone
  dimension: 384
  metric: cosine
  namespace: production

search:
  limit: 10
  min_score: 0.5
  timeout: 5.0
  cache_ttl: 3600

monitoring:
  health_check_interval: 60
  metrics_interval: 300
  log_level: INFO
  export_metrics: true
  alert_on_errors: true
EOF
```

3. Set environment variables:
```bash
cat > /etc/plexure/env << EOF
PINECONE_API_KEY=your_api_key
PINECONE_ENV=production
MODEL_NAME=all-MiniLM-L6-v2
LOG_LEVEL=INFO
EOF
```

### Service Setup

1. Create systemd service:
```bash
cat > /etc/systemd/system/plexure.service << EOF
[Unit]
Description=Plexure API Search
After=network.target

[Service]
Type=simple
User=plexure
Group=plexure
EnvironmentFile=/etc/plexure/env
ExecStart=/usr/local/bin/plexure-api-search serve
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
EOF
```

2. Enable and start service:
```bash
systemctl enable plexure
systemctl start plexure
```

## Docker Deployment

### Docker Setup

1. Create Dockerfile:
```dockerfile
FROM python:3.8-slim

# Install dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create user
RUN useradd -m -s /bin/bash plexure

# Install package
RUN pip install plexure-api-search

# Copy configuration
COPY config.yaml /etc/plexure/config.yaml
COPY env /etc/plexure/env

# Set user
USER plexure

# Run service
CMD ["plexure-api-search", "serve"]
```

2. Create docker-compose.yml:
```yaml
version: '3.8'

services:
  plexure:
    build: .
    ports:
      - "8000:8000"
    volumes:
      - ./apis:/var/lib/plexure/apis
      - ./cache:/var/cache/plexure
    env_file:
      - ./env
    restart: always
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

3. Build and run:
```bash
docker-compose up -d
```

## Kubernetes Deployment

### Kubernetes Setup

1. Create namespace:
```bash
kubectl create namespace plexure
```

2. Create config map:
```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: plexure-config
  namespace: plexure
data:
  config.yaml: |
    api:
      dir: /var/lib/plexure/apis
      formats: [yaml, json]
      recursive: true
    # ... rest of config
```

3. Create secret:
```bash
kubectl create secret generic plexure-secrets \
  --from-literal=PINECONE_API_KEY=your_api_key \
  --from-literal=PINECONE_ENV=production \
  --namespace plexure
```

4. Create deployment:
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: plexure
  namespace: plexure
spec:
  replicas: 3
  selector:
    matchLabels:
      app: plexure
  template:
    metadata:
      labels:
        app: plexure
    spec:
      containers:
      - name: plexure
        image: plexure-api-search:latest
        ports:
        - containerPort: 8000
        volumeMounts:
        - name: config
          mountPath: /etc/plexure
        - name: apis
          mountPath: /var/lib/plexure/apis
        - name: cache
          mountPath: /var/cache/plexure
        envFrom:
        - secretRef:
            name: plexure-secrets
        resources:
          requests:
            cpu: 1
            memory: 2Gi
          limits:
            cpu: 2
            memory: 4Gi
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 30
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 10
      volumes:
      - name: config
        configMap:
          name: plexure-config
      - name: apis
        persistentVolumeClaim:
          claimName: plexure-apis
      - name: cache
        persistentVolumeClaim:
          claimName: plexure-cache
```

5. Create service:
```yaml
apiVersion: v1
kind: Service
metadata:
  name: plexure
  namespace: plexure
spec:
  selector:
    app: plexure
  ports:
  - port: 80
    targetPort: 8000
  type: LoadBalancer
```

6. Apply configuration:
```bash
kubectl apply -f k8s/
```

## Monitoring Setup

### Prometheus Integration

1. Create Prometheus config:
```yaml
scrape_configs:
  - job_name: 'plexure'
    scrape_interval: 15s
    static_configs:
      - targets: ['localhost:8000']
```

2. Configure metrics endpoint:
```yaml
monitoring:
  prometheus:
    enabled: true
    port: 8000
    path: /metrics
```

### Grafana Dashboard

1. Import dashboard:
```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d @dashboards/plexure.json \
  http://localhost:3000/api/dashboards/db
```

2. Configure alerts:
```yaml
alerting:
  rules:
  - alert: HighErrorRate
    expr: rate(plexure_errors_total[5m]) > 0.1
    for: 5m
    labels:
      severity: warning
    annotations:
      summary: High error rate detected
```

## Backup and Recovery

### Backup Configuration

1. Create backup script:
```bash
#!/bin/bash
BACKUP_DIR="/var/backups/plexure"
DATE=$(date +%Y%m%d_%H%M%S)

# Backup configuration
tar czf $BACKUP_DIR/config_$DATE.tar.gz /etc/plexure

# Backup API files
tar czf $BACKUP_DIR/apis_$DATE.tar.gz /var/lib/plexure/apis

# Backup vector store
plexure-api-search backup \
  --output $BACKUP_DIR/vectors_$DATE.dump
```

2. Schedule backup:
```bash
echo "0 2 * * * /usr/local/bin/backup-plexure.sh" \
  | crontab -u plexure -
```

### Recovery Procedure

1. Restore configuration:
```bash
tar xzf config_backup.tar.gz -C /
```

2. Restore API files:
```bash
tar xzf apis_backup.tar.gz -C /
```

3. Restore vector store:
```bash
plexure-api-search restore \
  --input vectors_backup.dump
```

## Scaling

### Horizontal Scaling

1. Configure load balancer:
```nginx
upstream plexure {
    server plexure1:8000;
    server plexure2:8000;
    server plexure3:8000;
}

server {
    listen 80;
    location / {
        proxy_pass http://plexure;
    }
}
```

2. Configure sharding:
```yaml
vector_store:
  shards: 3
  replicas: 2
```

### Vertical Scaling

1. Increase resources:
```yaml
resources:
  requests:
    cpu: 4
    memory: 8Gi
  limits:
    cpu: 8
    memory: 16Gi
```

2. Optimize batch size:
```yaml
model:
  batch_size: 64
```

## Security

### Network Security

1. Configure firewall:
```bash
ufw allow 8000/tcp
ufw enable
```

2. Enable TLS:
```yaml
server:
  tls:
    enabled: true
    cert: /etc/plexure/cert.pem
    key: /etc/plexure/key.pem
```

### Access Control

1. Configure authentication:
```yaml
auth:
  type: jwt
  secret: your_secret_key
  expiry: 3600
```

2. Configure RBAC:
```yaml
rbac:
  roles:
    admin:
      - "*"
    user:
      - "search:*"
      - "index:read"
```

## Maintenance

### Updates

1. Update package:
```bash
pip install -U plexure-api-search
```

2. Restart service:
```bash
systemctl restart plexure
```

### Monitoring

1. Check logs:
```bash
journalctl -u plexure -f
```

2. Check metrics:
```bash
curl http://localhost:8000/metrics
```

### Troubleshooting

1. Check service status:
```bash
systemctl status plexure
```

2. Check health:
```bash
curl http://localhost:8000/health
```

3. Debug mode:
```bash
LOG_LEVEL=DEBUG plexure-api-search serve
```

## Performance Tuning

### Memory Optimization

1. Configure cache:
```yaml
cache:
  type: redis
  max_size: 1GB
  ttl: 3600
```

2. Configure batch processing:
```yaml
processing:
  batch_size: 100
  max_workers: 4
```

### Query Optimization

1. Configure search:
```yaml
search:
  cache_results: true
  min_score: 0.5
  timeout: 5.0
```

2. Configure vector store:
```yaml
vector_store:
  index_type: hnsw
  ef_search: 100
  m: 16
```

## Disaster Recovery

### Failover

1. Configure replication:
```yaml
replication:
  enabled: true
  replicas: 3
  sync: true
```

2. Configure backup:
```yaml
backup:
  schedule: "0 2 * * *"
  retention: 7
  storage: s3
```

### Recovery

1. Restore from backup:
```bash
plexure-api-search restore \
  --backup latest \
  --target new-instance
```

2. Verify recovery:
```bash
plexure-api-search verify \
  --source backup \
  --target new-instance
```

## Support

### Documentation

1. Access documentation:
```bash
plexure-api-search docs
```

2. Generate documentation:
```bash
plexure-api-search docs generate
```

### Getting Help

1. Community support:
- GitHub Issues
- Discord channel
- Stack Overflow

2. Commercial support:
- Email: support@plexure.com
- Phone: +1-XXX-XXX-XXXX
- Web: https://support.plexure.com 