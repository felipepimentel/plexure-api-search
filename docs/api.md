# API Documentation

## Overview

The Plexure API Search provides a powerful search engine for API contracts. This document describes all available endpoints and their usage.

## Base URL

All API endpoints are relative to the base URL:

```
http://localhost:8000/api/v1
```

## Authentication

Authentication is required for all endpoints. Use the `Authorization` header with a Bearer token:

```
Authorization: Bearer <your_token>
```

## Endpoints

### Search

#### Search API Contracts

```http
GET /search
```

Search for API contracts using natural language queries.

**Parameters:**

- `q` (string, required) - Search query
- `limit` (integer, optional) - Maximum number of results (default: 10)
- `offset` (integer, optional) - Result offset for pagination (default: 0)
- `filters` (object, optional) - Search filters
  - `method` (string) - HTTP method filter
  - `path` (string) - Path filter
  - `tags` (array) - Tag filters
- `context` (object, optional) - Search context
  - `user_id` (string) - User identifier
  - `domain` (string) - Business domain
  - `features` (array) - Enabled features

**Example Request:**

```http
GET /search?q=create user&limit=5
Authorization: Bearer <token>
```

**Example Response:**

```json
{
  "results": [
    {
      "method": "POST",
      "path": "/users",
      "description": "Create a new user",
      "parameters": [...],
      "responses": [...],
      "score": 0.95,
      "metadata": {
        "source": "users.yaml",
        "version": "1.0.0"
      }
    },
    ...
  ],
  "total": 42,
  "took": 0.125
}
```

### Indexing

#### Index API Files

```http
POST /index
```

Index API contract files.

**Parameters:**

- `path` (string, optional) - Path to API files (default: configured API directory)
- `force` (boolean, optional) - Force reindexing (default: false)
- `async` (boolean, optional) - Run indexing asynchronously (default: false)

**Example Request:**

```http
POST /index
Authorization: Bearer <token>
Content-Type: application/json

{
  "path": "/path/to/apis",
  "force": true
}
```

**Example Response:**

```json
{
  "job_id": "abc123",
  "status": "started",
  "message": "Indexing started"
}
```

#### Get Index Status

```http
GET /index/status
```

Get current indexing status.

**Example Response:**

```json
{
  "status": "running",
  "progress": 75,
  "total_files": 100,
  "processed_files": 75,
  "errors": [],
  "started_at": "2024-01-01T12:00:00Z",
  "updated_at": "2024-01-01T12:01:00Z"
}
```

### Monitoring

#### Get Health Status

```http
GET /health
```

Get system health status.

**Example Response:**

```json
{
  "status": "healthy",
  "components": {
    "search": {
      "status": "healthy",
      "latency": 0.050
    },
    "index": {
      "status": "healthy",
      "documents": 1000
    },
    "vector_store": {
      "status": "healthy",
      "vectors": 5000
    }
  },
  "version": "1.0.0"
}
```

#### Get Metrics

```http
GET /metrics
```

Get system metrics.

**Example Response:**

```json
{
  "search": {
    "requests": 1000,
    "latency_p95": 0.200,
    "errors": 5
  },
  "index": {
    "documents": 1000,
    "vectors": 5000,
    "last_update": "2024-01-01T12:00:00Z"
  },
  "resources": {
    "cpu_usage": 45.5,
    "memory_usage": 512.0,
    "disk_usage": 1024.0
  }
}
```

### Analytics

#### Get Search Analytics

```http
GET /analytics/search
```

Get search analytics data.

**Parameters:**

- `start` (string, optional) - Start time (ISO 8601)
- `end` (string, optional) - End time (ISO 8601)
- `interval` (string, optional) - Aggregation interval (default: 1h)

**Example Response:**

```json
{
  "total_queries": 1000,
  "unique_queries": 250,
  "avg_latency": 0.150,
  "success_rate": 0.995,
  "top_queries": [
    {
      "query": "create user",
      "count": 100,
      "avg_latency": 0.125
    },
    ...
  ],
  "trends": {
    "queries_per_hour": [...],
    "latency_p95": [...],
    "error_rate": [...]
  }
}
```

### Experiments

#### Create Experiment

```http
POST /experiments
```

Create new A/B test experiment.

**Request Body:**

```json
{
  "name": "ranking_algorithm",
  "description": "Test new ranking algorithm",
  "variants": [
    {
      "name": "control",
      "weight": 0.5,
      "config": {
        "algorithm": "default"
      }
    },
    {
      "name": "treatment",
      "weight": 0.5,
      "config": {
        "algorithm": "improved"
      }
    }
  ],
  "metrics": [
    "click_through_rate",
    "conversion_rate"
  ],
  "duration": "7d"
}
```

**Example Response:**

```json
{
  "experiment_id": "exp123",
  "status": "created",
  "start_time": "2024-01-01T12:00:00Z",
  "end_time": "2024-01-08T12:00:00Z"
}
```

#### Get Experiment Results

```http
GET /experiments/{experiment_id}/results
```

Get experiment results.

**Example Response:**

```json
{
  "experiment": {
    "name": "ranking_algorithm",
    "status": "running",
    "progress": 75
  },
  "results": {
    "control": {
      "users": 5000,
      "metrics": {
        "click_through_rate": 0.125,
        "conversion_rate": 0.050
      }
    },
    "treatment": {
      "users": 5000,
      "metrics": {
        "click_through_rate": 0.150,
        "conversion_rate": 0.075
      }
    }
  },
  "significance": {
    "click_through_rate": {
      "p_value": 0.001,
      "significant": true
    },
    "conversion_rate": {
      "p_value": 0.002,
      "significant": true
    }
  }
}
```

## Error Handling

The API uses standard HTTP status codes and returns error details in the response body:

```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid parameter value",
    "details": {
      "parameter": "limit",
      "reason": "Must be between 1 and 100"
    }
  }
}
```

Common status codes:

- `200 OK` - Success
- `400 Bad Request` - Invalid parameters
- `401 Unauthorized` - Missing or invalid authentication
- `403 Forbidden` - Insufficient permissions
- `404 Not Found` - Resource not found
- `429 Too Many Requests` - Rate limit exceeded
- `500 Internal Server Error` - Server error

## Rate Limiting

The API implements rate limiting per user and endpoint. Limits are included in response headers:

```
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 95
X-RateLimit-Reset: 1640995200
```

When rate limit is exceeded, the API returns `429 Too Many Requests` with a `Retry-After` header.

## Pagination

List endpoints support pagination using `limit` and `offset` parameters. Response includes pagination metadata:

```json
{
  "results": [...],
  "pagination": {
    "total": 100,
    "limit": 10,
    "offset": 0,
    "next": "/api/v1/search?q=test&limit=10&offset=10",
    "previous": null
  }
}
```

## Versioning

The API is versioned using URL path prefix (e.g. `/api/v1`). Breaking changes will be released in new API versions.

## SDKs

Official SDKs are available for:

- Python: [plexure-api-search-python](https://github.com/plexure/plexure-api-search-python)
- JavaScript: [plexure-api-search-js](https://github.com/plexure/plexure-api-search-js)

## Support

For API support and bug reports, please open an issue on GitHub or contact support@plexure.com. 