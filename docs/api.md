# KB Brain API Documentation

<div class="alert alert-info">
<strong>Version:</strong> 1.1.0<br>
<strong>Base URL:</strong> <code>http://localhost:8080/kb-brain</code><br>
<strong>Authentication:</strong> API Key required
</div>

## Overview

The KB Brain API provides programmatic access to the knowledge base search and management system. It supports both REST and streaming endpoints for various knowledge operations.

## Authentication

All API requests require authentication using an API key in the request headers:

```http
Authorization: Bearer your-api-key-here
X-KB-Brain-API-Key: your-api-key-here
```

<div class="alert alert-warning">
<strong>Security Note:</strong> Always use HTTPS in production and keep your API keys secure.
</div>

## Base Endpoints

### Health Check
Check if the API is running and accessible.

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "version": "1.1.0",
  "timestamp": "2024-01-15T10:30:00Z",
  "uptime": "2h 15m 30s"
}
```

### System Status
Get comprehensive system status information.

```http
GET /status
```

**Response:**
```json
{
  "system": {
    "status": "operational",
    "gpu_available": true,
    "performance_optimizations": true,
    "total_solutions": 1247,
    "knowledge_embeddings": 5621,
    "last_updated": "2024-01-15T10:25:00Z"
  },
  "performance": {
    "cpu_usage": 45.2,
    "memory_usage": 67.8,
    "disk_usage": 23.1,
    "avg_response_time": 0.125
  }
}
```

## Knowledge Search API

### Basic Search
Search the knowledge base for relevant solutions.

```http
POST /search
```

**Request Body:**
```json
{
  "query": "SSL certificate issues",
  "max_results": 10,
  "similarity_threshold": 0.3,
  "include_metadata": true,
  "search_depth": "deep"
}
```

**Parameters:**
- `query` (string, required): Search query text
- `max_results` (integer, optional): Maximum number of results (default: 10)
- `similarity_threshold` (float, optional): Minimum similarity score (default: 0.3)
- `include_metadata` (boolean, optional): Include metadata in results (default: true)
- `search_depth` (string, optional): Search depth level (`shallow`, `deep`, `comprehensive`)

**Response:**
```json
{
  "query": "SSL certificate issues",
  "results": [
    {
      "id": "ssl_cert_problem_001",
      "title": "SSL Certificate Configuration",
      "content": "Configure SSL certificates for corporate network...",
      "similarity_score": 0.89,
      "confidence": 0.92,
      "source": "troubleshooting_kb",
      "tags": ["ssl", "certificates", "network"],
      "last_updated": "2024-01-10T14:30:00Z",
      "metadata": {
        "success_rate": 0.85,
        "complexity": "medium",
        "estimated_time": "15 minutes"
      }
    }
  ],
  "total_found": 5,
  "search_time": 0.087,
  "processing_method": "gpu_accelerated"
}
```

### Advanced Search
Perform advanced search with multiple parameters and filters.

```http
POST /search/advanced
```

**Request Body:**
```json
{
  "query": "GPU memory optimization",
  "filters": {
    "tags": ["performance", "gpu"],
    "source": ["technical_kb", "performance_kb"],
    "date_range": {
      "start": "2024-01-01T00:00:00Z",
      "end": "2024-01-15T23:59:59Z"
    },
    "complexity": ["medium", "high"],
    "success_rate": {
      "min": 0.7,
      "max": 1.0
    }
  },
  "options": {
    "max_results": 20,
    "similarity_threshold": 0.4,
    "include_context": true,
    "rank_by": "relevance"
  }
}
```

**Response:**
```json
{
  "query": "GPU memory optimization",
  "results": [
    {
      "id": "gpu_mem_opt_001",
      "title": "CuPy Memory Pool Management",
      "content": "Use CuPy memory pool to optimize GPU memory usage...",
      "similarity_score": 0.94,
      "confidence": 0.88,
      "context": {
        "problem": "GPU memory exhaustion during large operations",
        "solution": "Implement memory pool management",
        "code_example": "import cupy as cp\npool = cp.get_default_memory_pool()"
      },
      "metadata": {
        "success_rate": 0.91,
        "complexity": "high",
        "estimated_time": "30 minutes",
        "dependencies": ["cupy", "cuda"]
      }
    }
  ],
  "filters_applied": {
    "tags": ["performance", "gpu"],
    "source": ["technical_kb"],
    "results_filtered": 12
  },
  "search_time": 0.156,
  "total_found": 8
}
```

## Knowledge Management API

### Add Knowledge
Add new knowledge entries to the knowledge base.

```http
POST /knowledge
```

**Request Body:**
```json
{
  "title": "New SSL Configuration Method",
  "content": "Alternative method for SSL certificate configuration...",
  "tags": ["ssl", "configuration", "alternative"],
  "metadata": {
    "complexity": "medium",
    "estimated_time": "20 minutes",
    "success_rate": 0.85
  },
  "source": "user_contributions",
  "category": "technical"
}
```

**Response:**
```json
{
  "id": "kb_entry_12345",
  "status": "created",
  "message": "Knowledge entry added successfully",
  "embedding_generated": true,
  "indexed": true,
  "created_at": "2024-01-15T10:30:00Z"
}
```

### Update Knowledge
Update existing knowledge entries.

```http
PUT /knowledge/{id}
```

**Request Body:**
```json
{
  "title": "Updated SSL Configuration Method",
  "content": "Updated method with new security considerations...",
  "tags": ["ssl", "configuration", "security", "updated"],
  "metadata": {
    "complexity": "medium",
    "estimated_time": "25 minutes",
    "success_rate": 0.88
  }
}
```

**Response:**
```json
{
  "id": "kb_entry_12345",
  "status": "updated",
  "message": "Knowledge entry updated successfully",
  "embedding_regenerated": true,
  "reindexed": true,
  "updated_at": "2024-01-15T10:35:00Z"
}
```

### Delete Knowledge
Remove knowledge entries from the knowledge base.

```http
DELETE /knowledge/{id}
```

**Response:**
```json
{
  "id": "kb_entry_12345",
  "status": "deleted",
  "message": "Knowledge entry deleted successfully",
  "embedding_removed": true,
  "indexed_removed": true,
  "deleted_at": "2024-01-15T10:40:00Z"
}
```

## Continue Integration API

### Code Completion
Get code completion suggestions using knowledge base context.

```http
POST /continue/completion
```

**Request Body:**
```json
{
  "code_snippet": "import cupy as cp\n# Optimize GPU memory",
  "language": "python",
  "cursor_position": 35,
  "context": {
    "file_path": "/src/gpu_optimization.py",
    "project_context": "machine learning",
    "intent": "completion"
  }
}
```

**Response:**
```json
{
  "completions": [
    {
      "text": "pool = cp.get_default_memory_pool()\npool.set_limit(size=2**30)  # 1GB limit",
      "confidence": 0.89,
      "kb_source": "gpu_optimization_kb",
      "explanation": "GPU memory pool management for efficient allocation"
    }
  ],
  "kb_matches": 3,
  "processing_time": 0.045
}
```

### Code Explanation
Get explanations for code snippets using knowledge base context.

```http
POST /continue/explain
```

**Request Body:**
```json
{
  "code_snippet": "vectorizer = TfidfVectorizer(max_features=5000, stop_words='english')",
  "language": "python",
  "context": {
    "file_path": "/src/text_processing.py",
    "intent": "explanation"
  }
}
```

**Response:**
```json
{
  "explanation": "Creates a TF-IDF vectorizer for text processing with 5000 maximum features and English stop words removed. This is commonly used for document similarity calculations.",
  "kb_context": [
    {
      "source": "text_processing_kb",
      "content": "TF-IDF vectorization best practices for knowledge base search",
      "relevance": 0.92
    }
  ],
  "related_concepts": ["text preprocessing", "document similarity", "feature extraction"],
  "confidence": 0.87
}
```

## Performance API

### Performance Metrics
Get system performance metrics and optimization status.

```http
GET /performance/metrics
```

**Response:**
```json
{
  "current_performance": {
    "avg_search_time": 0.125,
    "gpu_utilization": 67.3,
    "memory_usage": 45.2,
    "cache_hit_rate": 0.78,
    "throughput": 156.7
  },
  "optimizations": {
    "intel_extensions": {
      "enabled": true,
      "speedup_factor": 2.3,
      "status": "active"
    },
    "numba_jit": {
      "enabled": true,
      "speedup_factor": 1.8,
      "status": "active"
    },
    "gpu_acceleration": {
      "enabled": true,
      "cuda_version": "12.1",
      "status": "active"
    }
  },
  "recommendations": [
    "Consider increasing batch size for better GPU utilization",
    "Enable caching for frequently accessed embeddings"
  ]
}
```

### Performance Testing
Run performance tests and benchmarks.

```http
POST /performance/test
```

**Request Body:**
```json
{
  "test_type": "search_benchmark",
  "parameters": {
    "query_count": 100,
    "concurrent_requests": 10,
    "enable_optimizations": true
  }
}
```

**Response:**
```json
{
  "test_id": "perf_test_001",
  "status": "completed",
  "results": {
    "total_queries": 100,
    "avg_response_time": 0.089,
    "min_response_time": 0.023,
    "max_response_time": 0.234,
    "throughput": 178.5,
    "success_rate": 0.99,
    "error_rate": 0.01
  },
  "comparison": {
    "with_optimizations": 0.089,
    "without_optimizations": 0.156,
    "speedup_factor": 1.75
  },
  "completed_at": "2024-01-15T10:45:00Z"
}
```

## Error Handling

### Error Response Format
All API errors follow a consistent format:

```json
{
  "error": {
    "code": "INVALID_REQUEST",
    "message": "The request is invalid or malformed",
    "details": {
      "field": "query",
      "reason": "Query parameter is required"
    },
    "request_id": "req_12345",
    "timestamp": "2024-01-15T10:30:00Z"
  }
}
```

### Common Error Codes

| Code | Status | Description |
|------|--------|-------------|
| `INVALID_REQUEST` | 400 | Request is malformed or missing required parameters |
| `UNAUTHORIZED` | 401 | Invalid or missing API key |
| `FORBIDDEN` | 403 | Insufficient permissions for the requested operation |
| `NOT_FOUND` | 404 | Requested resource not found |
| `RATE_LIMITED` | 429 | Too many requests, rate limit exceeded |
| `INTERNAL_ERROR` | 500 | Internal server error |
| `SERVICE_UNAVAILABLE` | 503 | Service temporarily unavailable |

## Rate Limiting

The API implements rate limiting to ensure fair usage:

- **Default Limits**: 100 requests per minute per API key
- **Burst Limits**: 10 requests per second
- **Headers**: Rate limit information is included in response headers

```http
X-RateLimit-Limit: 100
X-RateLimit-Remaining: 87
X-RateLimit-Reset: 1642248000
```

## WebSocket API

### Real-time Search
Establish WebSocket connection for real-time search updates.

```javascript
const ws = new WebSocket('ws://localhost:8080/ws/search');

ws.onopen = function() {
    ws.send(JSON.stringify({
        type: 'search',
        query: 'SSL configuration',
        stream: true
    }));
};

ws.onmessage = function(event) {
    const data = JSON.parse(event.data);
    console.log('Search result:', data);
};
```

## SDK Examples

### Python SDK
```python
from kb_brain_client import KBBrainClient

client = KBBrainClient(
    api_key="your-api-key",
    base_url="http://localhost:8080/kb-brain"
)

# Search knowledge base
results = client.search("SSL certificate issues", max_results=5)

# Add knowledge
client.add_knowledge(
    title="New Solution",
    content="Solution content...",
    tags=["ssl", "security"]
)
```

### JavaScript SDK
```javascript
import { KBBrainClient } from 'kb-brain-js';

const client = new KBBrainClient({
    apiKey: 'your-api-key',
    baseUrl: 'http://localhost:8080/kb-brain'
});

// Search with async/await
const results = await client.search('GPU optimization', {
    maxResults: 10,
    includeMetadata: true
});

// Stream search results
client.searchStream('performance tuning', (result) => {
    console.log('New result:', result);
});
```

## Testing

### API Testing Tools
- **Postman Collection**: Available in `/docs/postman_collection.json`
- **OpenAPI Spec**: Available at `/docs/openapi.yaml`
- **Test Suite**: Run with `npm test` or `pytest tests/`

### Example Test Request
```bash
curl -X POST http://localhost:8080/kb-brain/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "SSL configuration",
    "max_results": 5
  }'
```

## Support

For API support and questions:
- **Documentation**: [API Reference](https://docs.kb-brain.com/api)
- **Issues**: [GitHub Issues](https://github.com/organization/kb-brain/issues)
- **Support**: support@yourorg.com

---

<div class="footer">
<p>KB Brain API Documentation • Version 1.1.0 • Last Updated: January 15, 2024</p>
</div>