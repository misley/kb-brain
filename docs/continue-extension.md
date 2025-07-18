# Continue Extension Integration

<div class="alert alert-info">
<strong>Quick Setup:</strong> Get KB Brain working with Continue extension in VS Code in under 5 minutes.
</div>

## Quick Start

### 1. Install Continue Extension
```bash
# Install Continue extension in VS Code
code --install-extension continue.continue
```

### 2. Start KB Brain Server
```bash
# Start KB Brain Continue integration server
kb-brain-continue start --port 8080 --optimize-performance
```

### 3. Configure Continue
Open VS Code settings and add KB Brain configuration:

**Method 1: VS Code Settings UI**
1. Open VS Code Settings (Ctrl+,)
2. Search for "Continue"
3. Add KB Brain model configuration

**Method 2: Configuration File**
Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "KB Brain",
      "provider": "custom",
      "model": "kb-brain-hybrid",
      "apiBase": "http://localhost:8080/kb-brain",
      "apiKey": "your-api-key-here",
      "systemMessage": "You are an AI assistant with access to a comprehensive knowledge base. Provide solutions based on proven patterns and organizational knowledge."
    }
  ]
}
```

### 4. Test Integration
```python
# In VS Code, try typing:
# /kb SSL certificate issues
# Continue should provide KB Brain-enhanced suggestions
```

## Detailed Configuration

### Complete Continue Configuration

```json
{
  "models": [
    {
      "title": "KB Brain",
      "provider": "custom",
      "model": "kb-brain-hybrid",
      "apiBase": "http://localhost:8080/kb-brain",
      "apiKey": "your-api-key-here",
      "systemMessage": "You are an AI assistant with access to a comprehensive knowledge base. Use KB Brain to find relevant solutions, code patterns, and context for development tasks. Always provide specific, actionable solutions based on proven patterns.",
      "requestOptions": {
        "headers": {
          "X-KB-Brain-Integration": "continue",
          "X-KB-Brain-Performance": "true"
        }
      }
    }
  ],
  "customCommands": [
    {
      "name": "kb-search",
      "description": "Search KB Brain knowledge base",
      "prompt": "Search the knowledge base for: {input}\n\nProvide specific solutions and code examples."
    },
    {
      "name": "kb-debug",
      "description": "Find debugging solutions",
      "prompt": "Find debugging solutions for this error: {input}\n\nSearch for similar errors and proven solutions. Include code examples."
    },
    {
      "name": "kb-explain",
      "description": "Explain code with KB context",
      "prompt": "Explain this code using knowledge base context: {input}\n\nProvide detailed explanations and related patterns."
    },
    {
      "name": "kb-refactor",
      "description": "Get refactoring suggestions",
      "prompt": "Suggest refactoring improvements for: {input}\n\nSearch for better patterns and best practices."
    },
    {
      "name": "kb-security",
      "description": "Security analysis",
      "prompt": "Analyze security aspects of: {input}\n\nProvide security recommendations from knowledge base."
    },
    {
      "name": "kb-performance",
      "description": "Performance optimization",
      "prompt": "Suggest performance optimizations for: {input}\n\nInclude specific optimization techniques."
    }
  ],
  "slashCommands": [
    {
      "name": "kb",
      "description": "Quick KB Brain search",
      "run": "kb-search {input}"
    },
    {
      "name": "debug",
      "description": "Debug with KB Brain",
      "run": "kb-debug {input}"
    },
    {
      "name": "explain",
      "description": "Explain with KB Brain",
      "run": "kb-explain {input}"
    },
    {
      "name": "refactor",
      "description": "Refactor with KB Brain",
      "run": "kb-refactor {input}"
    },
    {
      "name": "secure",
      "description": "Security analysis",
      "run": "kb-security {input}"
    },
    {
      "name": "optimize",
      "description": "Performance optimization",
      "run": "kb-performance {input}"
    }
  ],
  "contextProviders": [
    {
      "name": "kb-brain",
      "description": "KB Brain Knowledge Base",
      "type": "custom",
      "config": {
        "serverUrl": "http://localhost:8080/kb-brain",
        "apiKey": "your-api-key-here",
        "timeout": 10000,
        "maxSuggestions": 5,
        "enablePerformanceOptimizations": true
      }
    }
  ],
  "tabAutocompleteOptions": {
    "useCopyBuffer": true,
    "maxPromptTokens": 2048,
    "prefixPercentage": 0.75,
    "suffixPercentage": 0.25,
    "maxSuffixPercentage": 0.4,
    "debounceDelay": 300
  },
  "allowAnonymousTelemetry": false,
  "enableExperimentalFeatures": true
}
```

## Usage Examples

### 1. Slash Commands

#### Quick Knowledge Search
```javascript
// In VS Code, use slash commands:
/kb SSL certificate configuration
/kb GPU memory optimization
/kb Python performance patterns
```

#### Debugging Help
```python
# When you encounter an error:
/debug TypeError: 'NoneType' object is not subscriptable
/debug CUDA out of memory error
/debug SSL certificate verification failed
```

#### Code Explanation
```python
# Select code and use:
/explain this similarity search implementation
/explain TfidfVectorizer configuration
/explain CuPy memory management
```

### 2. Context-Aware Code Completion

#### SSL Configuration
```python
# Type this in VS Code:
import ssl
context = ssl.create_default_context()

# Continue suggests based on KB Brain:
# context.check_hostname = False
# context.verify_mode = ssl.CERT_NONE
# Based on corporate network patterns
```

#### GPU Optimization
```python
# Type this:
import cupy as cp
pool = cp.get_default_memory_pool()

# Continue suggests:
# pool.set_limit(size=2**30)  # 1GB limit
# pool.free_all_blocks()
# Based on KB Brain GPU patterns
```

#### Performance Patterns
```python
# Type this:
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(

# Continue suggests:
# max_features=5000,
# stop_words='english',
# ngram_range=(1, 2)
# Based on KB Brain performance knowledge
```

### 3. Interactive Chat

#### Problem Solving
```
You: How do I fix SSL certificate issues in a corporate network?

KB Brain (via Continue): Based on our knowledge base, here are the proven solutions:

1. **Configure Custom Certificate Bundle**:
   ```bash
   export SSL_CERT_PATH=/path/to/corporate/ca-certificates.crt
   export SSL_CA_BUNDLE=/path/to/ca-bundle.crt
   ```

2. **Python SSL Context**:
   ```python
   import ssl
   context = ssl.create_default_context()
   context.load_verify_locations('/path/to/corporate/ca-certificates.crt')
   ```

3. **Use KB Brain's SSL Configuration Script**:
   ```bash
   python3 scripts/configure_ssl.py
   ```

This approach has a 90% success rate based on our knowledge base.
```

#### Code Review
```
You: Can you review this similarity search implementation?

KB Brain (via Continue): Based on our performance optimization knowledge base, here's the analysis:

**Current Implementation**: Standard scikit-learn approach
**Performance**: ~0.156s per query (baseline)

**Recommended Optimizations**:
1. **Enable Performance Optimizations**:
   ```python
   brain = HybridGPUKBBrain(enable_performance_optimizations=True)
   ```

2. **Use GPU Acceleration**:
   ```python
   # 10-100x speedup for large datasets
   distances, indices = brain._optimized_similarity_computation(
       query_vector, embeddings_matrix, top_k=10
   )
   ```

**Expected Results**: ~0.089s per query (1.75x speedup)
```

## Advanced Features

### 1. Custom Context Providers

#### Create KB Brain Context Provider
```json
{
  "contextProviders": [
    {
      "name": "kb-brain-project",
      "description": "Project-Specific KB Brain Context",
      "type": "custom",
      "config": {
        "serverUrl": "http://localhost:8080/kb-brain",
        "apiKey": "your-api-key-here",
        "projectFilter": "current-project",
        "domainFilter": "technical",
        "maxResults": 3
      }
    }
  ]
}
```

### 2. Custom Model Configuration

#### Performance-Optimized Model
```json
{
  "models": [
    {
      "title": "KB Brain Performance",
      "provider": "custom",
      "model": "kb-brain-performance",
      "apiBase": "http://localhost:8080/kb-brain",
      "apiKey": "your-api-key-here",
      "systemMessage": "You are a performance optimization expert with access to proven optimization patterns. Focus on measurable improvements and specific techniques.",
      "requestOptions": {
        "headers": {
          "X-KB-Brain-Performance": "true",
          "X-KB-Brain-Domain": "performance"
        }
      }
    }
  ]
}
```

### 3. Workspace-Specific Configuration

#### Project-Specific Settings
```json
// .vscode/continue.json (project-specific)
{
  "models": [
    {
      "title": "KB Brain Project",
      "provider": "custom",
      "model": "kb-brain-hybrid",
      "apiBase": "http://localhost:8080/kb-brain",
      "apiKey": "${env:KB_BRAIN_API_KEY}",
      "systemMessage": "You are an AI assistant specialized in this project's patterns and practices. Use the knowledge base to provide project-specific solutions."
    }
  ],
  "customCommands": [
    {
      "name": "project-patterns",
      "description": "Find project-specific patterns",
      "prompt": "Find patterns specific to this project: {input}\n\nFocus on established project conventions."
    }
  ]
}
```

## Performance Optimization

### 1. Enable Performance Features

#### Server Configuration
```bash
# Start with performance optimizations
kb-brain-continue start --port 8080 --optimize-performance

# Monitor performance
kb-brain-continue monitor --port 8080 --interval 5
```

#### Client Configuration
```json
{
  "tabAutocompleteOptions": {
    "maxPromptTokens": 2048,
    "debounceDelay": 200,
    "useCopyBuffer": true
  },
  "contextProviders": [
    {
      "name": "kb-brain",
      "config": {
        "timeout": 5000,
        "maxSuggestions": 5,
        "enableCaching": true
      }
    }
  ]
}
```

### 2. Optimize Response Times

#### Caching Strategy
```json
{
  "kb-brain": {
    "cacheResponses": true,
    "cacheTimeout": 300,
    "maxCacheSize": 1000
  }
}
```

#### Batch Processing
```json
{
  "kb-brain": {
    "batchRequests": true,
    "batchSize": 5,
    "batchTimeout": 1000
  }
}
```

## Troubleshooting

### Common Issues

#### 1. Continue Not Connecting to KB Brain
```bash
# Check if KB Brain server is running
curl -X GET http://localhost:8080/kb-brain/status

# Expected response:
{
  "status": "healthy",
  "version": "1.1.0",
  "performance_optimizations": true
}
```

#### 2. No Suggestions Appearing
```bash
# Check Continue configuration
cat ~/.continue/config.json | jq '.models[] | select(.title == "KB Brain")'

# Verify API key
echo $KB_BRAIN_API_KEY
```

#### 3. Slow Response Times
```bash
# Enable performance optimizations
export KB_BRAIN_PERFORMANCE=true

# Check system resources
kb-brain status --optimize-performance

# Monitor Continue server
kb-brain-continue monitor --port 8080
```

### Debug Mode

#### Enable Debug Logging
```json
{
  "kb-brain": {
    "debug": true,
    "logLevel": "debug",
    "enableRequestLogging": true
  }
}
```

#### Debug Commands
```bash
# Test KB Brain API directly
curl -X POST http://localhost:8080/kb-brain/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{"query": "test", "max_results": 1}'

# Check Continue server logs
kb-brain-continue status --verbose
```

## Best Practices

### 1. Configuration Management
- Use environment variables for API keys
- Create project-specific Continue configurations
- Version control your Continue settings (without API keys)

### 2. Performance Optimization
- Enable performance optimizations
- Use appropriate debounce delays
- Limit result counts for better performance
- Monitor resource usage

### 3. Security
- Store API keys securely
- Use HTTPS in production
- Rotate API keys regularly
- Validate SSL certificates

### 4. Workflow Integration
- Use specific search terms for better results
- Leverage slash commands for quick access
- Combine with other VS Code extensions
- Regular KB Brain knowledge updates

## Next Steps

1. **Set up basic Continue integration** using the quick start guide
2. **Explore slash commands** for common development tasks
3. **Customize configurations** for your specific project needs
4. **Monitor performance** and optimize as needed
5. **Integrate with other tools** like GitHub Copilot for maximum benefit

---

<div class="footer">
<p>Continue Extension Guide • KB Brain Version 1.1.0 • Last Updated: January 15, 2024</p>
</div>