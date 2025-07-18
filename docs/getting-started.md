# Getting Started with KB Brain

<div class="alert alert-info">
<strong>Welcome to KB Brain!</strong> This guide will help you get up and running with the intelligent knowledge base system in just a few minutes.
</div>

## Quick Start

### 1. Installation

#### Using pip (Recommended)
```bash
pip install kb-brain
```

#### From Source
```bash
git clone https://github.com/organization/kb-brain.git
cd kb-brain
pip install -e .
```

### 2. Configuration Setup

Run the configuration setup script to create your environment:

```bash
cd kb-brain
python3 scripts/setup_config.py
```

This will:
- Create a `.env` file from the template
- Set up required directories
- Generate configuration files
- Validate your setup

### 3. Test Your Installation

Verify everything is working:

```bash
# Check system status
kb-brain status

# Run a quick search
kb-brain search "test query"

# Validate configuration
python3 scripts/validate_integration_simple.py
```

## Basic Usage

### Command Line Interface

KB Brain provides a comprehensive CLI for all operations:

```bash
# Get help
kb-brain --help

# Check system status
kb-brain status

# Search the knowledge base
kb-brain search "SSL certificate issues"

# Interactive mode
kb-brain interactive

# Rebuild knowledge index
kb-brain rebuild-index

# Run performance benchmarks
kb-brain benchmark
```

### Python API

Use KB Brain programmatically in your Python code:

```python
from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain

# Initialize KB Brain
brain = HybridGPUKBBrain(enable_performance_optimizations=True)

# Search for solutions
solutions = brain.find_best_solution_hybrid(
    "How to fix SSL certificate issues?",
    top_k=5
)

# Process results
for solution in solutions:
    print(f"Solution: {solution.solution_text}")
    print(f"Confidence: {solution.confidence:.2f}")
    print(f"Source: {solution.source_kb}")
    print("---")
```

### REST API

Start the API server and make HTTP requests:

```bash
# Start the server
kb-brain-continue start --port 8080

# Search via API
curl -X POST http://localhost:8080/kb-brain/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "SSL certificate configuration",
    "max_results": 5
  }'
```

## Configuration

### Environment Variables

KB Brain uses environment variables for configuration. Key settings:

```bash
# Core paths
KB_SYSTEM_PATH=/path/to/kb_system
KB_BRAIN_PATH=/path/to/kb-brain
KB_DATA_PATH=/path/to/data

# Performance settings
KB_BRAIN_PERFORMANCE=true
KB_BRAIN_MAX_WORKERS=4

# API settings
KB_BRAIN_API_KEY=your-secure-api-key
KB_BRAIN_API_BASE=http://localhost:8080

# SSL settings (if needed)
SSL_CERT_PATH=/path/to/certificates.crt
SSL_CA_BUNDLE=/path/to/ca-bundle.crt
```

### Configuration Files

The system uses these configuration files:

- `.env` - Environment variables (create from `.env.template`)
- `config/settings.py` - Centralized configuration management
- `continue_integration/continue_config.json` - Continue extension settings
- `config/mcp_config.json` - MCP server configuration

## Features Overview

### 🔍 **Intelligent Search**
- GPU-accelerated similarity search
- Multi-level knowledge base querying
- Context-aware result ranking
- Performance-optimized retrieval

### 🚀 **Performance Optimizations**
- Intel scikit-learn extensions (2-10x speedup)
- Numba JIT compilation
- GPU acceleration with CuPy
- Automatic optimization detection

### 🔧 **Developer Integration**
- Continue extension for VS Code
- REST API for custom integrations
- Python SDK for programmatic access
- CLI tools for automation

### 🔒 **Security & Configuration**
- Environment variable configuration
- SSL certificate management
- API key authentication
- Secure deployment practices

## Common Use Cases

### 1. Technical Problem Solving
```bash
# Search for SSL issues
kb-brain search "SSL certificate not trusted"

# Find GPU optimization solutions
kb-brain search "GPU memory optimization CuPy"

# Debug network problems
kb-brain search "network timeout error"
```

### 2. Code Development
```python
# Get code completion suggestions
from kb_brain.intelligence.sme_agent_system import SMEAgentSystem

sme = SMEAgentSystem()
agent_id, response = await sme.route_query(
    "How to implement similarity search?",
    context={"domain": "programming"}
)
```

### 3. Knowledge Management
```python
# Add new knowledge
brain.add_knowledge(
    title="New SSL Configuration Method",
    content="Step-by-step guide to configure SSL...",
    tags=["ssl", "configuration", "security"]
)

# Update existing knowledge
brain.update_knowledge(
    id="kb_entry_123",
    content="Updated solution with new approach..."
)
```

## Integration Examples

### Continue Extension (VS Code)

1. **Install Continue extension** in VS Code
2. **Configure KB Brain** in Continue settings:
   ```json
   {
     "models": [{
       "title": "KB Brain",
       "provider": "custom",
       "apiBase": "http://localhost:8080/kb-brain",
       "apiKey": "your-api-key"
     }]
   }
   ```
3. **Use in VS Code**: Type `/kb` to search knowledge base

### MCP Integration (Claude Desktop)

1. **Configure MCP server** in Claude Desktop:
   ```json
   {
     "mcpServers": {
       "kb-brain": {
         "command": "python3",
         "args": ["path/to/kb_brain/mcp/server.py"]
       }
     }
   }
   ```
2. **Use in Claude**: KB Brain tools are automatically available

### Custom Integration

```python
import requests

class KBBrainClient:
    def __init__(self, api_key, base_url):
        self.api_key = api_key
        self.base_url = base_url
        self.headers = {
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json'
        }
    
    def search(self, query, max_results=10):
        response = requests.post(
            f"{self.base_url}/search",
            json={
                "query": query,
                "max_results": max_results
            },
            headers=self.headers
        )
        return response.json()

# Usage
client = KBBrainClient("your-api-key", "http://localhost:8080/kb-brain")
results = client.search("SSL configuration")
```

## Performance Tuning

### Enable Performance Optimizations

```bash
# Enable optimizations in CLI
kb-brain search "query" --optimize-performance

# Enable in Python
brain = HybridGPUKBBrain(enable_performance_optimizations=True)

# Enable in environment
export KB_BRAIN_PERFORMANCE=true
```

### GPU Acceleration

If you have CUDA-compatible GPU:

```bash
# Install GPU dependencies
pip install cupy-cuda12x

# Verify GPU availability
python3 -c "import cupy; print('GPU available:', cupy.cuda.is_available())"

# Use GPU-accelerated search
kb-brain search "query" --optimize-performance
```

### Performance Benchmarking

```bash
# Run performance benchmarks
kb-brain benchmark

# Test with different configurations
KB_BRAIN_PERFORMANCE=true kb-brain benchmark
KB_BRAIN_PERFORMANCE=false kb-brain benchmark
```

## Troubleshooting

### Common Issues

#### 1. SSL Certificate Errors
```bash
# Configure SSL certificates
python3 scripts/configure_ssl.py

# Test SSL configuration
curl -I https://github.com
```

#### 2. Performance Optimization Not Working
```bash
# Check dependencies
pip install intel-extension-for-scikit-learn numba

# Verify configuration
python3 -c "from kb_brain.performance.performance_integration import PerformanceManager; pm = PerformanceManager(); print(pm.get_optimization_status())"
```

#### 3. Knowledge Base Not Found
```bash
# Check paths
echo $KB_SYSTEM_PATH
echo $KB_DATA_PATH

# Create directories
python3 -c "from kb_brain.config.settings import Settings; Settings.ensure_directories()"
```

### Debug Mode

Enable debug logging for troubleshooting:

```bash
export KB_BRAIN_LOG_LEVEL=DEBUG
kb-brain search "query"
```

### Getting Help

- **Documentation**: Check the [full documentation](docs/)
- **Issues**: Report problems on [GitHub Issues](https://github.com/organization/kb-brain/issues)
- **Support**: Contact support@yourorg.com
- **Community**: Join our [discussion forum](https://github.com/organization/kb-brain/discussions)

## Next Steps

Now that you have KB Brain set up, explore these advanced features:

1. **[API Reference](api.md)** - Complete API documentation
2. **[Configuration Guide](configuration.md)** - Advanced configuration options
3. **[Performance Guide](performance.md)** - Optimization techniques
4. **[Integration Examples](integrations.md)** - Real-world integration patterns
5. **[Security Guide](security.md)** - Security best practices

<div class="alert alert-success">
<strong>🎉 Congratulations!</strong> You're now ready to use KB Brain for intelligent knowledge management. Start by searching for solutions to your technical problems or add your own knowledge to build a comprehensive knowledge base.
</div>

---

<div class="footer">
<p>KB Brain Getting Started Guide • Version 1.1.0 • Last Updated: January 15, 2024</p>
</div>