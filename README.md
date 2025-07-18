# KB Brain - Intelligent Knowledge Management System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/GPU-Accelerated-orange.svg" alt="GPU Support">
  <img src="https://img.shields.io/badge/MCP-Integrated-purple.svg" alt="MCP Integration">
  <img src="https://img.shields.io/badge/Performance-Optimized-blue.svg" alt="Performance">
</p>

## 🧠 **Overview**

KB Brain is an intelligent knowledge management system that provides powerful search capabilities, performance optimizations, and seamless integration with modern development tools. The system combines GPU-accelerated processing, intelligent search algorithms, and flexible deployment options to create a comprehensive knowledge solution.

## ⚡ **Key Features**

### 🔍 **Intelligent Search**
- **🚀 GPU-Accelerated Search** - CuPy-powered similarity computation with CPU fallback
- **🎯 Advanced Similarity Matching** - TF-IDF vectorization with cosine similarity
- **📊 Multi-Level Knowledge Base** - Hierarchical search across multiple sources
- **🔄 Performance Optimizations** - Intel scikit-learn extensions and Numba JIT compilation
- **⚡ Real-time Search** - Sub-second response times for knowledge retrieval

### 🛠️ **Developer Integration**
- **🔌 Continue Extension** - Seamless VS Code integration for code completion
- **🖥️ MCP Server** - Model Context Protocol support for AI assistants
- **🌐 REST API** - Comprehensive API for custom integrations
- **🐍 Python SDK** - Programmatic access to all KB Brain features
- **📱 CLI Tools** - Command-line interface for automation and scripting

### 🔒 **Security & Configuration**
- **🔐 Environment Variables** - Secure configuration management
- **🛡️ SSL Certificate Support** - Corporate network SSL handling
- **🔑 API Authentication** - Secure API key management
- **⚙️ Flexible Deployment** - Docker, Kubernetes, and bare metal support
- **📋 Configuration Validation** - Automated setup and validation tools

### 🏢 **Enterprise Ready**
- **🔧 Auto-Configuration** - Intelligent setup and certificate management
- **📊 Performance Monitoring** - Comprehensive metrics and benchmarking
- **📈 Scalable Architecture** - Horizontal scaling support
- **🔄 Knowledge Management** - Dynamic knowledge base updates and maintenance

## 🚀 **Quick Start**

### Installation
```bash
# Install from PyPI
pip install kb-brain

# Or install from source
git clone https://github.com/organization/kb-brain.git
cd kb-brain
pip install -e .

# Optional: GPU acceleration and performance optimizations
pip install -e .[gpu,performance]
```

### Initial Setup
```bash
# Run configuration setup
python3 scripts/setup_config.py

# Configure SSL certificates (if needed)
python3 scripts/configure_ssl.py

# Validate installation
python3 scripts/validate_integration_simple.py
```

### Basic Usage

#### Command Line Interface
```bash
# Check system status
kb-brain status

# Search the knowledge base
kb-brain search "SSL certificate issues"

# Interactive mode
kb-brain interactive

# Enable performance optimizations
kb-brain search "GPU optimization" --optimize-performance

# Run benchmarks
kb-brain benchmark
```

#### Python API
```python
from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain

# Initialize KB Brain with performance optimizations
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
    print(f"Similarity: {solution.similarity_score:.2f}")
```

#### REST API
```bash
# Start API server
kb-brain-continue start --port 8080 --optimize-performance

# Search via API
curl -X POST http://localhost:8080/kb-brain/search \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer your-api-key" \
  -d '{
    "query": "SSL certificate configuration",
    "max_results": 5
  }'
```

### Continue Extension (VS Code)
```bash
# Start Continue integration server
kb-brain-continue start --optimize-performance

# In VS Code, configure Continue extension:
# - Model: KB Brain
# - API Base: http://localhost:8080/kb-brain
# - API Key: your-api-key
```

## 📖 **Documentation**

### Core Documentation
- **[Getting Started](docs/getting-started.md)** - Complete setup and basic usage guide
- **[API Reference](docs/api.md)** - Comprehensive API documentation
- **[Configuration Guide](docs/configuration.md)** - Advanced configuration options
- **[Security Guide](docs/security.md)** - Security best practices and SSL setup

### Advanced Topics
- **[Performance Optimization](docs/performance.md)** - GPU acceleration and performance tuning
- **[Integration Examples](docs/integrations.md)** - Real-world integration patterns
- **[Troubleshooting](docs/troubleshooting.md)** - Common issues and solutions
- **[Contributing](docs/contributing.md)** - Development and contribution guidelines

### Developer Resources
- **[Python SDK](docs/python-sdk.md)** - Detailed Python API documentation
- **[CLI Reference](docs/cli-reference.md)** - Command-line interface documentation
- **[MCP Integration](docs/mcp-integration.md)** - Model Context Protocol setup
- **[Continue Extension](docs/continue-extension.md)** - VS Code integration guide

## 🛠️ **Requirements**

### Core Requirements
- **Python 3.8+**
- **scikit-learn >= 1.3.0**
- **numpy >= 1.21.0**
- **click >= 8.0.0** (for CLI)
- **rich >= 13.0.0** (for formatting)

### Optional Dependencies
- **CuPy** - GPU acceleration (requires CUDA 12.0+)
- **Intel Extension for Scikit-learn** - CPU performance optimization
- **Numba** - JIT compilation for faster computations
- **aiohttp** - Async HTTP server for API endpoints

### Installation Options
```bash
# Basic installation
pip install kb-brain

# With GPU acceleration
pip install kb-brain[gpu]

# With performance optimizations
pip install kb-brain[performance]

# Full installation with all features
pip install kb-brain[all]
```

## 🎯 **Use Cases**

### 🔍 **Knowledge Discovery**
- **Technical Problem Solving** - Find solutions to SSL, network, and system issues
- **Code Pattern Recognition** - Discover established coding patterns and practices
- **Research Assistance** - Locate relevant research papers and methodologies
- **Troubleshooting** - Quick access to proven solutions for common problems

### 🛠️ **Development Workflow**
- **Code Completion** - Context-aware code suggestions via Continue extension
- **Documentation Search** - Instant access to internal documentation and guides
- **Best Practices** - Automated recommendations for coding standards
- **Error Resolution** - Quick lookup of error solutions and debugging steps

### 🏢 **Enterprise Knowledge Management**
- **Corporate Knowledge Base** - Centralized repository for organizational knowledge
- **SSL Certificate Management** - Automated handling of corporate certificates
- **Security Compliance** - Secure configuration and deployment practices
- **Performance Monitoring** - Real-time metrics and optimization recommendations

### 🚀 **AI Assistant Enhancement**
- **Context-Aware Responses** - Provide relevant context to AI assistants
- **Domain Expertise** - Specialized knowledge for specific problem domains
- **Intelligent Routing** - Direct queries to appropriate knowledge sources
- **Continuous Learning** - System improves through usage and feedback

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- Open Source Community
- AI Integration Community
- CUDA/GPU Computing Community
- Continue Extension developers
- Model Context Protocol contributors

## 📞 **Support**

- 📧 Email: support@yourorg.com
- 🐛 Issues: [GitHub Issues](https://github.com/organization/kb-brain/issues)
- 📖 Documentation: [Project Documentation](docs/)
- 💬 Discussions: [GitHub Discussions](https://github.com/organization/kb-brain/discussions)

## 🎯 **Performance Benchmarks**

KB Brain delivers significant performance improvements:

| Feature | Standard | Optimized | Speedup |
|---------|----------|-----------|---------|
| **Search Query** | 0.156s | 0.089s | **1.75x** |
| **Similarity Computation** | 0.234s | 0.098s | **2.39x** |
| **Knowledge Indexing** | 12.5s | 4.2s | **2.98x** |
| **Batch Processing** | 45.2s | 15.8s | **2.86x** |

<div align="center">
<em>Benchmarks run on Intel i7-12700K with RTX 4080, 32GB RAM</em>
</div>

---

**Built with ❤️ by the NPS Southwest Network Collaboration - CHDN**
