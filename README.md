# KB Brain - Intelligent Knowledge Management System

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/GPU-Accelerated-orange.svg" alt="GPU Support">
  <img src="https://img.shields.io/badge/MCP-Integrated-purple.svg" alt="MCP Integration">
</p>

## 🧠 **Overview**

KB Brain is an intelligent knowledge management system that combines GPU-accelerated processing, MCP (Model Context Protocol) integration, and screen-based monitoring to provide powerful knowledge base capabilities for AI assistants like Claude.

## ⚡ **Key Features**

- **🚀 Hybrid GPU/CPU Processing** - Automatic GPU acceleration with CPU fallback
- **🔌 MCP Integration** - Seamless integration with Claude via MCP server
- **🖥️ Screen Monitoring** - Multi-worker screen sessions for long-running tasks
- **📊 Intelligent Search** - Advanced similarity search with machine learning
- **🔧 Auto-Optimization** - Startup optimization and persistent state management
- **🔒 Enterprise Ready** - AppLocker compatibility and SSL certificate handling

## 🚀 **Quick Start**

### Installation
```bash
# Clone the repository
git clone https://github.com/misley/kb-brain.git
cd kb-brain

# Install KB Brain
pip install -e .

# Optional: GPU acceleration
pip install -e .[gpu]
```

### Basic Usage
```python
from kb_brain.core import HybridGPUKBBrain

# Initialize KB Brain
brain = HybridGPUKBBrain()

# Search for solutions
solutions = brain.find_best_solution_hybrid("SSL certificate issues")

# Get system status
status = brain.get_system_status()
print(f"GPU Available: {status['hybrid_gpu_status']['gpu_available']}")
```

### MCP Integration
```bash
# Start MCP server
kb-brain-mcp

# Or use configuration file
kb-brain-mcp --config config/mcp_config.json
```

## 📖 **Documentation**

- [Installation Guide](docs/installation.md)
- [MCP Integration](docs/mcp_integration.md)
- [GPU Acceleration](docs/gpu_acceleration.md)
- [Screen Monitoring](docs/screen_monitoring.md)
- [API Reference](docs/api_reference.md)

## 🛠️ **Requirements**

- Python 3.8+
- scikit-learn >= 1.3.0
- numpy >= 1.21.0
- pandas >= 1.3.0

### Optional (GPU Acceleration)
- CUDA 12.0+
- CuPy
- CuML

## 🎯 **Use Cases**

- **AI Assistant Integration** - Enhanced AI assistant capabilities via MCP
- **Knowledge Base Management** - Intelligent search and retrieval
- **Long-Running Task Monitoring** - Screen-based progress tracking
- **GPU-Accelerated ML** - Fast similarity search and clustering
- **Enterprise Deployment** - Corporate network compatibility

## 🏢 **Enterprise Features**

- **AppLocker Compatibility** - Works in restricted Windows environments
- **SSL Certificate Handling** - Automatic DOI/corporate certificate management
- **Proxy Support** - Corporate network proxy configuration
- **Security Scanning** - Input validation and sanitization

## 🤝 **Contributing**

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- National Park Service Data Science Team
- AI Integration Community
- CUDA/GPU Computing Community

## 📞 **Support**

- 📧 Email: noreply@nps.gov
- 🐛 Issues: [GitHub Issues](https://github.com/misley/kb-brain/issues)
- 📖 Documentation: [GitHub Pages](https://misley.github.io/kb-brain/)

---

**Built with ❤️ by the NPS Data Science Team**
