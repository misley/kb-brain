# KB Brain - Intelligent Knowledge Management System with SME Agents

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue.svg" alt="Python Version">
  <img src="https://img.shields.io/badge/License-MIT-green.svg" alt="License">
  <img src="https://img.shields.io/badge/GPU-Accelerated-orange.svg" alt="GPU Support">
  <img src="https://img.shields.io/badge/MCP-Integrated-purple.svg" alt="MCP Integration">
  <img src="https://img.shields.io/badge/SME%20Agents-Hierarchical-green.svg" alt="SME Agents">
</p>

## 🧠 **Overview**

KB Brain is an advanced intelligent knowledge management system featuring **hierarchical SME (Subject Matter Expert) agents** that automatically specialize as knowledge grows. The system combines GPU-accelerated processing, intelligent prompt routing, and autonomous agent spawning to create a scalable knowledge ecosystem.

## ⚡ **Key Features**

### 🤖 **SME Agent System**
- **🧬 Hierarchical Agents** - Parent-child agent relationships with automatic specialization
- **🎯 Domain Expertise** - Agents automatically specialize in specific knowledge domains
- **📡 Inter-Agent Communication** - Sophisticated messaging protocol for collaboration
- **⚖️ Intelligent Routing** - Queries routed to most appropriate expert agent
- **📈 Adaptive Learning** - Agents evolve expertise based on performance and usage

### 🚀 **Core Intelligence**
- **🔍 Intelligent Prompt Processing** - Advanced classification and routing system
- **🧠 Hybrid GPU/CPU Processing** - Automatic GPU acceleration with CPU fallback
- **🔌 MCP Integration** - Seamless integration with AI assistants via MCP server
- **🖥️ Screen Monitoring** - Multi-worker screen sessions for long-running tasks
- **📊 Advanced Search** - Cross-repository knowledge search with similarity matching
- **🔄 Knowledge Ingestion** - Automatic capture and integration of solutions

### 🏢 **Enterprise Features**
- **🔧 Auto-Consolidation** - Intelligent knowledge base consolidation and partitioning
- **🔒 Security** - AppLocker compatibility and SSL certificate handling
- **🌐 Proxy Support** - Corporate network configuration
- **📋 Monitoring** - Comprehensive system status and performance tracking

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

#### SME Intelligence System
```python
from kb_brain.intelligence import KBBrainIntelligence

# Initialize intelligent system with SME agents
system = KBBrainIntelligence()
await system.initialize()

# Process queries through SME routing
response = await system.process_query(
    "How do I fix SSL certificate issues in WSL?",
    context={"project": "dunes", "domain": "technical"}
)

print(f"Assigned to: {response['assigned_agent']}")
print(f"Response: {response['response']}")

# Get comprehensive system status
status = system.get_system_status()
print(f"Active SME agents: {status['system_stats']['sme_agents_active']}")
```

#### Core KB Brain
```python
from kb_brain.core import HybridGPUKBBrain

# Initialize core KB Brain
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

### Core Documentation
- [Installation Guide](docs/installation.md)
- [SME Agent System](docs/sme_agent_system.md)
- [Intelligence Components](docs/intelligence_system.md)
- [MCP Integration](docs/mcp_integration.md)
- [API Reference](docs/api_reference.md)

### Advanced Topics
- [GPU Acceleration](docs/gpu_acceleration.md)
- [Screen Monitoring](docs/screen_monitoring.md)
- [Knowledge Base Consolidation](docs/kb_consolidation.md)
- [Agent Communication Protocol](docs/communication_protocol.md)

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

### 🤖 **Intelligent Knowledge Organization**
- **Automatic Specialization** - System spawns domain experts as knowledge grows
- **Cross-Domain Collaboration** - SME agents consult each other for complex problems
- **Institutional Memory** - Hierarchical knowledge preservation and evolution
- **Scalable Expertise** - Handle unlimited domains without performance degradation

### 🔧 **Development & Operations**
- **AI Assistant Enhancement** - Intelligent routing and specialized responses
- **Knowledge Base Consolidation** - Merge multiple KB sources into unified structure
- **Background Processing** - Long-running analysis with screen-based monitoring
- **Enterprise Integration** - Corporate network and security compliance

### 📊 **Research & Analysis**
- **Multi-Repository Analysis** - Cross-project knowledge correlation
- **Domain-Specific Insights** - Specialized analysis by expert agents
- **Pattern Recognition** - Automatic identification of solution patterns
- **Continuous Learning** - System improves through usage and feedback

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

**Built with ❤️ by the NPS Southwest Network Collaboration - CHDN**
