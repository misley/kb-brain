#!/bin/bash

# KB Brain Installation Script
echo "🚀 Installing KB Brain..."

# Check Python version
python_version=$(python3 --version 2>&1 | cut -d' ' -f2)
echo "🐍 Python version: $python_version"

# Install base requirements
echo "📦 Installing base requirements..."
pip install -r requirements.txt

# Install KB Brain
echo "🧠 Installing KB Brain..."
pip install -e .

# Test installation
echo "🧪 Testing installation..."
python3 -c "from kb_brain.core import HybridGPUKBBrain; print('✅ KB Brain installed successfully')"

echo "🎉 KB Brain installation complete!"
echo "📋 Next steps:"
echo "  1. Run 'kb-brain --help' to see available commands"
echo "  2. Use 'kb-brain-mcp' to start MCP server"
echo "  3. Run './scripts/install_gpu.sh' for GPU acceleration"
