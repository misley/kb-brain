#!/bin/bash

# KB Brain GPU Installation Script
echo "🎮 Installing KB Brain GPU acceleration..."

# Check for CUDA
if command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA GPU detected"
    nvidia-smi
else
    echo "❌ No NVIDIA GPU detected"
    echo "GPU acceleration requires NVIDIA GPU with CUDA support"
    exit 1
fi

# Install GPU requirements
echo "📦 Installing GPU requirements..."
pip install -r requirements-gpu.txt

# Install KB Brain with GPU support
echo "🧠 Installing KB Brain with GPU support..."
pip install -e .[gpu]

# Test GPU functionality
echo "🧪 Testing GPU functionality..."
python3 -c "
import cupy as cp
from kb_brain.core import HybridGPUKBBrain
brain = HybridGPUKBBrain()
status = brain.get_hybrid_status()
print(f'✅ GPU Available: {status[\"gpu_available\"]}')
print(f'✅ GPU Enabled: {status[\"gpu_enabled\"]}')
"

echo "🎉 KB Brain GPU installation complete!"
