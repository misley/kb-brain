"""
KB Brain Performance Optimization Module
CPU-optimized processing with Intel extensions and JIT compilation
"""

from .cpu_optimizer import CPUOptimizer, enable_intel_optimizations
from .jit_similarity import JITSimilarityEngine
from .sparse_optimizer import SparseMatrixOptimizer

__all__ = [
    'CPUOptimizer',
    'enable_intel_optimizations', 
    'JITSimilarityEngine',
    'SparseMatrixOptimizer'
]

__version__ = "1.0.0"