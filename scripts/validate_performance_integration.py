#!/usr/bin/env python3
"""
Simple Performance Integration Validation Script
Quick verification that performance optimizations are properly integrated
"""

import sys
import time
from pathlib import Path

# Add kb_brain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

def test_basic_performance_integration():
    """Basic test of performance integration"""
    print("🧪 KB Brain Performance Integration Validation")
    print("=" * 50)
    
    try:
        # Test HybridGPUKBBrain with performance optimizations
        print("\n1. Testing HybridGPUKBBrain with performance optimizations...")
        from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
        
        brain = HybridGPUKBBrain(enable_performance_optimizations=True)
        status = brain.get_system_status()
        
        if brain.performance_manager:
            print("   ✅ Performance manager initialized")
            perf_status = brain.performance_manager.get_optimization_status()
            print(f"   Intel extensions: {'✅' if perf_status.get('intel_extensions_available') else '❌'}")
            print(f"   Numba JIT: {'✅' if perf_status.get('numba_available') else '❌'}")
        else:
            print("   ⚠️  Performance manager not available")
        
        print(f"   Knowledge embeddings: {status['hybrid_gpu_status']['knowledge_embeddings']}")
        print(f"   Brain type: {status['hybrid_gpu_status']['brain_type']}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        # Test KB Integration Engine
        print("\n2. Testing KB Integration Engine with performance optimizations...")
        from kb_brain.intelligence.kb_integration_engine import KBIntegrationEngine
        
        engine = KBIntegrationEngine(enable_performance_optimizations=True)
        
        if engine.performance_manager:
            print("   ✅ Performance manager initialized")
        else:
            print("   ⚠️  Performance manager not available")
        
        # Quick search test
        response = engine.search_knowledge("test query", max_results=3)
        print(f"   Search results: {len(response.results)}")
        print(f"   Processing time: {response.processing_time:.3f}s")
        print(f"   Confidence: {response.confidence:.2f}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        # Test SME Agent System
        print("\n3. Testing SME Agent System with performance optimizations...")
        from kb_brain.intelligence.sme_agent_system import SMEAgentSystem
        
        sme_system = SMEAgentSystem(enable_performance_optimizations=True)
        
        if sme_system.performance_manager:
            print("   ✅ Performance manager initialized")
        else:
            print("   ⚠️  Performance manager not available")
        
        system_status = sme_system.get_sme_system_status()
        print(f"   Total agents: {system_status['total_agents']}")
        print(f"   Active domains: {len(system_status['active_domains'])}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    try:
        # Test performance manager directly
        print("\n4. Testing PerformanceManager directly...")
        from kb_brain.performance.performance_integration import PerformanceManager
        import numpy as np
        
        perf_manager = PerformanceManager(auto_optimize=True)
        
        # Test matrix operations
        test_matrix = np.random.rand(50, 25)
        test_query = np.random.rand(1, 25)
        
        start_time = time.time()
        result = perf_manager.optimize_similarity_computation(
            test_matrix, test_query, top_k=5, metric="cosine"
        )
        computation_time = time.time() - start_time
        
        print(f"   ✅ Similarity computation successful")
        print(f"   Method used: {result['method']}")
        print(f"   Computation time: {computation_time:.3f}s")
        print(f"   Results found: {len(result['similarities']['scores'])}")
        
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n✅ Performance integration validation completed!")
    print("\n💡 To use performance optimizations in CLI:")
    print("   kb-brain status --optimize-performance")
    print("   kb-brain search 'your query' --optimize-performance")
    print("   kb-brain interactive --optimize-performance")

if __name__ == "__main__":
    test_basic_performance_integration()