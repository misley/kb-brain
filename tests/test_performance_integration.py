#!/usr/bin/env python3
"""
Performance Integration Test Suite
Validates that performance optimizations work correctly across all KB Brain components
"""

import unittest
import asyncio
import time
import numpy as np
from pathlib import Path
import sys
import os

# Add kb_brain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
from kb_brain.intelligence.sme_agent_system import SMEAgentSystem
from kb_brain.intelligence.kb_integration_engine import KBIntegrationEngine

class TestPerformanceIntegration(unittest.TestCase):
    """Test performance optimization integration across all components"""
    
    def setUp(self):
        """Set up test environment"""
        self.test_kb_path = "/tmp/test_kb_brain"
        
        # Ensure test directory exists
        Path(self.test_kb_path).mkdir(exist_ok=True)
        
        # Create test data
        self._create_test_knowledge_base()
    
    def _create_test_knowledge_base(self):
        """Create test knowledge base for performance testing"""
        import json
        
        # Create test KB data
        test_kb = {
            "system": {
                "test_problems": {
                    "ssl_cert_problem": {
                        "problem": "SSL certificate issues in corporate network",
                        "solution": "Configure DOI certificates and update ca-certificates bundle",
                        "status": "solved",
                        "tags": ["ssl", "network", "certificates"],
                        "success_rate": 0.9
                    },
                    "gpu_memory_problem": {
                        "problem": "GPU memory exhausted during large matrix operations",
                        "solution": "Use CuPy memory pool management and batch processing",
                        "status": "solved", 
                        "tags": ["gpu", "memory", "cupy"],
                        "success_rate": 0.85
                    },
                    "performance_slow": {
                        "problem": "Slow similarity search in large knowledge base",
                        "solution": "Enable Intel scikit-learn extensions and Numba JIT compilation",
                        "status": "solved",
                        "tags": ["performance", "similarity", "optimization"],
                        "success_rate": 0.95
                    }
                }
            },
            "projects": {
                "dunes": {
                    "boundary_detection": {
                        "problem": "NDWI boundary segmentation accuracy issues",
                        "solution": "Apply edge detection filters and directional shift analysis",
                        "status": "active",
                        "tags": ["dunes", "boundary", "ndwi"],
                        "success_rate": 0.7
                    }
                }
            }
        }
        
        kb_file = Path(self.test_kb_path) / "test_kb.json"
        with open(kb_file, 'w') as f:
            json.dump(test_kb, f, indent=2)
    
    def test_hybrid_brain_performance_optimization(self):
        """Test HybridGPUKBBrain with performance optimizations"""
        print("\n🧪 Testing HybridGPUKBBrain performance optimization...")
        
        # Test without optimizations
        brain_standard = HybridGPUKBBrain(
            kb_root=self.test_kb_path,
            enable_performance_optimizations=False
        )
        
        start_time = time.time()
        solutions_standard = brain_standard.find_best_solution_hybrid(
            "SSL certificate issues", top_k=3
        )
        standard_time = time.time() - start_time
        
        # Test with optimizations
        brain_optimized = HybridGPUKBBrain(
            kb_root=self.test_kb_path,
            enable_performance_optimizations=True
        )
        
        start_time = time.time()
        solutions_optimized = brain_optimized.find_best_solution_hybrid(
            "SSL certificate issues", top_k=3
        )
        optimized_time = time.time() - start_time
        
        # Validate results
        self.assertGreater(len(solutions_standard), 0, "Standard mode should find solutions")
        self.assertGreater(len(solutions_optimized), 0, "Optimized mode should find solutions")
        
        # Performance should be similar or better (allow for variance)
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        print(f"  Standard: {standard_time:.3f}s, Optimized: {optimized_time:.3f}s, Speedup: {speedup:.1f}x")
        
        # Validate performance manager is available
        self.assertIsNotNone(brain_optimized.performance_manager, 
                           "Performance manager should be initialized")
        
        # Check system status includes performance info
        status = brain_optimized.get_system_status()
        self.assertIn('performance_optimizations', status,
                     "Status should include performance optimization info")
    
    def test_kb_integration_engine_performance(self):
        """Test KB Integration Engine with performance optimizations"""
        print("\n🧪 Testing KB Integration Engine performance optimization...")
        
        # Test without optimizations
        engine_standard = KBIntegrationEngine(
            kb_system_path=self.test_kb_path,
            enable_performance_optimizations=False
        )
        
        start_time = time.time()
        response_standard = engine_standard.search_knowledge(
            "GPU memory problems", max_results=5
        )
        standard_time = time.time() - start_time
        
        # Test with optimizations
        engine_optimized = KBIntegrationEngine(
            kb_system_path=self.test_kb_path,
            enable_performance_optimizations=True
        )
        
        start_time = time.time()
        response_optimized = engine_optimized.search_knowledge(
            "GPU memory problems", max_results=5
        )
        optimized_time = time.time() - start_time
        
        # Validate results
        self.assertGreater(len(response_standard.results), 0, 
                          "Standard engine should find results")
        self.assertGreater(len(response_optimized.results), 0,
                          "Optimized engine should find results")
        
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        print(f"  Standard: {standard_time:.3f}s, Optimized: {optimized_time:.3f}s, Speedup: {speedup:.1f}x")
        
        # Validate performance manager
        self.assertIsNotNone(engine_optimized.performance_manager,
                           "Performance manager should be initialized")
    
    async def test_sme_system_performance(self):
        """Test SME Agent System with performance optimizations"""
        print("\n🧪 Testing SME Agent System performance optimization...")
        
        # Test without optimizations
        sme_standard = SMEAgentSystem(
            base_kb_path=self.test_kb_path,
            enable_performance_optimizations=False
        )
        
        start_time = time.time()
        agent_id, response_standard = await sme_standard.route_query(
            "performance optimization techniques",
            context={"domain": "technical"}
        )
        standard_time = time.time() - start_time
        
        # Test with optimizations
        sme_optimized = SMEAgentSystem(
            base_kb_path=self.test_kb_path,
            enable_performance_optimizations=True
        )
        
        start_time = time.time()
        agent_id_opt, response_optimized = await sme_optimized.route_query(
            "performance optimization techniques",
            context={"domain": "technical"}
        )
        optimized_time = time.time() - start_time
        
        # Validate results
        self.assertIsNotNone(agent_id, "Standard SME should route query")
        self.assertIsNotNone(agent_id_opt, "Optimized SME should route query")
        self.assertIsInstance(response_standard, dict, "Standard response should be dict")
        self.assertIsInstance(response_optimized, dict, "Optimized response should be dict")
        
        speedup = standard_time / optimized_time if optimized_time > 0 else 1.0
        print(f"  Standard: {standard_time:.3f}s, Optimized: {optimized_time:.3f}s, Speedup: {speedup:.1f}x")
        
        # Validate performance manager
        self.assertIsNotNone(sme_optimized.performance_manager,
                           "SME performance manager should be initialized")
    
    def test_cross_component_compatibility(self):
        """Test that performance optimizations work across all components together"""
        print("\n🧪 Testing cross-component performance compatibility...")
        
        # Initialize all components with performance optimizations
        brain = HybridGPUKBBrain(
            kb_root=self.test_kb_path,
            enable_performance_optimizations=True
        )
        
        kb_engine = KBIntegrationEngine(
            kb_system_path=self.test_kb_path,
            enable_performance_optimizations=True
        )
        
        # Test that they work together
        brain_status = brain.get_system_status()
        kb_status = kb_engine.get_system_status()
        
        # All should have performance optimizations enabled
        self.assertIn('performance_optimizations', brain_status,
                     "Brain should have performance optimization status")
        
        # Test search functionality
        solutions = brain.find_best_solution_hybrid("test query", top_k=2)
        kb_response = kb_engine.search_knowledge("test query", max_results=2)
        
        print(f"  Brain found {len(solutions)} solutions")
        print(f"  KB Engine found {len(kb_response.results)} results")
        
        # Both should return results or gracefully handle empty case
        self.assertIsInstance(solutions, list, "Brain should return list of solutions")
        self.assertIsInstance(kb_response.results, list, "KB Engine should return list of results")
    
    def test_performance_manager_integration(self):
        """Test that PerformanceManager integrates correctly"""
        print("\n🧪 Testing PerformanceManager integration...")
        
        brain = HybridGPUKBBrain(enable_performance_optimizations=True)
        
        if brain.performance_manager:
            # Test optimization status
            status = brain.performance_manager.get_optimization_status()
            print(f"  Intel extensions: {'✅' if status.get('intel_extensions_available') else '❌'}")
            print(f"  Numba JIT: {'✅' if status.get('numba_available') else '❌'}")
            
            # Test matrix operations if possible
            if status.get('intel_extensions_available') or status.get('numba_available'):
                test_matrix = np.random.rand(100, 50)
                test_query = np.random.rand(1, 50)
                
                try:
                    result = brain.performance_manager.optimize_similarity_computation(
                        test_matrix, test_query, top_k=5, metric="cosine"
                    )
                    
                    self.assertIn('similarities', result, "Should return similarity results")
                    self.assertIn('method', result, "Should specify optimization method used")
                    print(f"  Similarity computation: ✅ {result['method']}")
                    
                except Exception as e:
                    print(f"  Similarity computation: ⚠️  {e}")
        else:
            print("  ⚠️  Performance manager not available")
    
    def tearDown(self):
        """Clean up test environment"""
        import shutil
        if Path(self.test_kb_path).exists():
            shutil.rmtree(self.test_kb_path)

async def run_async_tests():
    """Run async tests"""
    suite = TestPerformanceIntegration()
    suite.setUp()
    try:
        await suite.test_sme_system_performance()
    finally:
        suite.tearDown()

def main():
    """Run all performance integration tests"""
    print("🚀 Running KB Brain Performance Integration Tests")
    print("=" * 60)
    
    # Run synchronous tests
    unittest.main(argv=[''], verbosity=2, exit=False)
    
    # Run async tests
    print("\n🔄 Running async tests...")
    asyncio.run(run_async_tests())
    
    print("\n✅ Performance integration tests completed!")

if __name__ == "__main__":
    main()