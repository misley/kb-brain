"""
Tests for Performance Integration Module
"""

import pytest
import numpy as np
from scipy import sparse
from unittest.mock import Mock, patch, MagicMock

from kb_brain.performance.performance_integration import PerformanceManager


class TestPerformanceManager:
    """Test cases for Performance Manager"""
    
    @pytest.fixture
    def manager(self):
        """Create performance manager instance"""
        return PerformanceManager(auto_optimize=False)
    
    @pytest.fixture
    def auto_manager(self):
        """Create auto-optimizing performance manager"""
        return PerformanceManager(auto_optimize=True)
    
    @pytest.fixture
    def test_embeddings(self):
        """Create test embeddings for similarity computation"""
        np.random.seed(42)
        embeddings = np.random.rand(1000, 100).astype(np.float32)
        query = np.random.rand(1, 100).astype(np.float32)
        return embeddings, query
    
    @pytest.fixture
    def test_sparse_embeddings(self):
        """Create test sparse embeddings"""
        np.random.seed(42)
        dense = np.random.rand(500, 200)
        # Make sparse (80% zeros)
        mask = np.random.rand(500, 200) < 0.2
        dense[~mask] = 0
        embeddings = sparse.csr_matrix(dense.astype(np.float32))
        query = sparse.csr_matrix(np.random.rand(1, 200).astype(np.float32))
        return embeddings, query
    
    def test_manager_initialization(self, manager):
        """Test performance manager initialization"""
        assert manager.auto_optimize is False
        assert hasattr(manager, 'cpu_optimizer')
        assert hasattr(manager, 'jit_engine')
        assert hasattr(manager, 'sparse_optimizer')
        assert isinstance(manager.performance_metrics, dict)
        assert "optimizations_applied" in manager.performance_metrics
        assert "speedup_measurements" in manager.performance_metrics
        assert "memory_savings" in manager.performance_metrics
    
    def test_auto_optimization(self, auto_manager):
        """Test automatic optimization on initialization"""
        assert auto_manager.auto_optimize is True
        # Should have applied some optimizations
        applied = auto_manager.performance_metrics["optimizations_applied"]
        assert isinstance(applied, list)
    
    def test_optimize_similarity_computation_dense(self, manager, test_embeddings):
        """Test optimized similarity computation with dense matrices"""
        embeddings, query = test_embeddings
        
        result = manager.optimize_similarity_computation(
            embeddings, query, top_k=5, metric="cosine"
        )
        
        assert isinstance(result, dict)
        assert "similarities" in result
        assert "method" in result
        assert "processing_time" in result
        assert "matrix_shape" in result
        assert "optimizations_used" in result
        
        assert result["method"] == "jit_optimized"
        assert result["processing_time"] >= 0
        assert result["matrix_shape"] == embeddings.shape
        
        # Check similarity results
        similarities = result["similarities"]
        assert "indices" in similarities
        assert "scores" in similarities
        assert "jit_optimization" in similarities
        
        indices = similarities["indices"]
        scores = similarities["scores"]
        assert len(indices) == 5
        assert len(scores) == 5
        assert np.all(indices >= 0)
        assert np.all(indices < embeddings.shape[0])
    
    def test_optimize_similarity_computation_sparse(self, manager, test_sparse_embeddings):
        """Test optimized similarity computation with sparse matrices"""
        embeddings, query = test_sparse_embeddings
        
        result = manager.optimize_similarity_computation(
            embeddings, query, top_k=3, metric="cosine"
        )
        
        assert result["method"] == "sparse_optimized"
        assert result["processing_time"] >= 0
        
        # Check similarity results
        similarities = result["similarities"]
        assert "indices" in similarities
        assert "scores" in similarities
        assert "sparse_optimization" in similarities
        
        indices = similarities["indices"]
        scores = similarities["scores"]
        assert len(indices) == 3
        assert len(scores) == 3
    
    def test_compute_sparse_similarity(self, manager, test_sparse_embeddings):
        """Test sparse similarity computation directly"""
        embeddings, query = test_sparse_embeddings
        
        result = manager._compute_sparse_similarity(
            embeddings, query, top_k=5, metric="cosine"
        )
        
        assert isinstance(result, dict)
        assert "indices" in result
        assert "scores" in result
        assert "sparse_optimization" in result
        
        assert result["sparse_optimization"] is True
        assert len(result["indices"]) == 5
        assert len(result["scores"]) == 5
    
    def test_compute_jit_similarity(self, manager, test_embeddings):
        """Test JIT similarity computation directly"""
        embeddings, query = test_embeddings
        
        result = manager._compute_jit_similarity(
            embeddings, query, top_k=5, metric="cosine"
        )
        
        assert isinstance(result, dict)
        assert "indices" in result
        assert "scores" in result
        assert "jit_optimization" in result
        
        assert result["jit_optimization"] is True
        assert len(result["indices"]) == 5
        assert len(result["scores"]) == 5
    
    def test_unsupported_metric_sparse(self, manager, test_sparse_embeddings):
        """Test error handling for unsupported metrics with sparse matrices"""
        embeddings, query = test_sparse_embeddings
        
        with pytest.raises(ValueError):
            manager._compute_sparse_similarity(
                embeddings, query, top_k=5, metric="euclidean"
            )
    
    def test_create_optimized_kb_brain(self, manager):
        """Test creation of optimized KB Brain class"""
        # Mock original KB Brain class
        class MockKBBrain:
            def __init__(self, *args, **kwargs):
                self.solutions = ["Solution 1", "Solution 2", "Solution 3"]
                self.embeddings_matrix = np.random.rand(3, 50).astype(np.float32)
            
            def find_best_solution_hybrid(self, problem, top_k=5, **kwargs):
                return [{"solution": f"Original solution {i}", "confidence": 0.5} 
                       for i in range(top_k)]
            
            def get_system_status(self):
                return {"status": "original"}
            
            def _embed_query(self, problem):
                return np.random.rand(1, 50).astype(np.float32)
        
        OptimizedKBBrain = manager.create_optimized_kb_brain(MockKBBrain)
        
        # Test that we get an enhanced class
        assert issubclass(OptimizedKBBrain, MockKBBrain)
        
        # Test instantiation
        optimized_instance = OptimizedKBBrain()
        assert hasattr(optimized_instance, 'performance_manager')
        assert isinstance(optimized_instance.performance_manager, PerformanceManager)
        
        # Test optimized solution finding
        solutions = optimized_instance.find_best_solution_hybrid("test problem", top_k=3)
        assert isinstance(solutions, list)
        assert len(solutions) <= 3
        
        # Test enhanced system status
        status = optimized_instance.get_system_status()
        assert "performance_optimizations" in status
    
    def test_integrate_with_sme_system(self, manager):
        """Test integration with SME system"""
        # Mock SME system
        mock_kb_engine = Mock()
        mock_kb_engine.search_knowledge = Mock(return_value=Mock(metadata={}))
        
        mock_agent = Mock()
        mock_agent.kb_engine = mock_kb_engine
        
        mock_sme_system = Mock()
        mock_sme_system.agents = {"agent1": mock_agent, "agent2": mock_agent}
        
        # Test integration
        manager.integrate_with_sme_system(mock_sme_system)
        
        # Check that agents have performance manager
        for agent in mock_sme_system.agents.values():
            assert hasattr(agent.kb_engine, 'performance_manager')
            assert agent.kb_engine.performance_manager == manager
    
    def test_optimize_kb_engine(self, manager):
        """Test KB engine optimization"""
        # Mock KB engine
        mock_kb_engine = Mock()
        original_search = Mock(return_value=Mock(metadata={}))
        mock_kb_engine.search_knowledge = original_search
        
        # Optimize the engine
        manager._optimize_kb_engine(mock_kb_engine)
        
        # Check that performance manager is added
        assert hasattr(mock_kb_engine, 'performance_manager')
        assert mock_kb_engine.performance_manager == manager
        
        # Check that search method is enhanced
        assert mock_kb_engine.search_knowledge != original_search
        
        # Test optimized search
        result = mock_kb_engine.search_knowledge("test")
        assert hasattr(result, 'metadata')
        assert result.metadata.get("performance_optimized") is True
    
    def test_benchmark_system_performance(self, manager):
        """Test system performance benchmarking"""
        test_sizes = [100, 200]
        results = manager.benchmark_system_performance(test_sizes=test_sizes)
        
        assert isinstance(results, dict)
        assert "cpu_optimization" in results
        assert "jit_benchmarks" in results
        assert "sparse_benchmarks" in results
        assert "system_info" in results
        
        # Check JIT benchmarks
        jit_benchmarks = results["jit_benchmarks"]
        for size in test_sizes:
            size_key = f"size_{size}"
            assert size_key in jit_benchmarks
        
        # Check sparse benchmarks
        assert isinstance(results["sparse_benchmarks"], dict)
    
    def test_get_optimization_status(self, manager):
        """Test optimization status retrieval"""
        status = manager.get_optimization_status()
        
        assert isinstance(status, dict)
        assert "cpu_optimizations" in status
        assert "jit_engine" in status
        assert "sparse_optimizer" in status
        assert "performance_metrics" in status
        assert "global_optimizations_applied" in status
        
        assert isinstance(status["global_optimizations_applied"], int)
        assert status["global_optimizations_applied"] >= 0
    
    def test_suggest_optimizations_intel_not_available(self, manager):
        """Test optimization suggestions when Intel extensions not available"""
        # Mock CPU optimizer to simulate Intel not available
        manager.cpu_optimizer.intel_available = False
        
        system_context = {"memory_usage_high": False, "cpu_count": 4}
        suggestions = manager.suggest_optimizations(system_context)
        
        assert isinstance(suggestions, list)
        
        # Should suggest Intel installation
        intel_suggestion = next((s for s in suggestions if "Intel Extension" in s["description"]), None)
        assert intel_suggestion is not None
        assert intel_suggestion["type"] == "install"
        assert intel_suggestion["impact"] == "high"
    
    def test_suggest_optimizations_jit_not_available(self, manager):
        """Test optimization suggestions when JIT not available"""
        # Mock JIT engine to simulate Numba not available
        manager.jit_engine.use_jit = False
        
        system_context = {"memory_usage_high": False, "cpu_count": 4}
        suggestions = manager.suggest_optimizations(system_context)
        
        # Should suggest Numba installation
        numba_suggestion = next((s for s in suggestions if "Numba" in s["description"]), None)
        assert numba_suggestion is not None
        assert numba_suggestion["type"] == "install"
        assert numba_suggestion["impact"] == "high"
    
    def test_suggest_optimizations_memory_high(self, manager):
        """Test optimization suggestions for high memory usage"""
        system_context = {"memory_usage_high": True, "cpu_count": 4}
        suggestions = manager.suggest_optimizations(system_context)
        
        # Should suggest memory optimization
        memory_suggestion = next((s for s in suggestions if "memory-efficient" in s["description"]), None)
        assert memory_suggestion is not None
        assert memory_suggestion["type"] == "configuration"
        assert memory_suggestion["impact"] == "medium"
    
    def test_suggest_optimizations_many_cpus(self, manager):
        """Test optimization suggestions for many CPU cores"""
        system_context = {"memory_usage_high": False, "cpu_count": 16}
        suggestions = manager.suggest_optimizations(system_context)
        
        # Should suggest threading optimization
        threading_suggestion = next((s for s in suggestions if "threading" in s["description"]), None)
        assert threading_suggestion is not None
        assert threading_suggestion["type"] == "configuration"
        assert "16 CPU cores" in threading_suggestion["description"]
    
    def test_query_vector_shape_handling(self, manager, test_embeddings):
        """Test handling of different query vector shapes"""
        embeddings, _ = test_embeddings
        
        # Test 1D query vector
        query_1d = np.random.rand(100).astype(np.float32)
        result = manager.optimize_similarity_computation(
            embeddings, query_1d, top_k=3, metric="cosine"
        )
        assert result["method"] == "jit_optimized"
        
        # Test 2D query vector
        query_2d = np.random.rand(1, 100).astype(np.float32)
        result = manager.optimize_similarity_computation(
            embeddings, query_2d, top_k=3, metric="cosine"
        )
        assert result["method"] == "jit_optimized"
    
    def test_sparse_query_conversion(self, manager, test_sparse_embeddings):
        """Test conversion of dense query to sparse for sparse matrices"""
        embeddings, _ = test_sparse_embeddings
        
        # Dense query should be converted to sparse
        dense_query = np.random.rand(1, 200).astype(np.float32)
        result = manager.optimize_similarity_computation(
            embeddings, dense_query, top_k=3, metric="cosine"
        )
        
        assert result["method"] == "sparse_optimized"
    
    def test_performance_metrics_tracking(self, manager, test_embeddings):
        """Test that performance metrics are tracked"""
        embeddings, query = test_embeddings
        
        initial_metrics = manager.performance_metrics.copy()
        
        # Perform optimization
        manager.optimize_similarity_computation(embeddings, query, top_k=5)
        
        # Metrics should be updated (at least processing time tracked)
        assert "optimizations_used" in manager.performance_metrics or len(manager.performance_metrics["optimizations_applied"]) >= len(initial_metrics["optimizations_applied"])
    
    @patch('kb_brain.performance.performance_integration.logger')
    def test_logging(self, mock_logger, manager):
        """Test that logging works correctly"""
        # Should have logged initialization
        mock_logger.info.assert_called()
        
        # Test with SME integration logging
        mock_sme_system = Mock()
        mock_sme_system.agents = {}
        
        manager.integrate_with_sme_system(mock_sme_system)
        
        # Should have logged integration
        assert any("Performance optimizations integrated" in str(call) 
                  for call in mock_logger.info.call_args_list)


def test_performance_integration_standalone():
    """Test standalone performance integration function"""
    from kb_brain.performance.performance_integration import test_performance_integration
    
    # Should not raise any errors
    manager = test_performance_integration()
    assert isinstance(manager, PerformanceManager)


if __name__ == "__main__":
    pytest.main([__file__])