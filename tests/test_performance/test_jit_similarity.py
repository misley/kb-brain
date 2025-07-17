"""
Tests for JIT Similarity Engine
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch

from kb_brain.performance.jit_similarity import (
    JITSimilarityEngine, NUMBA_AVAILABLE
)


class TestJITSimilarityEngine:
    """Test cases for JIT Similarity Engine"""
    
    @pytest.fixture
    def engine(self):
        """Create JIT similarity engine instance"""
        return JITSimilarityEngine(use_jit=True, parallel=True)
    
    @pytest.fixture
    def engine_no_jit(self):
        """Create non-JIT similarity engine instance"""
        return JITSimilarityEngine(use_jit=False, parallel=False)
    
    @pytest.fixture
    def test_data(self):
        """Create test data for similarity calculations"""
        np.random.seed(42)
        A = np.random.rand(50, 100).astype(np.float32)
        B = np.random.rand(80, 100).astype(np.float32)
        return A, B
    
    def test_engine_initialization(self, engine):
        """Test engine initialization"""
        assert engine.use_jit == (NUMBA_AVAILABLE)
        assert engine.parallel == (NUMBA_AVAILABLE)
        assert isinstance(engine.compiled_functions, dict)
    
    def test_engine_initialization_no_jit(self, engine_no_jit):
        """Test engine initialization without JIT"""
        assert engine_no_jit.use_jit is False
        assert engine_no_jit.parallel is False
    
    def test_cosine_similarity_jit(self, engine, test_data):
        """Test JIT cosine similarity calculation"""
        A, B = test_data
        
        similarities = engine.cosine_similarity_jit(A, B)
        
        assert similarities.shape == (A.shape[0], B.shape[0])
        assert similarities.dtype == np.float32
        assert np.all(similarities >= -1.0)
        assert np.all(similarities <= 1.0)
    
    def test_cosine_similarity_same_matrix(self, engine, test_data):
        """Test cosine similarity with same matrix"""
        A, _ = test_data
        
        similarities = engine.cosine_similarity_jit(A)
        
        assert similarities.shape == (A.shape[0], A.shape[0])
        # Diagonal should be close to 1 (self-similarity)
        diagonal = np.diag(similarities)
        assert np.allclose(diagonal, 1.0, atol=1e-6)
    
    def test_euclidean_distance_jit(self, engine, test_data):
        """Test JIT Euclidean distance calculation"""
        A, B = test_data
        
        distances = engine.euclidean_distance_jit(A, B)
        
        assert distances.shape == (A.shape[0], B.shape[0])
        assert distances.dtype == np.float32
        assert np.all(distances >= 0.0)
    
    def test_manhattan_distance_jit(self, engine, test_data):
        """Test JIT Manhattan distance calculation"""
        A, B = test_data
        
        distances = engine.manhattan_distance_jit(A, B)
        
        assert distances.shape == (A.shape[0], B.shape[0])
        assert distances.dtype == np.float32
        assert np.all(distances >= 0.0)
    
    def test_find_top_k_similar_cosine(self, engine, test_data):
        """Test top-k similar search with cosine similarity"""
        A, B = test_data
        k = 5
        
        indices, scores = engine.find_top_k_similar(A[:10], B, k=k, metric="cosine")
        
        assert indices.shape == (10, k)
        assert scores.shape == (10, k)
        assert indices.dtype == np.int32
        assert scores.dtype == np.float32
        
        # Check that indices are valid
        assert np.all(indices >= 0)
        assert np.all(indices < B.shape[0])
        
        # Check that scores are sorted in descending order
        for i in range(10):
            assert np.all(scores[i][:-1] >= scores[i][1:])
    
    def test_find_top_k_similar_euclidean(self, engine, test_data):
        """Test top-k similar search with Euclidean distance"""
        A, B = test_data
        k = 3
        
        indices, scores = engine.find_top_k_similar(A[:5], B, k=k, metric="euclidean")
        
        assert indices.shape == (5, k)
        assert scores.shape == (5, k)
        
        # Check that scores are sorted in ascending order (smaller distance is better)
        for i in range(5):
            assert np.all(scores[i][:-1] <= scores[i][1:])
    
    def test_find_top_k_similar_manhattan(self, engine, test_data):
        """Test top-k similar search with Manhattan distance"""
        A, B = test_data
        k = 3
        
        indices, scores = engine.find_top_k_similar(A[:5], B, k=k, metric="manhattan")
        
        assert indices.shape == (5, k)
        assert scores.shape == (5, k)
        
        # Check that scores are sorted in ascending order
        for i in range(5):
            assert np.all(scores[i][:-1] <= scores[i][1:])
    
    def test_invalid_metric(self, engine, test_data):
        """Test error handling for invalid metric"""
        A, B = test_data
        
        with pytest.raises(ValueError):
            engine.find_top_k_similar(A[:5], B, k=3, metric="invalid_metric")
    
    def test_benchmark_performance(self, engine):
        """Test performance benchmarking"""
        benchmark = engine.benchmark_performance(n_queries=100, n_candidates=500, n_features=50)
        
        assert isinstance(benchmark, dict)
        assert "n_queries" in benchmark
        assert "n_candidates" in benchmark
        assert "n_features" in benchmark
        assert "total_operations" in benchmark
        
        assert benchmark["n_queries"] == 100
        assert benchmark["n_candidates"] == 500
        assert benchmark["n_features"] == 50
        assert benchmark["total_operations"] == 100 * 500
        
        if engine.use_jit:
            assert "jit_time" in benchmark
            assert "jit_available" in benchmark
            assert benchmark["jit_available"] is True
    
    def test_engine_info(self, engine):
        """Test engine information retrieval"""
        info = engine.get_engine_info()
        
        assert isinstance(info, dict)
        assert "numba_available" in info
        assert "jit_enabled" in info
        assert "parallel_enabled" in info
        assert "compiled_functions" in info
        assert "engine_type" in info
        
        assert info["numba_available"] == NUMBA_AVAILABLE
        assert info["jit_enabled"] == engine.use_jit
        assert info["parallel_enabled"] == engine.parallel
        assert info["engine_type"] in ["JIT", "NumPy"]
    
    def test_fallback_to_sklearn(self, test_data):
        """Test fallback to sklearn when JIT is disabled"""
        engine_no_jit = JITSimilarityEngine(use_jit=False)
        A, B = test_data
        
        # This should work even without JIT
        similarities = engine_no_jit.cosine_similarity_jit(A[:10], B[:10])
        
        assert similarities.shape == (10, 10)
        assert np.all(similarities >= -1.0)
        assert np.all(similarities <= 1.0)
    
    def test_data_type_conversion(self, engine):
        """Test automatic data type conversion"""
        # Test with different input types
        A_int = np.random.randint(0, 10, size=(10, 20))
        B_float64 = np.random.rand(15, 20).astype(np.float64)
        
        similarities = engine.cosine_similarity_jit(A_int, B_float64)
        
        assert similarities.dtype == np.float32
        assert similarities.shape == (10, 15)
    
    def test_edge_cases(self, engine):
        """Test edge cases"""
        # Test with zero vectors
        A = np.zeros((5, 10), dtype=np.float32)
        B = np.random.rand(8, 10).astype(np.float32)
        
        similarities = engine.cosine_similarity_jit(A, B)
        
        assert similarities.shape == (5, 8)
        # Similarity with zero vector should be 0
        assert np.allclose(similarities, 0.0)
        
        # Test with single vector
        single_vector = np.random.rand(1, 10).astype(np.float32)
        similarities_single = engine.cosine_similarity_jit(single_vector, B)
        
        assert similarities_single.shape == (1, 8)
    
    def test_top_k_larger_than_candidates(self, engine, test_data):
        """Test top-k when k is larger than number of candidates"""
        A, B = test_data
        k = B.shape[0] + 10  # Larger than candidates
        
        indices, scores = engine.find_top_k_similar(A[:5], B, k=k, metric="cosine")
        
        # Should return at most B.shape[0] results
        assert indices.shape[1] <= B.shape[0]
        assert scores.shape[1] <= B.shape[0]
    
    @patch('kb_brain.performance.jit_similarity.NUMBA_AVAILABLE', False)
    def test_numba_not_available(self):
        """Test behavior when Numba is not available"""
        engine = JITSimilarityEngine(use_jit=True)
        
        assert engine.use_jit is False
        assert engine.parallel is False
        
        info = engine.get_engine_info()
        assert info["engine_type"] == "NumPy"
    
    def test_parallel_processing(self, engine):
        """Test that parallel processing can be enabled/disabled"""
        if NUMBA_AVAILABLE:
            engine_parallel = JITSimilarityEngine(use_jit=True, parallel=True)
            engine_sequential = JITSimilarityEngine(use_jit=True, parallel=False)
            
            assert engine_parallel.parallel is True
            assert engine_sequential.parallel is False
    
    def test_compilation_caching(self, engine):
        """Test that JIT functions are cached"""
        if engine.use_jit:
            A = np.random.rand(10, 5).astype(np.float32)
            B = np.random.rand(15, 5).astype(np.float32)
            
            # First call (compilation)
            similarities1 = engine.cosine_similarity_jit(A, B)
            
            # Second call (should use cached version)
            similarities2 = engine.cosine_similarity_jit(A, B)
            
            # Results should be identical
            assert np.allclose(similarities1, similarities2)


def test_module_imports():
    """Test that all required imports work"""
    from kb_brain.performance.jit_similarity import JITSimilarityEngine
    
    # Should not raise ImportError
    assert JITSimilarityEngine is not None


def test_numba_availability():
    """Test Numba availability detection"""
    from kb_brain.performance.jit_similarity import NUMBA_AVAILABLE
    
    assert isinstance(NUMBA_AVAILABLE, bool)


if __name__ == "__main__":
    pytest.main([__file__])