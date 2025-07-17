"""
Tests for Sparse Matrix Optimizer
"""

import pytest
import numpy as np
from scipy import sparse
from unittest.mock import Mock, patch
import warnings

from kb_brain.performance.sparse_optimizer import SparseMatrixOptimizer


class TestSparseMatrixOptimizer:
    """Test cases for Sparse Matrix Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create sparse matrix optimizer instance"""
        return SparseMatrixOptimizer(
            matrix_format="csr",
            dtype="float32",
            memory_efficient=True
        )
    
    @pytest.fixture
    def test_sparse_matrix(self):
        """Create test sparse matrix"""
        np.random.seed(42)
        # Create sparse matrix (TF-IDF-like)
        dense = np.random.rand(100, 500)
        # Make it sparse (90% zeros)
        mask = np.random.rand(100, 500) < 0.1
        dense[~mask] = 0
        return sparse.csr_matrix(dense.astype(np.float32))
    
    @pytest.fixture
    def test_tfidf_data(self):
        """Create test TF-IDF data"""
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            
            documents = [
                "This is a sample document about machine learning",
                "Another document discussing natural language processing", 
                "Text similarity and information retrieval methods",
                "Deep learning and neural networks for NLP",
                "Document clustering and classification techniques",
                "Information systems and data management",
                "Statistical analysis and pattern recognition",
                "Computer science and artificial intelligence"
            ] * 25  # Repeat for larger dataset
            
            vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            tfidf_matrix = vectorizer.fit_transform(documents)
            
            return tfidf_matrix, vectorizer, documents
            
        except ImportError:
            pytest.skip("sklearn not available")
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.matrix_format == "csr"
        assert optimizer.dtype == np.float32
        assert optimizer.memory_efficient is True
        assert isinstance(optimizer.compression_ratios, list)
        assert isinstance(optimizer.operation_times, list)
    
    def test_optimizer_initialization_different_formats(self):
        """Test optimizer with different matrix formats"""
        # Test CSC format
        csc_optimizer = SparseMatrixOptimizer(matrix_format="csc", dtype="float64")
        assert csc_optimizer.matrix_format == "csc"
        assert csc_optimizer.dtype == np.float64
        
        # Test COO format
        coo_optimizer = SparseMatrixOptimizer(matrix_format="coo")
        assert coo_optimizer.matrix_format == "coo"
    
    def test_optimize_tfidf_matrix(self, optimizer, test_sparse_matrix):
        """Test TF-IDF matrix optimization"""
        original_nnz = test_sparse_matrix.nnz
        
        optimized = optimizer.optimize_tfidf_matrix(test_sparse_matrix)
        
        assert sparse.isspmatrix_csr(optimized)
        assert optimized.dtype == np.float32
        assert optimized.nnz <= original_nnz  # Should not increase non-zeros
        assert optimized.shape == test_sparse_matrix.shape
        assert len(optimizer.compression_ratios) > 0
    
    def test_matrix_format_conversion(self, optimizer):
        """Test matrix format conversion"""
        # Start with COO matrix
        np.random.seed(42)
        dense = np.random.rand(50, 100)
        dense[dense < 0.9] = 0  # Make sparse
        coo_matrix = sparse.coo_matrix(dense)
        
        optimized = optimizer.optimize_tfidf_matrix(coo_matrix)
        
        assert sparse.isspmatrix_csr(optimized)
        assert optimized.dtype == np.float32
    
    def test_dtype_conversion(self, optimizer):
        """Test data type conversion"""
        # Create matrix with different dtype
        np.random.seed(42)
        dense = np.random.rand(30, 60).astype(np.float64)
        dense[dense < 0.8] = 0
        matrix = sparse.csr_matrix(dense)
        
        assert matrix.dtype == np.float64
        
        optimized = optimizer.optimize_tfidf_matrix(matrix)
        
        assert optimized.dtype == np.float32
    
    def test_fast_cosine_similarity(self, optimizer, test_sparse_matrix):
        """Test fast cosine similarity calculation"""
        # Test with same matrix
        similarities = optimizer.fast_cosine_similarity(test_sparse_matrix[:20])
        
        assert similarities.shape == (20, 20)
        
        # Diagonal should be close to 1 (self-similarity)
        if sparse.issparse(similarities):
            diagonal = similarities.diagonal()
        else:
            diagonal = np.diag(similarities)
        
        assert np.allclose(diagonal, 1.0, atol=1e-5)
        assert len(optimizer.operation_times) > 0
    
    def test_fast_cosine_similarity_different_matrices(self, optimizer, test_sparse_matrix):
        """Test cosine similarity with different matrices"""
        X = test_sparse_matrix[:30]
        Y = test_sparse_matrix[30:60]
        
        similarities = optimizer.fast_cosine_similarity(X, Y)
        
        assert similarities.shape == (30, 30)
        
        # Values should be between -1 and 1
        if sparse.issparse(similarities):
            similarities_dense = similarities.toarray()
        else:
            similarities_dense = similarities
        
        assert np.all(similarities_dense >= -1.0)
        assert np.all(similarities_dense <= 1.0)
    
    def test_fast_cosine_similarity_top_k(self, optimizer, test_sparse_matrix):
        """Test cosine similarity with top-k filtering"""
        k = 5
        similarities = optimizer.fast_cosine_similarity(
            test_sparse_matrix[:10], 
            test_sparse_matrix, 
            top_k=k
        )
        
        assert similarities.shape == (10, k)
        
        # Check that results are sorted in descending order
        for i in range(10):
            row = similarities[i]
            assert np.all(row[:-1] >= row[1:])
    
    def test_normalize_sparse_matrix(self, optimizer, test_sparse_matrix):
        """Test sparse matrix normalization"""
        normalized = optimizer._normalize_sparse_matrix(test_sparse_matrix)
        
        assert normalized.shape == test_sparse_matrix.shape
        assert sparse.issparse(normalized)
        
        # Check that rows are approximately unit length
        row_norms = np.array(normalized.multiply(normalized).sum(axis=1)).flatten()
        row_norms = np.sqrt(row_norms)
        
        # Most rows should have norm close to 1 (some might be zero)
        non_zero_norms = row_norms[row_norms > 0]
        if len(non_zero_norms) > 0:
            assert np.allclose(non_zero_norms, 1.0, atol=1e-5)
    
    def test_extract_top_k_sparse(self, optimizer):
        """Test top-k extraction from sparse similarities"""
        np.random.seed(42)
        similarities = np.random.rand(10, 20).astype(np.float32)
        k = 5
        
        top_k = optimizer._extract_top_k_sparse(similarities, k)
        
        assert top_k.shape == (10, k)
        
        # Check sorting
        for i in range(10):
            row = top_k[i]
            assert np.all(row[:-1] >= row[1:])
    
    def test_optimize_vectorizer_output(self, optimizer, test_tfidf_data):
        """Test vectorizer output optimization"""
        tfidf_matrix, vectorizer, documents = test_tfidf_data
        
        optimized_matrix, info = optimizer.optimize_vectorizer_output(
            vectorizer, documents[:50]
        )
        
        assert sparse.issparse(optimized_matrix)
        assert isinstance(info, dict)
        assert "original_shape" in info
        assert "optimized_shape" in info
        assert "sparsity" in info
        assert "format" in info
        assert "dtype" in info
        
        assert info["format"] == "csr"
        assert info["dtype"] == "float32"
        assert 0 <= info["sparsity"] <= 1
    
    def test_optimize_vectorizer_output_with_feature_selection(self, optimizer, test_tfidf_data):
        """Test vectorizer optimization with feature selection"""
        tfidf_matrix, vectorizer, documents = test_tfidf_data
        
        max_features = 100
        optimized_matrix, info = optimizer.optimize_vectorizer_output(
            vectorizer, documents[:50], max_features=max_features
        )
        
        assert optimized_matrix.shape[1] <= max_features
        assert info["optimized_shape"][1] <= max_features
    
    def test_select_top_features(self, optimizer, test_sparse_matrix):
        """Test feature selection"""
        max_features = 100
        selected = optimizer._select_top_features(test_sparse_matrix, max_features)
        
        assert selected.shape[0] == test_sparse_matrix.shape[0]
        assert selected.shape[1] == max_features
        assert sparse.issparse(selected)
    
    def test_create_similarity_index(self, optimizer, test_sparse_matrix):
        """Test similarity index creation"""
        index = optimizer.create_similarity_index(test_sparse_matrix, chunk_size=500)
        
        assert isinstance(index, dict)
        assert "matrix" in index
        assert "norms" in index
        assert "chunk_size" in index
        assert "n_documents" in index
        assert "n_features" in index
        assert "created_at" in index
        assert "format" in index
        
        assert index["n_documents"] == test_sparse_matrix.shape[0]
        assert index["n_features"] == test_sparse_matrix.shape[1]
        assert index["chunk_size"] == 500
        assert sparse.issparse(index["matrix"])
        assert isinstance(index["norms"], np.ndarray)
    
    def test_query_similarity_index(self, optimizer, test_sparse_matrix):
        """Test similarity index querying"""
        index = optimizer.create_similarity_index(test_sparse_matrix)
        query_vector = test_sparse_matrix[0]
        
        top_indices, top_scores = optimizer.query_similarity_index(
            index, query_vector, top_k=5
        )
        
        assert len(top_indices) == 5
        assert len(top_scores) == 5
        assert isinstance(top_indices, np.ndarray)
        assert isinstance(top_scores, np.ndarray)
        
        # Check that indices are valid
        assert np.all(top_indices >= 0)
        assert np.all(top_indices < test_sparse_matrix.shape[0])
        
        # Check that scores are sorted in descending order
        assert np.all(top_scores[:-1] >= top_scores[1:])
        
        # First result should be the query itself (highest similarity)
        assert top_indices[0] == 0
        assert np.isclose(top_scores[0], 1.0, atol=1e-5)
    
    def test_query_similarity_index_zero_vector(self, optimizer, test_sparse_matrix):
        """Test querying with zero vector"""
        index = optimizer.create_similarity_index(test_sparse_matrix)
        
        # Create zero query vector
        zero_vector = sparse.csr_matrix((1, test_sparse_matrix.shape[1]))
        
        top_indices, top_scores = optimizer.query_similarity_index(
            index, zero_vector, top_k=3
        )
        
        assert len(top_indices) == 3
        assert len(top_scores) == 3
        # Scores should all be 0 for zero vector
        assert np.allclose(top_scores, 0.0)
    
    def test_get_memory_usage(self, optimizer, test_sparse_matrix):
        """Test memory usage calculation"""
        memory_info = optimizer.get_memory_usage(test_sparse_matrix)
        
        assert isinstance(memory_info, dict)
        assert "data_bytes" in memory_info
        assert "indices_bytes" in memory_info
        assert "indptr_bytes" in memory_info
        assert "total_bytes" in memory_info
        assert "total_mb" in memory_info
        assert "density" in memory_info
        assert "sparsity" in memory_info
        assert "dense_matrix_mb" in memory_info
        assert "compression_ratio" in memory_info
        
        assert memory_info["total_bytes"] > 0
        assert memory_info["total_mb"] > 0
        assert 0 <= memory_info["density"] <= 1
        assert 0 <= memory_info["sparsity"] <= 1
        assert memory_info["density"] + memory_info["sparsity"] == pytest.approx(1.0)
        assert memory_info["compression_ratio"] >= 1.0  # Should be compressed
    
    def test_benchmark_operations(self, optimizer):
        """Test operation benchmarking"""
        matrix_sizes = [(100, 200), (200, 300)]
        results = optimizer.benchmark_operations(matrix_sizes=matrix_sizes, sparsity=0.9)
        
        assert isinstance(results, dict)
        
        for size_key in ["100x200", "200x300"]:
            assert size_key in results
            result = results[size_key]
            
            assert "optimization_time" in result
            assert "similarity_time" in result
            assert "memory_mb" in result
            assert "compression_ratio" in result
            assert "sparsity" in result
            
            assert result["optimization_time"] >= 0
            assert result["similarity_time"] >= 0
            assert result["memory_mb"] > 0
            assert result["compression_ratio"] >= 1.0
            assert 0 <= result["sparsity"] <= 1
    
    def test_get_optimization_stats(self, optimizer, test_sparse_matrix):
        """Test optimization statistics"""
        # Perform some operations to generate stats
        optimizer.optimize_tfidf_matrix(test_sparse_matrix)
        optimizer.fast_cosine_similarity(test_sparse_matrix[:10])
        
        stats = optimizer.get_optimization_stats()
        
        assert isinstance(stats, dict)
        assert "matrix_format" in stats
        assert "dtype" in stats
        assert "memory_efficient" in stats
        assert "operations_performed" in stats
        
        assert stats["matrix_format"] == "csr"
        assert stats["dtype"] == "float32"
        assert stats["memory_efficient"] is True
        assert stats["operations_performed"] > 0
        
        # Should have compression and timing stats
        assert "avg_compression_ratio" in stats
        assert "avg_operation_time" in stats
    
    def test_different_sparse_formats(self, optimizer):
        """Test with different sparse matrix formats"""
        np.random.seed(42)
        dense = np.random.rand(50, 80)
        dense[dense < 0.8] = 0
        
        # Test with different input formats
        csr_matrix = sparse.csr_matrix(dense)
        csc_matrix = sparse.csc_matrix(dense)
        coo_matrix = sparse.coo_matrix(dense)
        
        for matrix in [csr_matrix, csc_matrix, coo_matrix]:
            optimized = optimizer.optimize_tfidf_matrix(matrix)
            assert sparse.isspmatrix_csr(optimized)
            assert optimized.shape == matrix.shape
    
    def test_edge_cases(self, optimizer):
        """Test edge cases"""
        # Empty matrix
        empty_matrix = sparse.csr_matrix((10, 20))
        optimized_empty = optimizer.optimize_tfidf_matrix(empty_matrix)
        assert optimized_empty.nnz == 0
        assert optimized_empty.shape == (10, 20)
        
        # Single row matrix
        single_row = sparse.csr_matrix(np.random.rand(1, 50))
        optimized_single = optimizer.optimize_tfidf_matrix(single_row)
        assert optimized_single.shape == (1, 50)
        
        # Single column matrix
        single_col = sparse.csr_matrix(np.random.rand(50, 1))
        optimized_col = optimizer.optimize_tfidf_matrix(single_col)
        assert optimized_col.shape == (50, 1)
    
    def test_large_k_value(self, optimizer, test_sparse_matrix):
        """Test with k larger than matrix dimensions"""
        large_k = test_sparse_matrix.shape[1] + 100
        
        similarities = optimizer.fast_cosine_similarity(
            test_sparse_matrix[:5], 
            test_sparse_matrix, 
            top_k=large_k
        )
        
        # Should return at most the number of candidates
        assert similarities.shape[1] <= test_sparse_matrix.shape[0]
    
    def test_memory_efficient_mode(self):
        """Test memory efficient mode"""
        mem_optimizer = SparseMatrixOptimizer(memory_efficient=True)
        assert mem_optimizer.memory_efficient is True
        
        non_mem_optimizer = SparseMatrixOptimizer(memory_efficient=False)
        assert non_mem_optimizer.memory_efficient is False


def test_module_imports():
    """Test that all required imports work"""
    from kb_brain.performance.sparse_optimizer import SparseMatrixOptimizer
    
    # Should not raise ImportError
    assert SparseMatrixOptimizer is not None


if __name__ == "__main__":
    pytest.main([__file__])