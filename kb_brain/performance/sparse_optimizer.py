"""
Sparse Matrix Optimizer for Text Similarity
Optimized sparse matrix operations for TF-IDF and text processing
"""

import numpy as np
from scipy import sparse
import logging
from typing import Dict, Any, Tuple, Union, Optional
import warnings

logger = logging.getLogger(__name__)


class SparseMatrixOptimizer:
    """Optimized sparse matrix operations for text similarity"""
    
    def __init__(self, 
                 matrix_format: str = "csr",
                 dtype: str = "float32",
                 memory_efficient: bool = True):
        """
        Initialize sparse matrix optimizer
        
        Args:
            matrix_format: Sparse matrix format ("csr", "csc", "coo")
            dtype: Data type for matrices
            memory_efficient: Whether to prioritize memory efficiency
        """
        self.matrix_format = matrix_format
        self.dtype = np.dtype(dtype)
        self.memory_efficient = memory_efficient
        
        # Performance tracking
        self.compression_ratios = []
        self.operation_times = []
        
        logger.info(f"Sparse optimizer initialized: format={matrix_format}, dtype={dtype}")
    
    def optimize_tfidf_matrix(self, tfidf_matrix: sparse.spmatrix) -> sparse.spmatrix:
        """
        Optimize TF-IDF matrix for similarity calculations
        
        Args:
            tfidf_matrix: Input TF-IDF sparse matrix
        
        Returns:
            Optimized sparse matrix
        """
        
        # Convert to optimal format
        if not isinstance(tfidf_matrix, getattr(sparse, f"{self.matrix_format}_matrix")):
            logger.info(f"Converting matrix to {self.matrix_format} format")
            if self.matrix_format == "csr":
                optimized = tfidf_matrix.tocsr()
            elif self.matrix_format == "csc":
                optimized = tfidf_matrix.tocsc()
            elif self.matrix_format == "coo":
                optimized = tfidf_matrix.tocoo()
            else:
                optimized = tfidf_matrix
        else:
            optimized = tfidf_matrix.copy()
        
        # Convert data type
        if optimized.dtype != self.dtype:
            logger.info(f"Converting matrix dtype to {self.dtype}")
            optimized = optimized.astype(self.dtype)
        
        # Eliminate explicit zeros
        optimized.eliminate_zeros()
        
        # Sort indices for better cache performance
        if hasattr(optimized, 'sort_indices'):
            optimized.sort_indices()
        
        # Calculate compression ratio
        original_size = tfidf_matrix.shape[0] * tfidf_matrix.shape[1] * tfidf_matrix.dtype.itemsize
        sparse_size = optimized.data.nbytes + optimized.indices.nbytes + optimized.indptr.nbytes
        compression_ratio = original_size / sparse_size
        self.compression_ratios.append(compression_ratio)
        
        logger.info(f"Matrix optimized: {compression_ratio:.1f}x compression, {optimized.nnz} non-zeros")
        
        return optimized
    
    def fast_cosine_similarity(self, 
                             X: sparse.spmatrix, 
                             Y: Optional[sparse.spmatrix] = None,
                             top_k: Optional[int] = None) -> Union[sparse.spmatrix, np.ndarray]:
        """
        Fast cosine similarity for sparse matrices
        
        Args:
            X: First sparse matrix (n_samples_X, n_features)
            Y: Second sparse matrix (n_samples_Y, n_features), if None uses X
            top_k: If specified, return only top-k similarities per row
        
        Returns:
            Similarity matrix (sparse or dense based on sparsity)
        """
        
        import time
        start_time = time.time()
        
        if Y is None:
            Y = X
        
        # Ensure matrices are in CSR format for efficient row operations
        X_csr = X.tocsr() if not sparse.isspmatrix_csr(X) else X
        Y_csr = Y.tocsr() if not sparse.isspmatrix_csr(Y) else Y
        
        # Normalize matrices for cosine similarity
        X_normalized = self._normalize_sparse_matrix(X_csr)
        Y_normalized = self._normalize_sparse_matrix(Y_csr)
        
        # Compute similarity using matrix multiplication
        similarities = X_normalized @ Y_normalized.T
        
        # Convert to dense if result is not very sparse
        if similarities.nnz > (similarities.shape[0] * similarities.shape[1] * 0.1):
            similarities = similarities.toarray()
        
        # Extract top-k if requested
        if top_k is not None:
            similarities = self._extract_top_k_sparse(similarities, top_k)
        
        processing_time = time.time() - start_time
        self.operation_times.append(processing_time)
        
        logger.info(f"Cosine similarity computed in {processing_time:.3f}s")
        
        return similarities
    
    def _normalize_sparse_matrix(self, matrix: sparse.spmatrix) -> sparse.spmatrix:
        """Normalize sparse matrix rows to unit length"""
        
        # Calculate row norms
        squared_norms = np.array(matrix.multiply(matrix).sum(axis=1)).flatten()
        
        # Avoid division by zero
        norms = np.sqrt(squared_norms)
        norms[norms == 0] = 1.0
        
        # Create diagonal matrix for normalization
        norm_diag = sparse.diags(1.0 / norms, format='csr')
        
        # Normalize
        normalized = norm_diag @ matrix
        
        return normalized
    
    def _extract_top_k_sparse(self, 
                            similarities: Union[sparse.spmatrix, np.ndarray], 
                            k: int) -> np.ndarray:
        """Extract top-k similarities for each row"""
        
        if sparse.issparse(similarities):
            similarities_dense = similarities.toarray()
        else:
            similarities_dense = similarities
        
        # Get top-k indices and values
        top_k_indices = np.argsort(-similarities_dense, axis=1)[:, :k]
        
        n_rows = similarities_dense.shape[0]
        top_k_similarities = np.zeros((n_rows, k), dtype=self.dtype)
        
        for i in range(n_rows):
            for j in range(k):
                if j < similarities_dense.shape[1]:
                    top_k_similarities[i, j] = similarities_dense[i, top_k_indices[i, j]]
        
        return top_k_similarities
    
    def optimize_vectorizer_output(self, 
                                 vectorizer,
                                 texts: list,
                                 max_features: Optional[int] = None) -> Tuple[sparse.spmatrix, Dict[str, Any]]:
        """
        Optimize vectorizer output for performance
        
        Args:
            vectorizer: Fitted TF-IDF or Count vectorizer
            texts: List of text documents
            max_features: Maximum number of features to keep
        
        Returns:
            Tuple of (optimized_matrix, optimization_info)
        """
        
        # Transform texts
        matrix = vectorizer.transform(texts)
        
        # Feature selection if requested
        if max_features and matrix.shape[1] > max_features:
            matrix = self._select_top_features(matrix, max_features)
        
        # Optimize matrix
        optimized_matrix = self.optimize_tfidf_matrix(matrix)
        
        # Calculate optimization info
        info = {
            "original_shape": matrix.shape,
            "optimized_shape": optimized_matrix.shape,
            "original_nnz": matrix.nnz,
            "optimized_nnz": optimized_matrix.nnz,
            "sparsity": 1.0 - (optimized_matrix.nnz / (optimized_matrix.shape[0] * optimized_matrix.shape[1])),
            "format": optimized_matrix.format,
            "dtype": str(optimized_matrix.dtype)
        }
        
        return optimized_matrix, info
    
    def _select_top_features(self, matrix: sparse.spmatrix, max_features: int) -> sparse.spmatrix:
        """Select top features based on variance or frequency"""
        
        # Calculate feature importance (sum of values)
        feature_importance = np.array(matrix.sum(axis=0)).flatten()
        
        # Get top features
        top_feature_indices = np.argsort(-feature_importance)[:max_features]
        
        # Select columns
        selected_matrix = matrix[:, top_feature_indices]
        
        logger.info(f"Selected top {max_features} features from {matrix.shape[1]}")
        
        return selected_matrix
    
    def create_similarity_index(self, 
                              documents_matrix: sparse.spmatrix,
                              chunk_size: int = 1000) -> Dict[str, Any]:
        """
        Create an optimized similarity index for fast lookups
        
        Args:
            documents_matrix: Matrix of document vectors
            chunk_size: Size of chunks for processing
        
        Returns:
            Similarity index structure
        """
        
        logger.info(f"Creating similarity index for {documents_matrix.shape[0]} documents")
        
        # Optimize matrix
        optimized_matrix = self.optimize_tfidf_matrix(documents_matrix)
        
        # Pre-compute document norms for faster similarity
        doc_norms = np.array(optimized_matrix.multiply(optimized_matrix).sum(axis=1)).flatten()
        doc_norms = np.sqrt(doc_norms)
        
        # Create index structure
        index = {
            "matrix": optimized_matrix,
            "norms": doc_norms,
            "chunk_size": chunk_size,
            "n_documents": optimized_matrix.shape[0],
            "n_features": optimized_matrix.shape[1],
            "created_at": np.datetime64('now'),
            "format": optimized_matrix.format
        }
        
        logger.info(f"Similarity index created: {index['n_documents']} docs, {index['n_features']} features")
        
        return index
    
    def query_similarity_index(self, 
                             index: Dict[str, Any],
                             query_vector: sparse.spmatrix,
                             top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        Query the similarity index
        
        Args:
            index: Similarity index created by create_similarity_index
            query_vector: Query vector (1, n_features)
            top_k: Number of top results to return
        
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        
        # Normalize query vector
        query_norm = np.sqrt(query_vector.multiply(query_vector).sum())
        if query_norm > 0:
            query_normalized = query_vector / query_norm
        else:
            query_normalized = query_vector
        
        # Compute similarities
        similarities = query_normalized @ index["matrix"].T
        similarities = similarities.toarray().flatten()
        
        # Get top-k
        top_k_indices = np.argsort(-similarities)[:top_k]
        top_k_scores = similarities[top_k_indices]
        
        return top_k_indices, top_k_scores
    
    def get_memory_usage(self, matrix: sparse.spmatrix) -> Dict[str, Any]:
        """Get detailed memory usage information"""
        
        memory_info = {
            "data_bytes": matrix.data.nbytes,
            "indices_bytes": matrix.indices.nbytes if hasattr(matrix, 'indices') else 0,
            "indptr_bytes": matrix.indptr.nbytes if hasattr(matrix, 'indptr') else 0,
        }
        
        memory_info["total_bytes"] = sum(memory_info.values())
        memory_info["total_mb"] = memory_info["total_bytes"] / (1024 * 1024)
        
        # Calculate density
        total_elements = matrix.shape[0] * matrix.shape[1]
        memory_info["density"] = matrix.nnz / total_elements
        memory_info["sparsity"] = 1.0 - memory_info["density"]
        
        # Compare to dense matrix
        dense_bytes = total_elements * matrix.dtype.itemsize
        memory_info["dense_matrix_mb"] = dense_bytes / (1024 * 1024)
        memory_info["compression_ratio"] = dense_bytes / memory_info["total_bytes"]
        
        return memory_info
    
    def benchmark_operations(self, 
                           matrix_sizes: list = [(1000, 5000), (5000, 10000)],
                           sparsity: float = 0.95) -> Dict[str, Any]:
        """
        Benchmark sparse matrix operations
        
        Args:
            matrix_sizes: List of (n_samples, n_features) tuples to test
            sparsity: Sparsity level (0.95 = 95% zeros)
        
        Returns:
            Benchmark results
        """
        
        import time
        
        results = {}
        
        for n_samples, n_features in matrix_sizes:
            logger.info(f"Benchmarking {n_samples}x{n_features} matrix")
            
            # Create test matrix
            np.random.seed(42)
            dense_matrix = np.random.rand(n_samples, n_features)
            
            # Make sparse
            mask = np.random.rand(n_samples, n_features) < (1 - sparsity)
            dense_matrix[~mask] = 0
            sparse_matrix = sparse.csr_matrix(dense_matrix)
            
            # Optimize matrix
            start_time = time.time()
            optimized_matrix = self.optimize_tfidf_matrix(sparse_matrix)
            optimization_time = time.time() - start_time
            
            # Test cosine similarity
            start_time = time.time()
            similarities = self.fast_cosine_similarity(optimized_matrix[:100], optimized_matrix)
            similarity_time = time.time() - start_time
            
            # Memory usage
            memory_info = self.get_memory_usage(optimized_matrix)
            
            size_key = f"{n_samples}x{n_features}"
            results[size_key] = {
                "optimization_time": optimization_time,
                "similarity_time": similarity_time,
                "memory_mb": memory_info["total_mb"],
                "compression_ratio": memory_info["compression_ratio"],
                "sparsity": memory_info["sparsity"]
            }
        
        return results
    
    def get_optimization_stats(self) -> Dict[str, Any]:
        """Get optimization statistics"""
        
        stats = {
            "matrix_format": self.matrix_format,
            "dtype": str(self.dtype),
            "memory_efficient": self.memory_efficient,
            "operations_performed": len(self.operation_times),
        }
        
        if self.compression_ratios:
            stats.update({
                "avg_compression_ratio": np.mean(self.compression_ratios),
                "max_compression_ratio": np.max(self.compression_ratios),
                "min_compression_ratio": np.min(self.compression_ratios)
            })
        
        if self.operation_times:
            stats.update({
                "avg_operation_time": np.mean(self.operation_times),
                "total_operation_time": np.sum(self.operation_times)
            })
        
        return stats


def test_sparse_optimizer():
    """Test the sparse matrix optimizer"""
    
    print("=== Sparse Matrix Optimizer Test ===")
    
    # Create optimizer
    optimizer = SparseMatrixOptimizer()
    
    # Create test TF-IDF matrix
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    # Sample documents
    documents = [
        "This is a sample document about machine learning",
        "Another document discussing natural language processing",
        "Text similarity and information retrieval methods",
        "Deep learning and neural networks for NLP",
        "Document clustering and classification techniques"
    ] * 200  # Repeat for larger test
    
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(documents)
    
    print(f"Original TF-IDF matrix: {tfidf_matrix.shape}, {tfidf_matrix.nnz} non-zeros")
    
    # Optimize matrix
    optimized_matrix, info = optimizer.optimize_vectorizer_output(vectorizer, documents[:100])
    print(f"Optimization info: {info}")
    
    # Test similarity calculation
    print("\nTesting similarity calculation...")
    similarities = optimizer.fast_cosine_similarity(optimized_matrix[:10], optimized_matrix)
    print(f"Similarity matrix shape: {similarities.shape}")
    
    # Create similarity index
    print("\nCreating similarity index...")
    index = optimizer.create_similarity_index(optimized_matrix)
    
    # Query index
    query_vector = optimized_matrix[0]
    top_indices, top_scores = optimizer.query_similarity_index(index, query_vector, top_k=5)
    print(f"Top-5 similar documents: indices={top_indices}, scores={top_scores}")
    
    # Memory usage
    memory_info = optimizer.get_memory_usage(optimized_matrix)
    print(f"Memory usage: {memory_info}")
    
    # Optimization stats
    stats = optimizer.get_optimization_stats()
    print(f"Optimization stats: {stats}")
    
    return optimizer


if __name__ == "__main__":
    test_sparse_optimizer()