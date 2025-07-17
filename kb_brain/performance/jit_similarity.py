"""
JIT-Compiled Similarity Engine using Numba
High-performance similarity calculations with Just-In-Time compilation
"""

import numpy as np
import logging
from typing import Tuple, Optional, Union, Dict, Any
import warnings

logger = logging.getLogger(__name__)

# Try to import Numba for JIT compilation
try:
    from numba import jit, prange, types
    from numba.typed import Dict as NumbaDict
    NUMBA_AVAILABLE = True
    logger.info("Numba JIT compilation available")
except ImportError:
    NUMBA_AVAILABLE = False
    logger.info("Numba not available - using standard NumPy implementations")
    
    # Create dummy decorator for when Numba is not available
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    
    def prange(x):
        return range(x)


class JITSimilarityEngine:
    """High-performance similarity calculations with JIT compilation"""
    
    def __init__(self, use_jit: bool = True, parallel: bool = True):
        """
        Initialize JIT similarity engine
        
        Args:
            use_jit: Whether to use JIT compilation (if available)
            parallel: Whether to use parallel computation
        """
        self.use_jit = use_jit and NUMBA_AVAILABLE
        self.parallel = parallel and NUMBA_AVAILABLE
        self.compiled_functions = {}
        
        if self.use_jit:
            self._compile_functions()
            logger.info(f"JIT similarity engine initialized (parallel={self.parallel})")
        else:
            logger.info("JIT similarity engine using standard NumPy")
    
    def _compile_functions(self):
        """Pre-compile JIT functions for better performance"""
        
        if not self.use_jit:
            return
        
        try:
            # Compile with dummy data to cache
            dummy_a = np.random.rand(100, 50).astype(np.float32)
            dummy_b = np.random.rand(200, 50).astype(np.float32)
            
            # Trigger compilation
            self.cosine_similarity_jit(dummy_a, dummy_b)
            self.euclidean_distance_jit(dummy_a, dummy_b)
            
            logger.info("JIT functions pre-compiled successfully")
            
        except Exception as e:
            logger.warning(f"JIT compilation failed: {e}")
            self.use_jit = False
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _cosine_similarity_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Numba-compiled cosine similarity calculation"""
        
        n_samples_A, n_features = A.shape
        n_samples_B = B.shape[0]
        
        # Pre-compute norms
        norms_A = np.empty(n_samples_A, dtype=A.dtype)
        norms_B = np.empty(n_samples_B, dtype=B.dtype)
        
        for i in prange(n_samples_A):
            norms_A[i] = np.sqrt(np.sum(A[i] * A[i]))
        
        for i in prange(n_samples_B):
            norms_B[i] = np.sqrt(np.sum(B[i] * B[i]))
        
        # Compute similarities
        similarities = np.empty((n_samples_A, n_samples_B), dtype=A.dtype)
        
        for i in prange(n_samples_A):
            for j in range(n_samples_B):
                dot_product = 0.0
                for k in range(n_features):
                    dot_product += A[i, k] * B[j, k]
                
                norm_product = norms_A[i] * norms_B[j]
                if norm_product == 0.0:
                    similarities[i, j] = 0.0
                else:
                    similarities[i, j] = dot_product / norm_product
        
        return similarities
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _euclidean_distance_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Numba-compiled Euclidean distance calculation"""
        
        n_samples_A, n_features = A.shape
        n_samples_B = B.shape[0]
        
        distances = np.empty((n_samples_A, n_samples_B), dtype=A.dtype)
        
        for i in prange(n_samples_A):
            for j in range(n_samples_B):
                distance = 0.0
                for k in range(n_features):
                    diff = A[i, k] - B[j, k]
                    distance += diff * diff
                distances[i, j] = np.sqrt(distance)
        
        return distances
    
    @staticmethod
    @jit(nopython=True, parallel=True, cache=True)
    def _manhattan_distance_numba(A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """Numba-compiled Manhattan distance calculation"""
        
        n_samples_A, n_features = A.shape
        n_samples_B = B.shape[0]
        
        distances = np.empty((n_samples_A, n_samples_B), dtype=A.dtype)
        
        for i in prange(n_samples_A):
            for j in range(n_samples_B):
                distance = 0.0
                for k in range(n_features):
                    distance += abs(A[i, k] - B[j, k])
                distances[i, j] = distance
        
        return distances
    
    @staticmethod
    @jit(nopython=True, cache=True)
    def _top_k_indices_numba(similarities: np.ndarray, k: int) -> np.ndarray:
        """Numba-compiled top-k indices extraction"""
        
        n_samples = similarities.shape[0]
        n_candidates = similarities.shape[1]
        
        if k > n_candidates:
            k = n_candidates
        
        top_k = np.empty((n_samples, k), dtype=np.int32)
        
        for i in range(n_samples):
            # Get indices sorted by similarity (descending)
            row = similarities[i]
            indices = np.argsort(-row)[:k]
            top_k[i] = indices
        
        return top_k
    
    def cosine_similarity_jit(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute cosine similarity with JIT compilation
        
        Args:
            A: First matrix (n_samples_A, n_features)
            B: Second matrix (n_samples_B, n_features), if None uses A
        
        Returns:
            Similarity matrix (n_samples_A, n_samples_B)
        """
        
        if B is None:
            B = A
        
        # Ensure float32 for better performance
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        
        if self.use_jit:
            return self._cosine_similarity_numba(A, B)
        else:
            # Fallback to sklearn
            from sklearn.metrics.pairwise import cosine_similarity
            return cosine_similarity(A, B)
    
    def euclidean_distance_jit(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Euclidean distance with JIT compilation
        
        Args:
            A: First matrix (n_samples_A, n_features) 
            B: Second matrix (n_samples_B, n_features), if None uses A
        
        Returns:
            Distance matrix (n_samples_A, n_samples_B)
        """
        
        if B is None:
            B = A
        
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        
        if self.use_jit:
            return self._euclidean_distance_numba(A, B)
        else:
            # Fallback to sklearn
            from sklearn.metrics.pairwise import euclidean_distances
            return euclidean_distances(A, B)
    
    def manhattan_distance_jit(self, A: np.ndarray, B: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute Manhattan distance with JIT compilation
        
        Args:
            A: First matrix (n_samples_A, n_features)
            B: Second matrix (n_samples_B, n_features), if None uses A
        
        Returns:
            Distance matrix (n_samples_A, n_samples_B)
        """
        
        if B is None:
            B = A
        
        A = np.asarray(A, dtype=np.float32)
        B = np.asarray(B, dtype=np.float32)
        
        if self.use_jit:
            return self._manhattan_distance_numba(A, B)
        else:
            # Fallback to sklearn
            from sklearn.metrics.pairwise import manhattan_distances
            return manhattan_distances(A, B)
    
    def find_top_k_similar(self, 
                          query: np.ndarray, 
                          candidates: np.ndarray, 
                          k: int = 5,
                          metric: str = "cosine") -> Tuple[np.ndarray, np.ndarray]:
        """
        Find top-k most similar items
        
        Args:
            query: Query vector(s) (n_queries, n_features)
            candidates: Candidate vectors (n_candidates, n_features)
            k: Number of top results to return
            metric: Similarity metric ("cosine", "euclidean", "manhattan")
        
        Returns:
            Tuple of (top_k_indices, top_k_scores)
        """
        
        if metric == "cosine":
            similarities = self.cosine_similarity_jit(query, candidates)
            descending = True
        elif metric == "euclidean":
            similarities = self.euclidean_distance_jit(query, candidates)
            descending = False  # Lower distance is better
        elif metric == "manhattan":
            similarities = self.manhattan_distance_jit(query, candidates)
            descending = False
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Get top-k indices
        if self.use_jit and metric == "cosine":
            # Use JIT-compiled top-k for cosine similarity
            if not descending:
                similarities = -similarities  # Convert to descending order
            top_k_indices = self._top_k_indices_numba(similarities, k)
        else:
            # Use NumPy for other cases
            if descending:
                top_k_indices = np.argsort(-similarities, axis=1)[:, :k]
            else:
                top_k_indices = np.argsort(similarities, axis=1)[:, :k]
        
        # Get corresponding scores
        top_k_scores = np.empty((similarities.shape[0], k), dtype=similarities.dtype)
        for i in range(similarities.shape[0]):
            for j in range(k):
                if j < similarities.shape[1]:
                    top_k_scores[i, j] = similarities[i, top_k_indices[i, j]]
                else:
                    top_k_scores[i, j] = 0.0
        
        return top_k_indices, top_k_scores
    
    def benchmark_performance(self, 
                            n_queries: int = 1000, 
                            n_candidates: int = 10000,
                            n_features: int = 100) -> Dict[str, float]:
        """
        Benchmark JIT vs non-JIT performance
        
        Args:
            n_queries: Number of query vectors
            n_candidates: Number of candidate vectors  
            n_features: Feature dimensionality
        
        Returns:
            Performance comparison results
        """
        
        import time
        
        # Generate test data
        np.random.seed(42)
        queries = np.random.rand(n_queries, n_features).astype(np.float32)
        candidates = np.random.rand(n_candidates, n_features).astype(np.float32)
        
        results = {}
        
        # Benchmark JIT version
        if self.use_jit:
            start_time = time.time()
            jit_similarities = self.cosine_similarity_jit(queries, candidates)
            jit_time = time.time() - start_time
            results["jit_time"] = jit_time
            results["jit_available"] = True
        else:
            results["jit_available"] = False
        
        # Benchmark standard version
        try:
            from sklearn.metrics.pairwise import cosine_similarity
            start_time = time.time()
            sklearn_similarities = cosine_similarity(queries, candidates)
            sklearn_time = time.time() - start_time
            results["sklearn_time"] = sklearn_time
            
            if self.use_jit:
                results["speedup"] = sklearn_time / jit_time
                # Verify results are similar
                max_diff = np.max(np.abs(jit_similarities - sklearn_similarities))
                results["max_difference"] = max_diff
                results["results_match"] = max_diff < 1e-6
            
        except ImportError:
            results["sklearn_available"] = False
        
        results.update({
            "n_queries": n_queries,
            "n_candidates": n_candidates,
            "n_features": n_features,
            "total_operations": n_queries * n_candidates
        })
        
        return results
    
    def get_engine_info(self) -> Dict[str, Any]:
        """Get information about the JIT engine"""
        
        return {
            "numba_available": NUMBA_AVAILABLE,
            "jit_enabled": self.use_jit,
            "parallel_enabled": self.parallel,
            "compiled_functions": list(self.compiled_functions.keys()),
            "engine_type": "JIT" if self.use_jit else "NumPy"
        }


def test_jit_engine():
    """Test the JIT similarity engine"""
    
    print("=== JIT Similarity Engine Test ===")
    
    # Create engine
    engine = JITSimilarityEngine()
    
    # Get engine info
    info = engine.get_engine_info()
    print(f"Engine Info: {info}")
    
    # Test similarity calculation
    np.random.seed(42)
    A = np.random.rand(100, 50).astype(np.float32)
    B = np.random.rand(200, 50).astype(np.float32)
    
    print("\nTesting cosine similarity...")
    similarities = engine.cosine_similarity_jit(A, B)
    print(f"Similarity matrix shape: {similarities.shape}")
    print(f"Similarity range: [{similarities.min():.4f}, {similarities.max():.4f}]")
    
    # Test top-k search
    print("\nTesting top-k search...")
    query = A[:5]  # First 5 vectors as queries
    top_k_indices, top_k_scores = engine.find_top_k_similar(query, B, k=3)
    print(f"Top-k indices shape: {top_k_indices.shape}")
    print(f"Top-k scores shape: {top_k_scores.shape}")
    
    # Benchmark performance
    print("\nBenchmarking performance...")
    benchmark = engine.benchmark_performance(n_queries=500, n_candidates=5000, n_features=100)
    print(f"Benchmark results: {benchmark}")
    
    return engine


if __name__ == "__main__":
    test_jit_engine()