"""
Performance Integration Module
Integrates all performance optimizations into existing KB Brain components
"""

import logging
from typing import Dict, Any, Optional, Union
import numpy as np
from scipy import sparse

from .cpu_optimizer import CPUOptimizer, enable_intel_optimizations
from .jit_similarity import JITSimilarityEngine
from .sparse_optimizer import SparseMatrixOptimizer

logger = logging.getLogger(__name__)


class PerformanceManager:
    """Unified performance optimization manager for KB Brain"""
    
    def __init__(self, auto_optimize: bool = True):
        """
        Initialize performance manager
        
        Args:
            auto_optimize: Whether to automatically apply optimizations
        """
        self.auto_optimize = auto_optimize
        
        # Initialize optimizers
        self.cpu_optimizer = CPUOptimizer(auto_optimize=auto_optimize)
        self.jit_engine = JITSimilarityEngine(use_jit=True, parallel=True)
        self.sparse_optimizer = SparseMatrixOptimizer(
            matrix_format="csr",
            dtype="float32",
            memory_efficient=True
        )
        
        # Performance tracking
        self.performance_metrics = {
            "optimizations_applied": [],
            "speedup_measurements": {},
            "memory_savings": {}
        }
        
        if auto_optimize:
            self._apply_global_optimizations()
        
        logger.info("Performance Manager initialized")
    
    def _apply_global_optimizations(self):
        """Apply global performance optimizations"""
        
        # CPU optimizations
        cpu_results = self.cpu_optimizer.apply_optimizations()
        for opt_name, result in cpu_results.items():
            if result.get("enabled", False):
                self.performance_metrics["optimizations_applied"].append(opt_name)
        
        logger.info(f"Applied {len(self.performance_metrics['optimizations_applied'])} global optimizations")
    
    def optimize_similarity_computation(self, 
                                      embeddings_matrix: Union[np.ndarray, sparse.spmatrix],
                                      query_vector: Union[np.ndarray, sparse.spmatrix],
                                      top_k: int = 10,
                                      metric: str = "cosine") -> Dict[str, Any]:
        """
        Optimized similarity computation using best available method
        
        Args:
            embeddings_matrix: Matrix of embeddings to search
            query_vector: Query vector(s)
            top_k: Number of top results
            metric: Similarity metric
        
        Returns:
            Optimized similarity results with performance info
        """
        
        import time
        start_time = time.time()
        
        # Determine best optimization strategy
        if sparse.issparse(embeddings_matrix):
            # Use sparse optimization
            similarities = self._compute_sparse_similarity(
                embeddings_matrix, query_vector, top_k, metric
            )
            method = "sparse_optimized"
        else:
            # Use JIT optimization for dense matrices
            similarities = self._compute_jit_similarity(
                embeddings_matrix, query_vector, top_k, metric
            )
            method = "jit_optimized"
        
        processing_time = time.time() - start_time
        
        return {
            "similarities": similarities,
            "method": method,
            "processing_time": processing_time,
            "matrix_shape": embeddings_matrix.shape,
            "optimizations_used": self.performance_metrics["optimizations_applied"]
        }
    
    def _compute_sparse_similarity(self, 
                                 embeddings_matrix: sparse.spmatrix,
                                 query_vector: Union[np.ndarray, sparse.spmatrix],
                                 top_k: int,
                                 metric: str) -> Dict[str, Any]:
        """Compute similarity using sparse optimization"""
        
        # Ensure query is sparse
        if not sparse.issparse(query_vector):
            query_vector = sparse.csr_matrix(query_vector)
        
        # Optimize matrices
        opt_embeddings = self.sparse_optimizer.optimize_tfidf_matrix(embeddings_matrix)
        opt_query = self.sparse_optimizer.optimize_tfidf_matrix(query_vector)
        
        if metric == "cosine":
            similarities = self.sparse_optimizer.fast_cosine_similarity(
                opt_query, opt_embeddings, top_k=top_k
            )
        else:
            raise ValueError(f"Metric {metric} not supported for sparse matrices")
        
        # Get top-k results
        if sparse.issparse(similarities):
            similarities = similarities.toarray()
        
        top_k_indices = np.argsort(-similarities[0])[:top_k]
        top_k_scores = similarities[0][top_k_indices]
        
        return {
            "indices": top_k_indices,
            "scores": top_k_scores,
            "sparse_optimization": True
        }
    
    def _compute_jit_similarity(self,
                              embeddings_matrix: np.ndarray,
                              query_vector: np.ndarray,
                              top_k: int,
                              metric: str) -> Dict[str, Any]:
        """Compute similarity using JIT optimization"""
        
        # Ensure proper shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)
        
        # Use JIT engine
        top_k_indices, top_k_scores = self.jit_engine.find_top_k_similar(
            query_vector, embeddings_matrix, k=top_k, metric=metric
        )
        
        return {
            "indices": top_k_indices[0],  # Return first query results
            "scores": top_k_scores[0],
            "jit_optimization": True
        }
    
    def create_optimized_kb_brain(self, kb_brain_class):
        """
        Create an optimized version of KB Brain with performance enhancements
        
        Args:
            kb_brain_class: Original KB Brain class
        
        Returns:
            Performance-optimized KB Brain instance
        """
        
        class OptimizedKBBrain(kb_brain_class):
            """Performance-optimized KB Brain"""
            
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.performance_manager = PerformanceManager(auto_optimize=True)
                logger.info("KB Brain enhanced with performance optimizations")
            
            def find_best_solution_hybrid(self, problem, top_k=5, **kwargs):
                """Optimized solution finding"""
                
                # Get embeddings (assuming they exist)
                if hasattr(self, 'embeddings_matrix') and self.embeddings_matrix is not None:
                    # Use optimized similarity computation
                    query_embedding = self._embed_query(problem)
                    
                    result = self.performance_manager.optimize_similarity_computation(
                        self.embeddings_matrix,
                        query_embedding,
                        top_k=top_k,
                        metric="cosine"
                    )
                    
                    # Convert to expected format
                    solutions = []
                    for idx, score in zip(result["similarities"]["indices"], 
                                        result["similarities"]["scores"]):
                        solutions.append({
                            "solution": self.solutions[idx] if hasattr(self, 'solutions') else f"Solution {idx}",
                            "confidence": float(score),
                            "method": result["method"]
                        })
                    
                    return solutions
                else:
                    # Fallback to original method
                    return super().find_best_solution_hybrid(problem, top_k=top_k, **kwargs)
            
            def get_system_status(self):
                """Enhanced system status with performance info"""
                status = super().get_system_status()
                
                # Add performance information
                status["performance_optimizations"] = self.performance_manager.get_optimization_status()
                
                return status
        
        return OptimizedKBBrain
    
    def integrate_with_sme_system(self, sme_system):
        """
        Integrate performance optimizations with SME system
        
        Args:
            sme_system: SME agent system instance
        """
        
        # Optimize each SME agent's KB engine
        for agent_id, agent in sme_system.agents.items():
            if hasattr(agent, 'kb_engine'):
                self._optimize_kb_engine(agent.kb_engine)
        
        logger.info(f"Performance optimizations integrated with {len(sme_system.agents)} SME agents")
    
    def _optimize_kb_engine(self, kb_engine):
        """Optimize a KB engine instance"""
        
        # Add performance manager to KB engine
        kb_engine.performance_manager = self
        
        # Monkey patch similarity computation
        original_search = kb_engine.search_knowledge
        
        def optimized_search(*args, **kwargs):
            # Use optimized search if possible
            result = original_search(*args, **kwargs)
            
            # Add performance info
            if hasattr(result, 'metadata'):
                result.metadata = result.metadata or {}
                result.metadata["performance_optimized"] = True
            
            return result
        
        kb_engine.search_knowledge = optimized_search
    
    def benchmark_system_performance(self, test_sizes: list = None) -> Dict[str, Any]:
        """
        Comprehensive system performance benchmark
        
        Args:
            test_sizes: List of test sizes to benchmark
        
        Returns:
            Benchmark results
        """
        
        if test_sizes is None:
            test_sizes = [1000, 5000, 10000]
        
        results = {
            "cpu_optimization": self.cpu_optimizer.get_optimization_status(),
            "jit_benchmarks": {},
            "sparse_benchmarks": {},
            "system_info": self.cpu_optimizer.get_optimization_status()
        }
        
        # JIT benchmarks
        for size in test_sizes:
            jit_result = self.jit_engine.benchmark_performance(
                n_queries=min(100, size//10),
                n_candidates=size,
                n_features=100
            )
            results["jit_benchmarks"][f"size_{size}"] = jit_result
        
        # Sparse benchmarks
        sparse_result = self.sparse_optimizer.benchmark_operations(
            matrix_sizes=[(size, 1000) for size in test_sizes[:2]]  # Limit for memory
        )
        results["sparse_benchmarks"] = sparse_result
        
        return results
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get comprehensive optimization status"""
        
        return {
            "cpu_optimizations": self.cpu_optimizer.get_optimization_status(),
            "jit_engine": self.jit_engine.get_engine_info(),
            "sparse_optimizer": self.sparse_optimizer.get_optimization_stats(),
            "performance_metrics": self.performance_metrics,
            "global_optimizations_applied": len(self.performance_metrics["optimizations_applied"])
        }
    
    def suggest_optimizations(self, system_context: Dict[str, Any]) -> list:
        """
        Suggest additional optimizations based on system context
        
        Args:
            system_context: Current system context and usage patterns
        
        Returns:
            List of optimization suggestions
        """
        
        suggestions = []
        
        # Check if Intel optimizations are available but not enabled
        if not self.cpu_optimizer.intel_available:
            suggestions.append({
                "type": "install",
                "description": "Install Intel Extension for scikit-learn for 2-10x speedup",
                "command": "pip install intel-extension-for-scikit-learn",
                "impact": "high"
            })
        
        # Check JIT availability
        if not self.jit_engine.use_jit:
            suggestions.append({
                "type": "install", 
                "description": "Install Numba for JIT compilation speedup",
                "command": "pip install numba",
                "impact": "high"
            })
        
        # Memory optimization suggestions
        if system_context.get("memory_usage_high", False):
            suggestions.append({
                "type": "configuration",
                "description": "Enable memory-efficient sparse matrix operations",
                "action": "Set memory_efficient=True in sparse optimizer",
                "impact": "medium"
            })
        
        # CPU utilization suggestions  
        cpu_count = system_context.get("cpu_count", 1)
        if cpu_count > 4:
            suggestions.append({
                "type": "configuration",
                "description": f"Optimize threading for {cpu_count} CPU cores",
                "action": "Enable parallel processing in all components",
                "impact": "medium"
            })
        
        return suggestions


def test_performance_integration():
    """Test the performance integration system"""
    
    print("=== Performance Integration Test ===")
    
    # Create performance manager
    manager = PerformanceManager(auto_optimize=True)
    
    # Get optimization status
    status = manager.get_optimization_status()
    print(f"Optimization status: {status}")
    
    # Test similarity computation
    print("\nTesting optimized similarity computation...")
    np.random.seed(42)
    embeddings = np.random.rand(1000, 100).astype(np.float32)
    query = np.random.rand(1, 100).astype(np.float32)
    
    result = manager.optimize_similarity_computation(embeddings, query, top_k=5)
    print(f"Similarity result: {result['method']}, time: {result['processing_time']:.3f}s")
    
    # Test sparse optimization
    print("\nTesting sparse optimization...")
    from scipy import sparse
    sparse_embeddings = sparse.random(1000, 1000, density=0.1, format='csr')
    sparse_query = sparse.random(1, 1000, density=0.1, format='csr')
    
    sparse_result = manager.optimize_similarity_computation(sparse_embeddings, sparse_query, top_k=5)
    print(f"Sparse result: {sparse_result['method']}, time: {sparse_result['processing_time']:.3f}s")
    
    # Benchmark performance
    print("\nBenchmarking system performance...")
    benchmark = manager.benchmark_system_performance(test_sizes=[1000, 2000])
    print(f"Benchmark completed with {len(benchmark)} result categories")
    
    # Get optimization suggestions
    system_context = {"memory_usage_high": False, "cpu_count": 8}
    suggestions = manager.suggest_optimizations(system_context)
    print(f"Optimization suggestions: {len(suggestions)} recommendations")
    
    return manager


if __name__ == "__main__":
    test_performance_integration()