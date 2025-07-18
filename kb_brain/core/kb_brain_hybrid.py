#!/usr/bin/env python3
"""
KB Brain Hybrid GPU - Uses CuPy for GPU operations with scikit-learn fallback
Optimized for environments where CuPy works but CuML is unavailable
"""

import json
import numpy as np
import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import hashlib
import warnings

# Import performance optimizations
try:
    from ..performance.performance_integration import PerformanceManager
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    warnings.warn("Performance optimizations not available")

# Test CuPy availability
try:
    import cupy as cp
    CUPY_AVAILABLE = True
    print("✅ CuPy available for GPU acceleration")
except ImportError:
    CUPY_AVAILABLE = False
    print("❌ CuPy not available")

# CPU fallback with scikit-learn
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors
    from sklearn.cluster import KMeans
    from sklearn.metrics.pairwise import cosine_similarity
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    warnings.warn("scikit-learn not available")

@dataclass
class HybridSolutionCandidate:
    """Solution candidate with hybrid GPU/CPU computation"""
    solution_id: str
    problem_context: str
    solution_text: str
    success_rate: float
    confidence: float
    similarity_score: float
    source_kb: str
    last_used: str
    tags: List[str]
    failed_approaches: List[str]
    embedding: Optional[np.ndarray] = None

class HybridGPUKBBrain:
    """Hybrid GPU/CPU KB Brain using CuPy + scikit-learn with performance optimizations"""
    
    def __init__(self, kb_root: Optional[str] = None, 
                 enable_performance_optimizations: bool = True):
        # Import settings here to avoid circular imports
        from ..config.settings import Settings
        
        self.kb_root = Path(kb_root) if kb_root else Settings.KB_DATA_PATH
        self.use_gpu = CUPY_AVAILABLE
        
        # Initialize performance manager
        self.performance_manager = None
        if enable_performance_optimizations and PERFORMANCE_AVAILABLE:
            self.performance_manager = PerformanceManager(auto_optimize=True)
            print("🚀 Performance optimizations enabled (Intel extensions + JIT compilation)")
        
        # Initialize models
        self.vectorizer = None
        self.similarity_model = None
        self.clustering_model = None
        
        # Knowledge storage
        self.knowledge_embeddings = {}
        self.solution_texts = {}
        self.solution_metadata = {}
        
        # Brain state
        self.brain_state_file = self.kb_root / "brain_state_hybrid.json"
        self.embeddings_file = self.kb_root / "knowledge_embeddings_hybrid.npz"
        
        print(f"🧠 Hybrid KB Brain initialized with {'CuPy GPU' if self.use_gpu else 'CPU'} acceleration")
        
        # Load existing state
        self.load_brain_state()
        self.initialize_models()
    
    def initialize_models(self):
        """Initialize hybrid GPU/CPU models"""
        if not SKLEARN_AVAILABLE:
            print("❌ scikit-learn not available - limited functionality")
            return
            
        # Use scikit-learn for ML operations (reliable)
        self.vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            lowercase=True
        )
        
        self.similarity_model = NearestNeighbors(
            n_neighbors=10,
            algorithm='brute',
            metric='cosine'
        )
        
        self.clustering_model = KMeans(
            n_clusters=50,
            random_state=42
        )
        
        print("✅ Hybrid models initialized (scikit-learn + CuPy)")
    
    def load_brain_state(self):
        """Load brain state and embeddings"""
        if self.brain_state_file.exists():
            with open(self.brain_state_file, 'r') as f:
                state = json.load(f)
                self.solution_metadata = state.get('solution_metadata', {})
        
        if self.embeddings_file.exists():
            try:
                data = np.load(self.embeddings_file, allow_pickle=True)
                self.knowledge_embeddings = data['embeddings'].item()
                self.solution_texts = data['texts'].item()
                print(f"📚 Loaded {len(self.knowledge_embeddings)} cached embeddings")
            except Exception as e:
                print(f"⚠️  Error loading embeddings: {e}")
    
    def save_brain_state(self):
        """Save brain state and embeddings"""
        state = {
            'solution_metadata': self.solution_metadata,
            'last_updated': datetime.datetime.now().isoformat(),
            'gpu_enabled': self.use_gpu,
            'hybrid_mode': True
        }
        
        with open(self.brain_state_file, 'w') as f:
            json.dump(state, f, indent=2)
        
        if self.knowledge_embeddings:
            np.savez_compressed(
                self.embeddings_file,
                embeddings=self.knowledge_embeddings,
                texts=self.solution_texts
            )
    
    def _optimized_similarity_computation(self, query_vector: np.ndarray, embeddings_matrix: np.ndarray, top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Performance-optimized similarity computation with fallback strategy"""
        
        # Try performance-optimized computation first
        if self.performance_manager:
            try:
                result = self.performance_manager.optimize_similarity_computation(
                    embeddings_matrix, query_vector, top_k=top_k, metric="cosine"
                )
                
                similarities = result["similarities"]
                indices = similarities["indices"]
                scores = similarities["scores"]
                distances = 1 - scores  # Convert similarity to distance
                
                print(f"🚀 Used {result['method']} optimization (took {result['processing_time']:.3f}s)")
                return distances, indices
                
            except Exception as e:
                print(f"⚠️  Performance-optimized computation failed: {e}")
                print("🔄 Falling back to GPU/CPU...")
        
        # GPU acceleration fallback
        if self.use_gpu:
            try:
                # Move to GPU
                query_gpu = cp.asarray(query_vector)
                embeddings_gpu = cp.asarray(embeddings_matrix)
                
                # Compute cosine similarities on GPU
                query_norm = cp.linalg.norm(query_gpu, axis=1, keepdims=True)
                embeddings_norm = cp.linalg.norm(embeddings_gpu, axis=1, keepdims=True)
                
                # Normalized dot product = cosine similarity
                similarities = cp.dot(query_gpu, embeddings_gpu.T) / (query_norm * embeddings_norm.T)
                similarities = similarities.flatten()
                
                # Convert to distances and get top indices
                distances = 1 - similarities
                indices = cp.argsort(distances)[:top_k]
                
                # Move back to CPU
                distances_cpu = cp.asnumpy(distances[indices])
                indices_cpu = cp.asnumpy(indices)
                
                print("🎯 Used GPU acceleration")
                return distances_cpu, indices_cpu
                
            except Exception as e:
                print(f"⚠️  GPU similarity computation failed: {e}")
                print("🔄 Falling back to CPU...")
        
        # CPU fallback
        similarities = cosine_similarity(query_vector, embeddings_matrix).flatten()
        distances = 1 - similarities
        indices = np.argsort(distances)[:top_k]
        print("💻 Used CPU fallback")
        return distances[indices], indices
    
    def rebuild_knowledge_index(self):
        """Rebuild knowledge index with hybrid GPU/CPU processing"""
        print("🔄 Rebuilding knowledge index with hybrid processing...")
        
        # Collect all knowledge text
        all_texts = []
        text_to_id = {}
        
        # Process all KB files
        for kb_file in self.kb_root.rglob("*.json"):
            try:
                with open(kb_file, 'r') as f:
                    data = json.load(f)
                    
                    # Extract text content
                    text_content = self._extract_text_content(data)
                    if text_content:
                        solution_id = f"{kb_file.stem}_{hashlib.md5(text_content.encode()).hexdigest()[:8]}"
                        all_texts.append(text_content)
                        text_to_id[len(all_texts) - 1] = solution_id
                        self.solution_texts[solution_id] = text_content
                        
            except Exception as e:
                print(f"⚠️  Error processing {kb_file}: {e}")
        
        if not all_texts:
            print("❌ No text content found for indexing")
            return
        
        # Create embeddings using scikit-learn
        if self.vectorizer:
            try:
                print(f"📊 Vectorizing {len(all_texts)} documents...")
                
                # Fit and transform with scikit-learn
                tfidf_matrix = self.vectorizer.fit_transform(all_texts)
                embeddings_array = tfidf_matrix.toarray()
                
                # Store embeddings
                for i, (text_idx, solution_id) in enumerate(text_to_id.items()):
                    self.knowledge_embeddings[solution_id] = embeddings_array[i]
                
                # Fit similarity model
                self.similarity_model.fit(embeddings_array)
                
                # Optional clustering
                if len(all_texts) > 10:
                    cluster_labels = self.clustering_model.fit_predict(embeddings_array)
                    self._organize_knowledge_clusters(text_to_id, cluster_labels)
                
                print(f"✅ Knowledge index rebuilt with {len(self.knowledge_embeddings)} embeddings")
                print(f"🎯 GPU acceleration: {'enabled' if self.use_gpu else 'disabled'}")
                self.save_brain_state()
                
            except Exception as e:
                print(f"❌ Error building embeddings: {e}")
    
    def find_best_solution_hybrid(self, problem_description: str, 
                                context: Dict = None, top_k: int = 5) -> List[HybridSolutionCandidate]:
        """Find solutions using hybrid GPU/CPU processing"""
        
        if not self.vectorizer or not self.knowledge_embeddings:
            print("🔄 Knowledge index not built. Building now...")
            self.rebuild_knowledge_index()
        
        if not self.knowledge_embeddings:
            print("❌ No knowledge available for search")
            return []
        
        try:
            # Vectorize the problem description
            problem_vector = self.vectorizer.transform([problem_description])
            problem_array = problem_vector.toarray()
            
            # Create embeddings matrix
            solution_ids = list(self.knowledge_embeddings.keys())
            embeddings_matrix = np.array([self.knowledge_embeddings[sid] for sid in solution_ids])
            
            # Performance-optimized similarity search
            distances, indices = self._optimized_similarity_computation(problem_array, embeddings_matrix, top_k)
            
            # Create hybrid candidates
            candidates = []
            
            for i, (distance, idx) in enumerate(zip(distances, indices)):
                if idx < len(solution_ids):
                    solution_id = solution_ids[idx]
                    similarity_score = 1.0 - distance
                    
                    # Get solution details
                    solution_text = self.solution_texts.get(solution_id, "")
                    
                    candidate = HybridSolutionCandidate(
                        solution_id=solution_id,
                        problem_context=self._extract_problem_context_from_id(solution_id),
                        solution_text=solution_text,
                        success_rate=self._get_success_rate(solution_id),
                        confidence=self._calculate_enhanced_confidence(solution_id, similarity_score),
                        similarity_score=similarity_score,
                        source_kb=self._get_source_kb(solution_id),
                        last_used=self._get_last_used(solution_id),
                        tags=self._get_tags(solution_id),
                        failed_approaches=[],
                        embedding=self.knowledge_embeddings.get(solution_id)
                    )
                    
                    candidates.append(candidate)
            
            # Rank candidates
            ranked_candidates = self._rank_hybrid_solutions(candidates, problem_description)
            
            return ranked_candidates[:top_k]
            
        except Exception as e:
            print(f"❌ Error in hybrid search: {e}")
            return []
    
    def _extract_text_content(self, data: Dict) -> str:
        """Extract meaningful text content from KB data"""
        text_parts = []
        
        def extract_from_dict(d, prefix=""):
            for key, value in d.items():
                if isinstance(value, str) and len(value) > 20:
                    text_parts.append(f"{prefix}{key}: {value}")
                elif isinstance(value, dict):
                    extract_from_dict(value, f"{prefix}{key}.")
                elif isinstance(value, list):
                    for item in value:
                        if isinstance(item, str) and len(item) > 10:
                            text_parts.append(f"{prefix}{key}: {item}")
        
        extract_from_dict(data)
        return " ".join(text_parts)
    
    def _organize_knowledge_clusters(self, text_to_id: Dict, cluster_labels: np.ndarray):
        """Organize knowledge into clusters"""
        clusters = {}
        
        for text_idx, solution_id in text_to_id.items():
            cluster_id = int(cluster_labels[text_idx])
            
            if cluster_id not in clusters:
                clusters[cluster_id] = []
            
            clusters[cluster_id].append(solution_id)
        
        self.solution_metadata['clusters'] = clusters
        print(f"📊 Organized knowledge into {len(clusters)} clusters")
    
    def _calculate_enhanced_confidence(self, solution_id: str, similarity_score: float) -> float:
        """Calculate enhanced confidence using hybrid processing"""
        # Similar to GPU version but with hybrid processing
        similarity_contribution = similarity_score * 0.4
        success_contribution = self._get_success_rate(solution_id) * 0.3
        recency_contribution = self._get_recency_score(solution_id) * 0.2
        quality_contribution = self._assess_content_quality(solution_id) * 0.1
        
        return min(similarity_contribution + success_contribution + 
                  recency_contribution + quality_contribution, 1.0)
    
    def _rank_hybrid_solutions(self, candidates: List[HybridSolutionCandidate], 
                              problem_description: str) -> List[HybridSolutionCandidate]:
        """Rank solutions using hybrid processing"""
        def scoring_function(candidate: HybridSolutionCandidate) -> float:
            return (candidate.similarity_score * 0.35 + 
                   candidate.confidence * 0.25 + 
                   candidate.success_rate * 0.25 + 
                   self._get_recency_score(candidate.solution_id) * 0.15)
        
        return sorted(candidates, key=scoring_function, reverse=True)
    
    def _get_success_rate(self, solution_id: str) -> float:
        """Get success rate for solution"""
        if solution_id in self.solution_metadata:
            stats = self.solution_metadata[solution_id].get('stats', {})
            total = stats.get('total_uses', 0)
            successes = stats.get('successes', 0)
            return successes / total if total > 0 else 0.5
        return 0.5
    
    def _get_recency_score(self, solution_id: str) -> float:
        """Get recency score for solution"""
        if solution_id in self.solution_metadata:
            last_used = self.solution_metadata[solution_id].get('last_used', '')
            if last_used:
                try:
                    last_date = datetime.datetime.fromisoformat(last_used)
                    days_old = (datetime.datetime.now() - last_date).days
                    return max(0, min(1, 1 - days_old / 365))
                except:
                    pass
        return 0.3
    
    def _assess_content_quality(self, solution_id: str) -> float:
        """Assess content quality of solution"""
        text = self.solution_texts.get(solution_id, "")
        if not text:
            return 0.1
        
        length_score = min(len(text) / 500, 1.0)
        structure_score = 0.5
        
        return (length_score + structure_score) / 2
    
    def _extract_problem_context_from_id(self, solution_id: str) -> str:
        """Extract problem context from solution ID"""
        return f"Context for {solution_id}"
    
    def _get_source_kb(self, solution_id: str) -> str:
        """Get source KB for solution"""
        return solution_id.split('_')[0] if '_' in solution_id else "unknown"
    
    def _get_last_used(self, solution_id: str) -> str:
        """Get last used date for solution"""
        if solution_id in self.solution_metadata:
            return self.solution_metadata[solution_id].get('last_used', '')
        return datetime.datetime.now().isoformat()
    
    def _get_tags(self, solution_id: str) -> List[str]:
        """Get tags for solution"""
        if solution_id in self.solution_metadata:
            return self.solution_metadata[solution_id].get('tags', [])
        return []
    
    def record_solution_feedback(self, solution_id: str, success: bool):
        """Record feedback on solution effectiveness"""
        if solution_id not in self.solution_metadata:
            self.solution_metadata[solution_id] = {
                'stats': {'total_uses': 0, 'successes': 0, 'failures': 0}
            }
        
        stats = self.solution_metadata[solution_id]['stats']
        stats['total_uses'] += 1
        
        if success:
            stats['successes'] += 1
        else:
            stats['failures'] += 1
        
        self.solution_metadata[solution_id]['last_used'] = datetime.datetime.now().isoformat()
        self.save_brain_state()
    
    def get_hybrid_status(self) -> Dict:
        """Get hybrid system status"""
        return {
            'gpu_available': CUPY_AVAILABLE,
            'sklearn_available': SKLEARN_AVAILABLE,
            'hybrid_mode': True,
            'gpu_enabled': self.use_gpu,
            'knowledge_embeddings': len(self.knowledge_embeddings),
            'solution_texts': len(self.solution_texts),
            'brain_type': 'Hybrid GPU/CPU'
        }
    
    def get_system_status(self) -> Dict:
        """Get system status compatible with base class"""
        status = {
            'total_solutions': len(self.solution_metadata),
            'total_patterns': len(self.knowledge_embeddings),
            'avg_success_rate': self._calculate_avg_success_rate(),
            'last_updated': datetime.datetime.now().isoformat(),
            'kb_files_count': len(list(self.kb_root.rglob("*.json"))),
            'hybrid_gpu_status': self.get_hybrid_status()
        }
        
        # Add performance optimization status
        if self.performance_manager:
            status['performance_optimizations'] = self.performance_manager.get_optimization_status()
        
        return status
    
    def _calculate_avg_success_rate(self) -> float:
        """Calculate average success rate across all solutions"""
        if not self.solution_metadata:
            return 0.0
        
        total_rate = 0
        count = 0
        
        for solution_id, metadata in self.solution_metadata.items():
            stats = metadata.get('stats', {})
            total_uses = stats.get('total_uses', 0)
            successes = stats.get('successes', 0)
            if total_uses > 0:
                rate = successes / total_uses
                total_rate += rate
                count += 1
        
        return total_rate / count if count > 0 else 0.0

def main():
    """Test the hybrid GPU KB Brain"""
    brain = HybridGPUKBBrain()
    
    # Test basic functionality
    print("🧪 Testing hybrid KB Brain...")
    status = brain.get_system_status()
    print(f"Status: {status}")
    
    # Test search
    solutions = brain.find_best_solution_hybrid("SSL certificate issues")
    print(f"Found {len(solutions)} solutions")
    
    for i, solution in enumerate(solutions, 1):
        print(f"Solution {i}: {solution.solution_text[:100]}...")

if __name__ == "__main__":
    main()