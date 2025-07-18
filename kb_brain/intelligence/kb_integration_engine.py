"""
KB Integration Engine for multi-level knowledge base search and cross-repository analysis
Provides intelligent knowledge retrieval with context awareness and relationship mapping
"""

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime, timedelta

# Import performance optimizations
try:
    from ..performance.performance_integration import PerformanceManager
    PERFORMANCE_AVAILABLE = True
except ImportError:
    PERFORMANCE_AVAILABLE = False
    logger.warning("Performance optimizations not available for KB Integration Engine")

logger = logging.getLogger(__name__)


class SearchDepth(Enum):
    """Knowledge base search depth levels"""
    SHALLOW = "shallow"     # Single KB, basic search
    DEEP = "deep"          # Multiple KBs, comprehensive search
    CROSS_REPO = "cross_repo"  # Cross-repository analysis
    COMPREHENSIVE = "comprehensive"  # Full institutional memory


class KnowledgeLevel(Enum):
    """Three-tier knowledge base levels"""
    SYSTEM = "system"      # System-wide knowledge
    USER = "user"         # User-specific knowledge  
    PROJECT = "project"   # Project-specific knowledge


@dataclass
class KnowledgeSource:
    """Information about a knowledge base source"""
    path: str
    level: KnowledgeLevel
    domain: str
    last_updated: datetime
    size: int
    repository: Optional[str] = None


@dataclass
class KnowledgeResult:
    """Single knowledge retrieval result"""
    content: Dict[str, Any]
    source: KnowledgeSource
    relevance_score: float
    confidence: float
    problem_status: str  # active, solved, archived
    timestamp: datetime
    tags: List[str]
    related_problems: List[str]


@dataclass
class KnowledgeResponse:
    """Complete knowledge retrieval response"""
    query: str
    results: List[KnowledgeResult]
    search_depth: SearchDepth
    total_sources_searched: int
    processing_time: float
    cross_repo_insights: Dict[str, Any]
    recommendations: List[str]
    confidence: float


class KBIntegrationEngine:
    """Advanced knowledge base integration and search engine"""
    
    def __init__(self, kb_system_path: Optional[str] = None,
                 enable_performance_optimizations: bool = True):
        # Import settings here to avoid circular imports
        from ..config.settings import Settings
        
        self.kb_system_path = Path(kb_system_path) if kb_system_path else Settings.KB_SYSTEM_PATH
        self.knowledge_sources = {}
        self.repository_map = {}
        self.domain_index = {}
        
        # Initialize performance manager
        self.performance_manager = None
        if enable_performance_optimizations and PERFORMANCE_AVAILABLE:
            self.performance_manager = PerformanceManager(auto_optimize=True)
            logger.info("🚀 KB Integration Engine performance optimizations enabled")
        
        self._initialize_engine()
    
    def _initialize_engine(self):
        """Initialize the knowledge base engine"""
        logger.info("Initializing KB Integration Engine")
        
        # Discover all knowledge base sources
        self._discover_knowledge_sources()
        
        # Build repository relationship map
        self._build_repository_map()
        
        # Create domain index
        self._build_domain_index()
        
        logger.info(f"Initialized engine with {len(self.knowledge_sources)} knowledge sources")
    
    def _discover_knowledge_sources(self):
        """Discover all available knowledge base files"""
        
        kb_patterns = [
            "*.json",
            "*_kb.json", 
            "troubleshooting_kb.json",
            "*knowledge*.json"
        ]
        
        for pattern in kb_patterns:
            for kb_file in self.kb_system_path.glob(pattern):
                try:
                    source = self._analyze_kb_source(kb_file)
                    if source:
                        self.knowledge_sources[kb_file.name] = source
                except Exception as e:
                    logger.warning(f"Could not analyze KB source {kb_file}: {e}")
    
    def _analyze_kb_source(self, kb_file: Path) -> Optional[KnowledgeSource]:
        """Analyze a knowledge base file to extract metadata"""
        
        try:
            with open(kb_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Determine knowledge level
            level = self._determine_knowledge_level(kb_file, data)
            
            # Determine domain
            domain = self._determine_domain(kb_file, data)
            
            # Extract repository information
            repository = self._extract_repository_info(kb_file, data)
            
            return KnowledgeSource(
                path=str(kb_file),
                level=level,
                domain=domain,
                last_updated=datetime.fromtimestamp(kb_file.stat().st_mtime),
                size=kb_file.stat().st_size,
                repository=repository
            )
            
        except Exception as e:
            logger.error(f"Error analyzing KB source {kb_file}: {e}")
            return None
    
    def _determine_knowledge_level(self, kb_file: Path, data: Dict) -> KnowledgeLevel:
        """Determine the knowledge base level (system/user/project)"""
        
        filename = kb_file.name.lower()
        
        # System-level indicators
        if any(indicator in filename for indicator in ['system', 'global', 'general']):
            return KnowledgeLevel.SYSTEM
        
        # Project-specific indicators
        if any(indicator in filename for indicator in ['dunes', 'mspc', 'whsa', 'project']):
            return KnowledgeLevel.PROJECT
        
        # Check data structure for level hints
        if isinstance(data, dict):
            if 'system' in data or 'global' in data:
                return KnowledgeLevel.SYSTEM
            elif any(key in data for key in ['projects', 'repositories', 'research']):
                return KnowledgeLevel.PROJECT
        
        # Default to user level
        return KnowledgeLevel.USER
    
    def _determine_domain(self, kb_file: Path, data: Dict) -> str:
        """Determine the knowledge domain"""
        
        filename = kb_file.name.lower()
        
        # Domain indicators in filename
        domain_map = {
            'dunes': 'geospatial',
            'mspc': 'data_science',
            'research': 'research',
            'troubleshooting': 'technical',
            'system': 'system_admin',
            'code': 'programming'
        }
        
        for indicator, domain in domain_map.items():
            if indicator in filename:
                return domain
        
        # Analyze content for domain hints
        content_str = json.dumps(data).lower()
        if any(term in content_str for term in ['landsat', 'satellite', 'boundary', 'gis']):
            return 'geospatial'
        elif any(term in content_str for term in ['ml', 'model', 'analysis', 'data']):
            return 'data_science'
        elif any(term in content_str for term in ['publication', 'research', 'paper']):
            return 'research'
        
        return 'general'
    
    def _extract_repository_info(self, kb_file: Path, data: Dict) -> Optional[str]:
        """Extract repository information from KB data"""
        
        # Check for explicit repository information
        if isinstance(data, dict):
            if 'repository' in data:
                return data['repository']
            elif 'project' in data and isinstance(data['project'], dict):
                return data['project'].get('repository')
        
        # Infer from file path
        path_parts = kb_file.parts
        for part in path_parts:
            if any(indicator in part.lower() for indicator in ['dunes', 'whsa', 'mspc']):
                return part
        
        return None
    
    def _build_repository_map(self):
        """Build cross-repository relationship mapping"""
        
        for source_name, source in self.knowledge_sources.items():
            if source.repository:
                if source.repository not in self.repository_map:
                    self.repository_map[source.repository] = []
                self.repository_map[source.repository].append(source_name)
        
        # Add known repository relationships
        known_relationships = {
            'dunes': ['dunes_core_kb.json', 'troubleshooting_kb.json'],
            'whsa': ['dunes_core_kb.json', 'mspc_kb.json'],
            'mspc': ['mspc_kb.json', 'troubleshooting_kb.json']
        }
        
        for repo, related_kbs in known_relationships.items():
            if repo not in self.repository_map:
                self.repository_map[repo] = []
            for kb in related_kbs:
                if kb in self.knowledge_sources and kb not in self.repository_map[repo]:
                    self.repository_map[repo].append(kb)
    
    def _build_domain_index(self):
        """Build domain-based knowledge index"""
        
        for source_name, source in self.knowledge_sources.items():
            domain = source.domain
            if domain not in self.domain_index:
                self.domain_index[domain] = []
            self.domain_index[domain].append(source_name)
    
    def search_knowledge(self, 
                        query: str, 
                        search_depth: SearchDepth = SearchDepth.DEEP,
                        domain_filter: Optional[str] = None,
                        repository_filter: Optional[str] = None,
                        max_results: int = 10) -> KnowledgeResponse:
        """
        Intelligent knowledge base search with multi-level analysis
        
        Args:
            query: Search query
            search_depth: Depth of search to perform
            domain_filter: Optional domain to focus search
            repository_filter: Optional repository to focus search
            max_results: Maximum number of results to return
        
        Returns:
            KnowledgeResponse with comprehensive results
        """
        
        start_time = datetime.now()
        
        # Determine which sources to search
        sources_to_search = self._select_search_sources(search_depth, domain_filter, repository_filter)
        
        # Perform the search
        results = []
        for source_name in sources_to_search:
            source_results = self._search_single_source(query, source_name)
            results.extend(source_results)
        
        # Rank and filter results
        ranked_results = self._rank_results(results, query)[:max_results]
        
        # Generate cross-repository insights
        cross_repo_insights = self._generate_cross_repo_insights(ranked_results, query)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(ranked_results, search_depth)
        
        # Calculate overall confidence
        confidence = self._calculate_response_confidence(ranked_results, len(sources_to_search))
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return KnowledgeResponse(
            query=query,
            results=ranked_results,
            search_depth=search_depth,
            total_sources_searched=len(sources_to_search),
            processing_time=processing_time,
            cross_repo_insights=cross_repo_insights,
            recommendations=recommendations,
            confidence=confidence
        )
    
    def _select_search_sources(self, 
                              search_depth: SearchDepth, 
                              domain_filter: Optional[str],
                              repository_filter: Optional[str]) -> List[str]:
        """Select which knowledge sources to search based on parameters"""
        
        sources = []
        
        if search_depth == SearchDepth.SHALLOW:
            # Just search the most relevant source
            if domain_filter and domain_filter in self.domain_index:
                sources = self.domain_index[domain_filter][:1]
            else:
                sources = list(self.knowledge_sources.keys())[:1]
                
        elif search_depth == SearchDepth.DEEP:
            # Search multiple relevant sources
            if domain_filter and domain_filter in self.domain_index:
                sources = self.domain_index[domain_filter]
            else:
                sources = list(self.knowledge_sources.keys())
                
        elif search_depth == SearchDepth.CROSS_REPO:
            # Search across repositories
            if repository_filter and repository_filter in self.repository_map:
                sources = self.repository_map[repository_filter]
                # Add related repositories
                for repo_sources in self.repository_map.values():
                    if any(source in sources for source in repo_sources):
                        sources.extend(repo_sources)
            else:
                sources = list(self.knowledge_sources.keys())
                
        elif search_depth == SearchDepth.COMPREHENSIVE:
            # Search everything
            sources = list(self.knowledge_sources.keys())
        
        # Remove duplicates while preserving order
        return list(dict.fromkeys(sources))
    
    def _search_single_source(self, query: str, source_name: str) -> List[KnowledgeResult]:
        """Search a single knowledge base source"""
        
        if source_name not in self.knowledge_sources:
            return []
        
        source = self.knowledge_sources[source_name]
        results = []
        
        try:
            with open(source.path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Search through the data structure
            matches = self._find_matches_in_data(query, data, source)
            results.extend(matches)
            
        except Exception as e:
            logger.error(f"Error searching source {source_name}: {e}")
        
        return results
    
    def _find_matches_in_data(self, query: str, data: Any, source: KnowledgeSource, path: str = "") -> List[KnowledgeResult]:
        """Recursively find matches in knowledge base data"""
        
        results = []
        query_lower = query.lower()
        
        if isinstance(data, dict):
            for key, value in data.items():
                current_path = f"{path}.{key}" if path else key
                
                # Check if this looks like a problem/solution entry
                if self._is_problem_entry(key, value):
                    relevance = self._calculate_relevance(query_lower, value)
                    if relevance > 0.3:  # Threshold for relevance
                        result = self._create_knowledge_result(value, source, relevance, current_path)
                        if result:
                            results.append(result)
                
                # Recursively search nested structures
                if isinstance(value, (dict, list)):
                    nested_results = self._find_matches_in_data(query, value, source, current_path)
                    results.extend(nested_results)
                    
        elif isinstance(data, list):
            for i, item in enumerate(data):
                current_path = f"{path}[{i}]" if path else f"[{i}]"
                nested_results = self._find_matches_in_data(query, item, source, current_path)
                results.extend(nested_results)
        
        return results
    
    def _is_problem_entry(self, key: str, value: Any) -> bool:
        """Check if a data entry represents a problem/solution"""
        
        if not isinstance(value, dict):
            return False
        
        # Look for problem/solution indicators
        problem_indicators = ['problem', 'issue', 'error', 'description', 'solution', 'approach']
        
        return any(indicator in str(value).lower() for indicator in problem_indicators) or \
               any(indicator in key.lower() for indicator in problem_indicators)
    
    def _calculate_relevance(self, query: str, content: Any) -> float:
        """Calculate relevance score for content against query with performance optimization"""
        
        content_str = json.dumps(content).lower() if not isinstance(content, str) else content.lower()
        query_words = set(query.split())
        
        # Try performance-optimized similarity computation for longer texts
        if self.performance_manager and len(content_str) > 100:
            try:
                # Use TF-IDF based similarity if available
                import numpy as np
                from sklearn.feature_extraction.text import TfidfVectorizer
                
                # Create simple TF-IDF vectors
                vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
                texts = [query.lower(), content_str]
                
                try:
                    tfidf_matrix = vectorizer.fit_transform(texts)
                    
                    # Use performance-optimized similarity calculation
                    result = self.performance_manager.optimize_similarity_computation(
                        tfidf_matrix.toarray(), 
                        tfidf_matrix[0].toarray(),
                        top_k=1,
                        metric="cosine"
                    )
                    
                    similarities = result["similarities"]
                    if similarities and "scores" in similarities:
                        # Get similarity score (first result is self-similarity, second is content)
                        similarity_score = similarities["scores"][0] if len(similarities["scores"]) > 0 else 0.0
                        logger.debug(f"🚀 Used {result['method']} for relevance calculation: {similarity_score:.3f}")
                        return float(similarity_score)
                        
                except Exception as e:
                    logger.debug(f"Performance optimization failed, using fallback: {e}")
                    
            except Exception as e:
                logger.debug(f"Could not use performance optimization: {e}")
        
        # Fallback to simple text matching
        exact_matches = sum(1 for word in query_words if word in content_str)
        
        # Partial matches
        partial_matches = sum(1 for word in query_words 
                            if any(word in content_word for content_word in content_str.split()))
        
        # Calculate relevance score
        total_words = len(query_words)
        if total_words == 0:
            return 0.0
        
        relevance = (exact_matches * 1.0 + partial_matches * 0.5) / total_words
        return min(1.0, relevance)
    
    def _create_knowledge_result(self, content: Any, source: KnowledgeSource, relevance: float, path: str) -> Optional[KnowledgeResult]:
        """Create a KnowledgeResult from matched content"""
        
        try:
            # Extract key information
            if isinstance(content, dict):
                problem_status = content.get('status', 'unknown')
                timestamp = self._extract_timestamp(content)
                tags = self._extract_tags(content)
                related_problems = content.get('related_problems', [])
            else:
                problem_status = 'unknown'
                timestamp = source.last_updated
                tags = []
                related_problems = []
            
            # Calculate confidence based on source quality and recency
            confidence = self._calculate_result_confidence(content, source, timestamp)
            
            return KnowledgeResult(
                content=content if isinstance(content, dict) else {'data': content},
                source=source,
                relevance_score=relevance,
                confidence=confidence,
                problem_status=problem_status,
                timestamp=timestamp,
                tags=tags,
                related_problems=related_problems
            )
            
        except Exception as e:
            logger.error(f"Error creating knowledge result: {e}")
            return None
    
    def _extract_timestamp(self, content: Dict) -> datetime:
        """Extract timestamp from content"""
        
        timestamp_fields = ['timestamp', 'last_updated', 'created', 'date']
        
        for field in timestamp_fields:
            if field in content:
                try:
                    if isinstance(content[field], str):
                        return datetime.fromisoformat(content[field].replace('Z', '+00:00'))
                    elif isinstance(content[field], (int, float)):
                        return datetime.fromtimestamp(content[field])
                except Exception:
                    continue
        
        return datetime.now()
    
    def _extract_tags(self, content: Dict) -> List[str]:
        """Extract tags from content"""
        
        tags = []
        
        # Direct tags field
        if 'tags' in content and isinstance(content['tags'], list):
            tags.extend(content['tags'])
        
        # Extract from other fields
        tag_sources = ['category', 'type', 'domain', 'keywords']
        for source in tag_sources:
            if source in content:
                value = content[source]
                if isinstance(value, str):
                    tags.append(value)
                elif isinstance(value, list):
                    tags.extend(value)
        
        return list(set(tags))  # Remove duplicates
    
    def _calculate_result_confidence(self, content: Any, source: KnowledgeSource, timestamp: datetime) -> float:
        """Calculate confidence score for a result"""
        
        base_confidence = 0.7
        
        # Source quality factor
        if source.level == KnowledgeLevel.SYSTEM:
            source_factor = 0.2
        elif source.level == KnowledgeLevel.PROJECT:
            source_factor = 0.15
        else:
            source_factor = 0.1
        
        # Recency factor
        age_days = (datetime.now() - timestamp).days
        if age_days < 7:
            recency_factor = 0.15
        elif age_days < 30:
            recency_factor = 0.1
        elif age_days < 90:
            recency_factor = 0.05
        else:
            recency_factor = 0.0
        
        # Content completeness factor
        if isinstance(content, dict):
            completeness = len(content) / 10.0  # Assume 10 fields is "complete"
            completeness_factor = min(0.1, completeness)
        else:
            completeness_factor = 0.05
        
        return min(1.0, base_confidence + source_factor + recency_factor + completeness_factor)
    
    def _rank_results(self, results: List[KnowledgeResult], query: str) -> List[KnowledgeResult]:
        """Rank results by relevance, confidence, and recency"""
        
        def ranking_score(result: KnowledgeResult) -> float:
            # Weighted combination of factors
            relevance_weight = 0.4
            confidence_weight = 0.3
            recency_weight = 0.2
            status_weight = 0.1
            
            # Recency score (newer is better)
            age_days = (datetime.now() - result.timestamp).days
            recency_score = max(0, 1.0 - (age_days / 365.0))  # 1.0 for today, 0.0 for 1 year old
            
            # Status score (solved > active > archived)
            status_scores = {'solved': 1.0, 'active': 0.8, 'archived': 0.3, 'unknown': 0.5}
            status_score = status_scores.get(result.problem_status, 0.5)
            
            return (result.relevance_score * relevance_weight +
                   result.confidence * confidence_weight +
                   recency_score * recency_weight +
                   status_score * status_weight)
        
        return sorted(results, key=ranking_score, reverse=True)
    
    def _generate_cross_repo_insights(self, results: List[KnowledgeResult], query: str) -> Dict[str, Any]:
        """Generate insights from cross-repository analysis"""
        
        insights = {
            'repositories_found': [],
            'common_patterns': [],
            'solution_evolution': [],
            'related_projects': []
        }
        
        # Analyze repositories represented
        repo_sources = {}
        for result in results:
            repo = result.source.repository or 'unknown'
            if repo not in repo_sources:
                repo_sources[repo] = []
            repo_sources[repo].append(result)
        
        insights['repositories_found'] = list(repo_sources.keys())
        
        # Find common patterns across repositories
        all_tags = []
        for result in results:
            all_tags.extend(result.tags)
        
        # Count tag frequency
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Common patterns are tags that appear in multiple repos
        if len(repo_sources) > 1:
            common_tags = [tag for tag, count in tag_counts.items() if count >= 2]
            insights['common_patterns'] = common_tags[:5]  # Top 5
        
        # Related projects (based on shared tags)
        query_words = set(query.lower().split())
        related_projects = set()
        for result in results:
            if result.source.repository and result.source.repository != 'unknown':
                # Check if result is related to query
                content_words = set(json.dumps(result.content).lower().split())
                if query_words.intersection(content_words):
                    related_projects.add(result.source.repository)
        
        insights['related_projects'] = list(related_projects)
        
        return insights
    
    def _generate_recommendations(self, results: List[KnowledgeResult], search_depth: SearchDepth) -> List[str]:
        """Generate actionable recommendations based on results"""
        
        recommendations = []
        
        if not results:
            recommendations.append("No relevant knowledge found. Consider expanding search or documenting new solution.")
            return recommendations
        
        # Analyze result patterns
        active_problems = [r for r in results if r.problem_status == 'active']
        solved_problems = [r for r in results if r.problem_status == 'solved']
        
        if active_problems:
            recommendations.append(f"Found {len(active_problems)} active problems related to your query.")
            
        if solved_problems:
            recommendations.append(f"Found {len(solved_problems)} solved problems that may provide solutions.")
            
        # Check for recent vs old solutions
        recent_results = [r for r in results if (datetime.now() - r.timestamp).days < 30]
        old_results = [r for r in results if (datetime.now() - r.timestamp).days > 180]
        
        if recent_results:
            recommendations.append("Recent solutions available - prioritize these for current relevance.")
            
        if old_results and not recent_results:
            recommendations.append("Only older solutions found - may need updating for current context.")
        
        # Cross-repository recommendations
        repositories = set(r.source.repository for r in results if r.source.repository)
        if len(repositories) > 1:
            recommendations.append(f"Solution patterns found across {len(repositories)} repositories - consider consolidating approaches.")
        
        # Search depth recommendations
        if search_depth == SearchDepth.SHALLOW and len(results) < 3:
            recommendations.append("Consider deeper search for more comprehensive results.")
        
        return recommendations
    
    def _calculate_response_confidence(self, results: List[KnowledgeResult], sources_searched: int) -> float:
        """Calculate overall confidence in the response"""
        
        if not results:
            return 0.0
        
        # Average result confidence
        avg_confidence = sum(r.confidence for r in results) / len(results)
        
        # Coverage factor (more sources = higher confidence)
        coverage_factor = min(1.0, sources_searched / 5.0)  # Cap at 5 sources
        
        # Result quantity factor
        quantity_factor = min(1.0, len(results) / 10.0)  # Cap at 10 results
        
        return (avg_confidence * 0.6 + coverage_factor * 0.2 + quantity_factor * 0.2)
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'total_sources': len(self.knowledge_sources),
            'sources_by_level': {
                level.value: len([s for s in self.knowledge_sources.values() if s.level == level])
                for level in KnowledgeLevel
            },
            'sources_by_domain': {
                domain: len(sources) for domain, sources in self.domain_index.items()
            },
            'repositories': list(self.repository_map.keys()),
            'last_updated': max(s.last_updated for s in self.knowledge_sources.values()) if self.knowledge_sources else None
        }


def test_kb_integration():
    """Test the KB integration engine"""
    
    engine = KBIntegrationEngine()
    
    # Test queries
    test_queries = [
        "SSL certificate issues",
        "Dunes boundary analysis status", 
        "GPU acceleration problems",
        "Publication timeline research themes"
    ]
    
    for query in test_queries:
        print(f"\n=== Query: {query} ===")
        
        # Test different search depths
        for depth in [SearchDepth.SHALLOW, SearchDepth.DEEP, SearchDepth.CROSS_REPO]:
            response = engine.search_knowledge(query, search_depth=depth, max_results=3)
            
            print(f"\n{depth.value.upper()} SEARCH:")
            print(f"Sources searched: {response.total_sources_searched}")
            print(f"Results found: {len(response.results)}")
            print(f"Confidence: {response.confidence:.2f}")
            print(f"Processing time: {response.processing_time:.3f}s")
            
            if response.results:
                top_result = response.results[0]
                print(f"Top result: {top_result.relevance_score:.2f} relevance, {top_result.confidence:.2f} confidence")
            
            if response.recommendations:
                print(f"Recommendations: {response.recommendations[0]}")
    
    # System status
    print(f"\n=== System Status ===")
    status = engine.get_system_status()
    for key, value in status.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    test_kb_integration()