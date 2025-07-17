"""
Knowledge Ingestion System for KB Brain
Captures and integrates solutions, patterns, and insights from AI processing sessions
"""

import json
import hashlib
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class IngestionSource(Enum):
    """Sources of knowledge for ingestion"""
    AI_PROCESSING = "ai_processing"
    USER_INTERACTION = "user_interaction"
    PROBLEM_RESOLUTION = "problem_resolution"
    RESEARCH_FINDINGS = "research_findings"
    CROSS_REPO_ANALYSIS = "cross_repo_analysis"
    MANUAL_ENTRY = "manual_entry"


class KnowledgeType(Enum):
    """Types of knowledge being ingested"""
    SOLUTION = "solution"
    PATTERN = "pattern"
    INSIGHT = "insight"
    FAILURE = "failure"
    METHODOLOGY = "methodology"
    VALIDATION = "validation"


class ConfidenceLevel(Enum):
    """Confidence levels for ingested knowledge"""
    LOW = 0.3
    MEDIUM = 0.6
    HIGH = 0.8
    VERIFIED = 0.95


@dataclass
class KnowledgeEntry:
    """Single knowledge entry for ingestion"""
    id: str
    type: KnowledgeType
    source: IngestionSource
    content: Dict[str, Any]
    tags: List[str]
    context: Dict[str, Any]
    confidence: float
    relevance_score: float
    created_at: datetime
    related_entries: List[str] = None
    validation_status: str = "pending"
    
    def __post_init__(self):
        if self.related_entries is None:
            self.related_entries = []


@dataclass
class IngestionBatch:
    """Batch of knowledge entries for processing"""
    batch_id: str
    entries: List[KnowledgeEntry]
    source_session: Optional[str] = None
    processed_at: Optional[datetime] = None
    processing_notes: List[str] = None
    
    def __post_init__(self):
        if self.processing_notes is None:
            self.processing_notes = []


class KnowledgeIngestionSystem:
    """System for capturing and integrating knowledge from various sources"""
    
    def __init__(self, 
                 kb_system_path: str = "/mnt/c/Users/misley/Documents/Projects/kb_system",
                 ingestion_buffer_size: int = 100):
        """
        Initialize knowledge ingestion system
        
        Args:
            kb_system_path: Path to knowledge base system
            ingestion_buffer_size: Size of ingestion buffer before auto-processing
        """
        self.kb_system_path = Path(kb_system_path)
        self.ingestion_buffer_size = ingestion_buffer_size
        
        # Ingestion tracking
        self.pending_entries: List[KnowledgeEntry] = []
        self.ingestion_history: List[IngestionBatch] = []
        
        # Pattern recognition
        self.known_patterns: Dict[str, Any] = {}
        self.solution_templates: Dict[str, Any] = {}
        
        # Quality thresholds
        self.min_confidence = 0.4
        self.min_relevance = 0.3
        
        self._initialize_system()
        logger.info("Knowledge Ingestion System initialized")
    
    def _initialize_system(self):
        """Initialize the ingestion system"""
        
        # Load existing patterns and templates
        self._load_existing_patterns()
        
        # Create ingestion directories
        ingestion_dir = self.kb_system_path / "ingestion"
        ingestion_dir.mkdir(exist_ok=True)
        
        # Subdirectories for different knowledge types
        for knowledge_type in KnowledgeType:
            (ingestion_dir / knowledge_type.value).mkdir(exist_ok=True)
    
    def _load_existing_patterns(self):
        """Load existing knowledge patterns from KB system"""
        
        try:
            # Load troubleshooting KB for patterns
            troubleshooting_kb = self.kb_system_path / "troubleshooting_kb.json"
            if troubleshooting_kb.exists():
                with open(troubleshooting_kb, 'r') as f:
                    data = json.load(f)
                
                # Extract patterns from existing solutions
                self._extract_patterns_from_kb(data)
                
        except Exception as e:
            logger.error(f"Error loading existing patterns: {e}")
    
    def _extract_patterns_from_kb(self, kb_data: Dict[str, Any]):
        """Extract patterns from existing KB data"""
        
        if isinstance(kb_data, dict):
            for key, value in kb_data.items():
                if key == "solved_problems" and isinstance(value, list):
                    for problem in value:
                        if isinstance(problem, dict) and "solution" in problem:
                            pattern = self._identify_solution_pattern(problem)
                            if pattern:
                                pattern_id = self._generate_pattern_id(pattern)
                                self.known_patterns[pattern_id] = pattern
    
    def ingest_ai_processing_result(self, 
                                  session_name: str,
                                  ai_output: Dict[str, Any],
                                  processing_context: Dict[str, Any]) -> str:
        """
        Ingest results from AI processing session
        
        Args:
            session_name: Name of the processing session
            ai_output: Output from AI processing
            processing_context: Context of the processing
        
        Returns:
            Batch ID for the ingested knowledge
        """
        
        batch_id = f"ai_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{session_name}"
        entries = []
        
        # Extract different types of knowledge from AI output
        
        # 1. Main solution/response
        if "response" in ai_output:
            solution_entry = self._create_solution_entry(
                ai_output["response"],
                processing_context,
                ai_output.get("confidence", 0.7)
            )
            entries.append(solution_entry)
        
        # 2. Insights and patterns
        if "insights" in ai_output:
            insight_entries = self._extract_insights(
                ai_output["insights"],
                processing_context
            )
            entries.extend(insight_entries)
        
        # 3. Processing stages (methodology)
        if "processing_stages" in ai_output:
            methodology_entry = self._create_methodology_entry(
                ai_output["processing_stages"],
                processing_context
            )
            entries.append(methodology_entry)
        
        # 4. Failures or errors
        if "error" in ai_output:
            failure_entry = self._create_failure_entry(
                ai_output["error"],
                processing_context
            )
            entries.append(failure_entry)
        
        # Create ingestion batch
        batch = IngestionBatch(
            batch_id=batch_id,
            entries=entries,
            source_session=session_name
        )
        
        # Add to pending for processing
        self.pending_entries.extend(entries)
        
        # Auto-process if buffer is full
        if len(self.pending_entries) >= self.ingestion_buffer_size:
            self._process_pending_entries()
        
        logger.info(f"Ingested {len(entries)} knowledge entries from AI session {session_name}")
        
        return batch_id
    
    def ingest_user_interaction(self,
                              prompt: str,
                              response: str,
                              context: Dict[str, Any],
                              user_feedback: Optional[Dict[str, Any]] = None) -> str:
        """
        Ingest knowledge from user interaction
        
        Args:
            prompt: User's original prompt
            response: System response
            context: Interaction context
            user_feedback: Optional user feedback on response quality
        
        Returns:
            Entry ID for the ingested knowledge
        """
        
        # Determine knowledge type based on interaction
        if user_feedback and user_feedback.get("helpful", False):
            knowledge_type = KnowledgeType.SOLUTION
            confidence = 0.8
        elif user_feedback and not user_feedback.get("helpful", True):
            knowledge_type = KnowledgeType.FAILURE
            confidence = 0.6
        else:
            knowledge_type = KnowledgeType.PATTERN
            confidence = 0.7
        
        entry = KnowledgeEntry(
            id=self._generate_entry_id(),
            type=knowledge_type,
            source=IngestionSource.USER_INTERACTION,
            content={
                "prompt": prompt,
                "response": response,
                "user_feedback": user_feedback
            },
            tags=self._extract_tags_from_text(prompt + " " + response),
            context=context,
            confidence=confidence,
            relevance_score=self._calculate_relevance(prompt, response),
            created_at=datetime.now()
        )
        
        self.pending_entries.append(entry)
        
        logger.info(f"Ingested user interaction: {entry.id}")
        
        return entry.id
    
    def ingest_problem_resolution(self,
                                problem_description: str,
                                solution: str,
                                validation_results: Dict[str, Any],
                                context: Dict[str, Any]) -> str:
        """
        Ingest knowledge from problem resolution
        
        Args:
            problem_description: Description of the problem
            solution: The solution that worked
            validation_results: Results of solution validation
            context: Problem context
        
        Returns:
            Entry ID for the ingested knowledge
        """
        
        # Determine confidence based on validation
        confidence = self._assess_solution_confidence(validation_results)
        
        solution_entry = KnowledgeEntry(
            id=self._generate_entry_id(),
            type=KnowledgeType.SOLUTION,
            source=IngestionSource.PROBLEM_RESOLUTION,
            content={
                "problem": problem_description,
                "solution": solution,
                "validation": validation_results
            },
            tags=self._extract_tags_from_text(problem_description + " " + solution),
            context=context,
            confidence=confidence,
            relevance_score=0.9,  # High relevance for validated solutions
            created_at=datetime.now(),
            validation_status="validated"
        )
        
        self.pending_entries.append(solution_entry)
        
        # Also create validation entry
        validation_entry = KnowledgeEntry(
            id=self._generate_entry_id(),
            type=KnowledgeType.VALIDATION,
            source=IngestionSource.PROBLEM_RESOLUTION,
            content={
                "solution_id": solution_entry.id,
                "validation_method": validation_results.get("method", "unknown"),
                "success_rate": validation_results.get("success_rate", 0.0),
                "test_results": validation_results.get("tests", [])
            },
            tags=["validation"] + solution_entry.tags[:3],
            context=context,
            confidence=confidence,
            relevance_score=0.8,
            created_at=datetime.now(),
            related_entries=[solution_entry.id]
        )
        
        self.pending_entries.append(validation_entry)
        
        logger.info(f"Ingested problem resolution: {solution_entry.id}")
        
        return solution_entry.id
    
    def _create_solution_entry(self, 
                             response: str, 
                             context: Dict[str, Any], 
                             confidence: float) -> KnowledgeEntry:
        """Create a solution knowledge entry"""
        
        return KnowledgeEntry(
            id=self._generate_entry_id(),
            type=KnowledgeType.SOLUTION,
            source=IngestionSource.AI_PROCESSING,
            content={
                "solution": response,
                "approach": context.get("processing_mode", "unknown")
            },
            tags=self._extract_tags_from_text(response),
            context=context,
            confidence=confidence,
            relevance_score=self._calculate_relevance_from_context(context),
            created_at=datetime.now()
        )
    
    def _extract_insights(self, 
                         insights: Dict[str, Any], 
                         context: Dict[str, Any]) -> List[KnowledgeEntry]:
        """Extract insight entries from AI processing insights"""
        
        entries = []
        
        for insight_key, insight_value in insights.items():
            if isinstance(insight_value, (str, dict, list)) and insight_value:
                entry = KnowledgeEntry(
                    id=self._generate_entry_id(),
                    type=KnowledgeType.INSIGHT,
                    source=IngestionSource.AI_PROCESSING,
                    content={
                        "insight_type": insight_key,
                        "insight": insight_value
                    },
                    tags=[insight_key] + self._extract_tags_from_content(insight_value),
                    context=context,
                    confidence=0.7,
                    relevance_score=0.6,
                    created_at=datetime.now()
                )
                entries.append(entry)
        
        return entries
    
    def _create_methodology_entry(self, 
                                stages: List[Dict[str, Any]], 
                                context: Dict[str, Any]) -> KnowledgeEntry:
        """Create methodology entry from processing stages"""
        
        return KnowledgeEntry(
            id=self._generate_entry_id(),
            type=KnowledgeType.METHODOLOGY,
            source=IngestionSource.AI_PROCESSING,
            content={
                "processing_stages": stages,
                "methodology": context.get("processing_mode", "unknown")
            },
            tags=["methodology", "processing"] + [stage.get("stage", "") for stage in stages[:3]],
            context=context,
            confidence=0.8,
            relevance_score=0.7,
            created_at=datetime.now()
        )
    
    def _create_failure_entry(self, 
                            error: str, 
                            context: Dict[str, Any]) -> KnowledgeEntry:
        """Create failure knowledge entry"""
        
        return KnowledgeEntry(
            id=self._generate_entry_id(),
            type=KnowledgeType.FAILURE,
            source=IngestionSource.AI_PROCESSING,
            content={
                "error": error,
                "context": context.get("processing_mode", "unknown")
            },
            tags=["error", "failure"] + self._extract_tags_from_text(error),
            context=context,
            confidence=0.9,  # High confidence in failures
            relevance_score=0.8,  # High relevance for learning
            created_at=datetime.now()
        )
    
    def _process_pending_entries(self):
        """Process pending knowledge entries"""
        
        if not self.pending_entries:
            return
        
        # Filter entries by quality thresholds
        quality_entries = [
            entry for entry in self.pending_entries
            if entry.confidence >= self.min_confidence 
            and entry.relevance_score >= self.min_relevance
        ]
        
        # Group entries by type and domain
        grouped_entries = self._group_entries(quality_entries)
        
        # Process each group
        for group_key, entries in grouped_entries.items():
            self._process_entry_group(group_key, entries)
        
        # Create processed batch
        batch = IngestionBatch(
            batch_id=f"processed_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            entries=quality_entries,
            processed_at=datetime.now(),
            processing_notes=[f"Processed {len(quality_entries)} entries"]
        )
        
        self.ingestion_history.append(batch)
        
        # Clear pending entries
        self.pending_entries = []
        
        logger.info(f"Processed {len(quality_entries)} knowledge entries")
    
    def _group_entries(self, entries: List[KnowledgeEntry]) -> Dict[str, List[KnowledgeEntry]]:
        """Group entries by type and domain for processing"""
        
        groups = {}
        
        for entry in entries:
            # Create group key based on type and primary tags
            primary_tags = entry.tags[:2] if entry.tags else ["general"]
            group_key = f"{entry.type.value}_{primary_tags[0]}"
            
            if group_key not in groups:
                groups[group_key] = []
            
            groups[group_key].append(entry)
        
        return groups
    
    def _process_entry_group(self, group_key: str, entries: List[KnowledgeEntry]):
        """Process a group of similar entries"""
        
        # Identify patterns within the group
        patterns = self._identify_group_patterns(entries)
        
        # Merge similar entries
        merged_entries = self._merge_similar_entries(entries)
        
        # Update knowledge base
        self._update_knowledge_base(group_key, merged_entries, patterns)
    
    def _identify_group_patterns(self, entries: List[KnowledgeEntry]) -> List[Dict[str, Any]]:
        """Identify patterns within a group of entries"""
        
        patterns = []
        
        # Look for common solution patterns
        if all(entry.type == KnowledgeType.SOLUTION for entry in entries):
            solution_pattern = self._find_solution_pattern(entries)
            if solution_pattern:
                patterns.append(solution_pattern)
        
        # Look for common failure patterns
        if any(entry.type == KnowledgeType.FAILURE for entry in entries):
            failure_pattern = self._find_failure_pattern(entries)
            if failure_pattern:
                patterns.append(failure_pattern)
        
        return patterns
    
    def _find_solution_pattern(self, entries: List[KnowledgeEntry]) -> Optional[Dict[str, Any]]:
        """Find common solution pattern in entries"""
        
        # Extract common keywords and approaches
        all_tags = []
        all_approaches = []
        
        for entry in entries:
            all_tags.extend(entry.tags)
            if "approach" in entry.content:
                all_approaches.append(entry.content["approach"])
        
        # Find most common elements
        tag_counts = {}
        for tag in all_tags:
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        common_tags = [tag for tag, count in tag_counts.items() if count >= len(entries) * 0.5]
        
        if common_tags:
            return {
                "type": "solution_pattern",
                "common_tags": common_tags,
                "frequency": len(entries),
                "confidence": sum(e.confidence for e in entries) / len(entries)
            }
        
        return None
    
    def _find_failure_pattern(self, entries: List[KnowledgeEntry]) -> Optional[Dict[str, Any]]:
        """Find common failure pattern in entries"""
        
        failure_entries = [e for e in entries if e.type == KnowledgeType.FAILURE]
        
        if len(failure_entries) < 2:
            return None
        
        # Extract common error keywords
        error_keywords = []
        for entry in failure_entries:
            if "error" in entry.content:
                words = re.findall(r'\b\w+\b', entry.content["error"].lower())
                error_keywords.extend(words)
        
        # Find most common error patterns
        keyword_counts = {}
        for keyword in error_keywords:
            keyword_counts[keyword] = keyword_counts.get(keyword, 0) + 1
        
        common_keywords = [kw for kw, count in keyword_counts.items() if count >= 2]
        
        if common_keywords:
            return {
                "type": "failure_pattern",
                "common_errors": common_keywords[:5],
                "frequency": len(failure_entries),
                "pattern_id": self._generate_pattern_id({"errors": common_keywords})
            }
        
        return None
    
    def _merge_similar_entries(self, entries: List[KnowledgeEntry]) -> List[KnowledgeEntry]:
        """Merge similar entries to reduce redundancy"""
        
        if len(entries) <= 1:
            return entries
        
        merged = []
        processed_ids = set()
        
        for entry in entries:
            if entry.id in processed_ids:
                continue
            
            # Find similar entries
            similar_entries = [entry]
            for other_entry in entries:
                if (other_entry.id != entry.id and 
                    other_entry.id not in processed_ids and
                    self._are_entries_similar(entry, other_entry)):
                    similar_entries.append(other_entry)
                    processed_ids.add(other_entry.id)
            
            # Merge if multiple similar entries found
            if len(similar_entries) > 1:
                merged_entry = self._merge_entry_group(similar_entries)
                merged.append(merged_entry)
            else:
                merged.append(entry)
            
            processed_ids.add(entry.id)
        
        return merged
    
    def _are_entries_similar(self, entry1: KnowledgeEntry, entry2: KnowledgeEntry) -> bool:
        """Check if two entries are similar enough to merge"""
        
        # Must be same type
        if entry1.type != entry2.type:
            return False
        
        # Check tag overlap
        tags1 = set(entry1.tags)
        tags2 = set(entry2.tags)
        
        if tags1 and tags2:
            overlap = len(tags1.intersection(tags2)) / len(tags1.union(tags2))
            return overlap >= 0.5
        
        return False
    
    def _merge_entry_group(self, entries: List[KnowledgeEntry]) -> KnowledgeEntry:
        """Merge a group of similar entries"""
        
        # Use the highest confidence entry as base
        base_entry = max(entries, key=lambda e: e.confidence)
        
        # Combine content
        merged_content = base_entry.content.copy()
        merged_content["merged_from"] = [e.id for e in entries if e.id != base_entry.id]
        merged_content["merge_count"] = len(entries)
        
        # Combine tags (unique)
        all_tags = []
        for entry in entries:
            all_tags.extend(entry.tags)
        merged_tags = list(set(all_tags))
        
        # Average confidence and relevance
        avg_confidence = sum(e.confidence for e in entries) / len(entries)
        avg_relevance = sum(e.relevance_score for e in entries) / len(entries)
        
        # Create merged entry
        merged_entry = KnowledgeEntry(
            id=self._generate_entry_id(),
            type=base_entry.type,
            source=base_entry.source,
            content=merged_content,
            tags=merged_tags,
            context=base_entry.context,
            confidence=avg_confidence,
            relevance_score=avg_relevance,
            created_at=datetime.now(),
            related_entries=[e.id for e in entries if e.id != base_entry.id]
        )
        
        return merged_entry
    
    def _update_knowledge_base(self, 
                             group_key: str, 
                             entries: List[KnowledgeEntry], 
                             patterns: List[Dict[str, Any]]):
        """Update the knowledge base with processed entries"""
        
        # Determine target KB file based on group
        if "dunes" in group_key or "whsa" in group_key:
            kb_file = self.kb_system_path / "dunes_core_kb.json"
        elif "system" in group_key or "general" in group_key:
            kb_file = self.kb_system_path / "troubleshooting_kb.json"
        else:
            kb_file = self.kb_system_path / "troubleshooting_kb.json"  # Default
        
        try:
            # Load existing KB
            if kb_file.exists():
                with open(kb_file, 'r') as f:
                    kb_data = json.load(f)
            else:
                kb_data = {}
            
            # Add entries to appropriate sections
            for entry in entries:
                self._add_entry_to_kb(kb_data, entry)
            
            # Add patterns
            if patterns:
                if "patterns" not in kb_data:
                    kb_data["patterns"] = []
                kb_data["patterns"].extend(patterns)
            
            # Update metadata
            if "metadata" not in kb_data:
                kb_data["metadata"] = {}
            
            kb_data["metadata"]["last_ingestion"] = datetime.now().isoformat()
            kb_data["metadata"]["total_entries"] = kb_data["metadata"].get("total_entries", 0) + len(entries)
            
            # Save updated KB
            with open(kb_file, 'w') as f:
                json.dump(kb_data, f, indent=2)
            
            logger.info(f"Updated KB {kb_file.name} with {len(entries)} entries")
            
        except Exception as e:
            logger.error(f"Error updating knowledge base {kb_file}: {e}")
    
    def _add_entry_to_kb(self, kb_data: Dict[str, Any], entry: KnowledgeEntry):
        """Add a single entry to KB data structure"""
        
        # Determine section based on entry type
        if entry.type == KnowledgeType.SOLUTION:
            section = "solved_problems"
        elif entry.type == KnowledgeType.FAILURE:
            section = "failed_approaches"
        elif entry.type == KnowledgeType.PATTERN:
            section = "patterns"
        else:
            section = "insights"
        
        if section not in kb_data:
            kb_data[section] = []
        
        # Convert entry to KB format
        kb_entry = {
            "id": entry.id,
            "type": entry.type.value,
            "content": entry.content,
            "tags": entry.tags,
            "confidence": entry.confidence,
            "relevance": entry.relevance_score,
            "timestamp": entry.created_at.isoformat(),
            "validation_status": entry.validation_status
        }
        
        if entry.related_entries:
            kb_entry["related_entries"] = entry.related_entries
        
        kb_data[section].append(kb_entry)
    
    def _generate_entry_id(self) -> str:
        """Generate unique entry ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_part = hashlib.md5(f"{timestamp}_{len(self.pending_entries)}".encode()).hexdigest()[:8]
        return f"entry_{timestamp}_{random_part}"
    
    def _generate_pattern_id(self, pattern: Dict[str, Any]) -> str:
        """Generate unique pattern ID"""
        pattern_str = json.dumps(pattern, sort_keys=True)
        return hashlib.md5(pattern_str.encode()).hexdigest()[:16]
    
    def _identify_solution_pattern(self, problem: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Identify solution pattern from problem data"""
        
        if "solution" not in problem:
            return None
        
        solution = problem["solution"]
        tags = problem.get("tags", [])
        
        # Extract pattern elements
        pattern = {
            "solution_type": self._classify_solution_type(solution),
            "domain": tags[0] if tags else "general",
            "approach": self._extract_approach(solution)
        }
        
        return pattern if pattern["solution_type"] else None
    
    def _classify_solution_type(self, solution: str) -> Optional[str]:
        """Classify the type of solution"""
        
        solution_lower = solution.lower()
        
        if any(word in solution_lower for word in ["install", "pip", "apt", "brew"]):
            return "installation"
        elif any(word in solution_lower for word in ["config", "setting", "environment"]):
            return "configuration"
        elif any(word in solution_lower for word in ["debug", "fix", "error"]):
            return "debugging"
        elif any(word in solution_lower for word in ["implement", "code", "function"]):
            return "implementation"
        
        return None
    
    def _extract_approach(self, solution: str) -> str:
        """Extract the general approach from solution"""
        
        # Simple keyword-based approach extraction
        if "command" in solution.lower() or "run" in solution.lower():
            return "command_line"
        elif "code" in solution.lower() or "function" in solution.lower():
            return "programming"
        elif "config" in solution.lower() or "setting" in solution.lower():
            return "configuration"
        else:
            return "manual"
    
    def _extract_tags_from_text(self, text: str) -> List[str]:
        """Extract relevant tags from text content"""
        
        # Technical keywords
        technical_keywords = [
            "python", "javascript", "ssl", "certificate", "gpu", "cuda", "api",
            "database", "server", "network", "docker", "kubernetes", "git",
            "ml", "ai", "data", "analysis", "visualization", "jupyter"
        ]
        
        text_lower = text.lower()
        found_tags = []
        
        for keyword in technical_keywords:
            if keyword in text_lower:
                found_tags.append(keyword)
        
        # Extract domain-specific terms
        words = re.findall(r'\b[a-zA-Z]{4,}\b', text_lower)
        important_words = [w for w in words if len(w) >= 4 and w not in found_tags][:5]
        
        found_tags.extend(important_words)
        
        return found_tags[:10]  # Limit to top 10 tags
    
    def _extract_tags_from_content(self, content: Any) -> List[str]:
        """Extract tags from various content types"""
        
        if isinstance(content, str):
            return self._extract_tags_from_text(content)
        elif isinstance(content, dict):
            text = " ".join(str(v) for v in content.values())
            return self._extract_tags_from_text(text)
        elif isinstance(content, list):
            text = " ".join(str(item) for item in content)
            return self._extract_tags_from_text(text)
        else:
            return []
    
    def _calculate_relevance(self, prompt: str, response: str) -> float:
        """Calculate relevance score between prompt and response"""
        
        prompt_words = set(prompt.lower().split())
        response_words = set(response.lower().split())
        
        if not prompt_words:
            return 0.5
        
        # Calculate word overlap
        common_words = prompt_words.intersection(response_words)
        relevance = len(common_words) / len(prompt_words)
        
        return min(1.0, relevance * 2)  # Scale up to account for semantic similarity
    
    def _calculate_relevance_from_context(self, context: Dict[str, Any]) -> float:
        """Calculate relevance from processing context"""
        
        # Base relevance
        relevance = 0.6
        
        # Boost for specific contexts
        if context.get("kb_results"):
            relevance += 0.2
        
        if context.get("cross_repo_analysis"):
            relevance += 0.1
        
        if context.get("validation"):
            relevance += 0.1
        
        return min(1.0, relevance)
    
    def _assess_solution_confidence(self, validation_results: Dict[str, Any]) -> float:
        """Assess confidence based on validation results"""
        
        success_rate = validation_results.get("success_rate", 0.5)
        test_count = len(validation_results.get("tests", []))
        
        # Base confidence from success rate
        confidence = success_rate
        
        # Boost for comprehensive testing
        if test_count >= 3:
            confidence += 0.1
        elif test_count >= 5:
            confidence += 0.2
        
        return min(1.0, confidence)
    
    def get_ingestion_status(self) -> Dict[str, Any]:
        """Get current ingestion system status"""
        
        return {
            "pending_entries": len(self.pending_entries),
            "buffer_size": self.ingestion_buffer_size,
            "known_patterns": len(self.known_patterns),
            "solution_templates": len(self.solution_templates),
            "processing_history": len(self.ingestion_history),
            "quality_thresholds": {
                "min_confidence": self.min_confidence,
                "min_relevance": self.min_relevance
            }
        }
    
    def force_process_pending(self) -> int:
        """Force processing of pending entries"""
        
        pending_count = len(self.pending_entries)
        if pending_count > 0:
            self._process_pending_entries()
        
        return pending_count


def test_knowledge_ingestion():
    """Test the knowledge ingestion system"""
    
    ingestion = KnowledgeIngestionSystem()
    
    # Test AI processing result ingestion
    ai_output = {
        "response": "To fix SSL certificate issues, run the fix-wsl-doi-certs.sh script",
        "confidence": 0.9,
        "insights": {
            "approach": "script_based",
            "domain": "networking"
        },
        "processing_stages": [
            {"stage": "analysis", "status": "completed"},
            {"stage": "solution", "status": "completed"}
        ]
    }
    
    context = {
        "processing_mode": "kb_enhanced",
        "domain": "system_admin"
    }
    
    batch_id = ingestion.ingest_ai_processing_result("test_session", ai_output, context)
    print(f"Ingested AI result: {batch_id}")
    
    # Test user interaction ingestion
    entry_id = ingestion.ingest_user_interaction(
        prompt="How do I fix SSL errors?",
        response="Run the certificate fix script",
        context={"project": "dunes"},
        user_feedback={"helpful": True, "rating": 5}
    )
    print(f"Ingested user interaction: {entry_id}")
    
    # Test problem resolution ingestion
    solution_id = ingestion.ingest_problem_resolution(
        problem_description="SSL certificate validation failing in WSL",
        solution="Install DOI root certificate and update cert bundle",
        validation_results={
            "success_rate": 0.95,
            "method": "automated_test",
            "tests": ["cert_validation", "ssl_connection", "download_test"]
        },
        context={"environment": "wsl", "network": "corporate"}
    )
    print(f"Ingested problem resolution: {solution_id}")
    
    # Force processing
    processed_count = ingestion.force_process_pending()
    print(f"Processed {processed_count} pending entries")
    
    # Get status
    status = ingestion.get_ingestion_status()
    print(f"Ingestion status: {status}")


if __name__ == "__main__":
    test_knowledge_ingestion()