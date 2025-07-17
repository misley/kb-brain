"""
KB Consolidation and Data Partitioning System
Merges existing kb_system and kb_brain into unified structure with intelligent partitioning
"""

import json
import shutil
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import re

logger = logging.getLogger(__name__)


class PartitionStrategy(Enum):
    """Data partitioning strategies"""
    BY_DOMAIN = "by_domain"
    BY_COMPLEXITY = "by_complexity" 
    BY_FREQUENCY = "by_frequency"
    BY_PROJECT = "by_project"
    HYBRID = "hybrid"


class DataCategory(Enum):
    """Categories of knowledge data"""
    SYSTEM_KNOWLEDGE = "system"
    PROJECT_KNOWLEDGE = "project"
    DOMAIN_EXPERTISE = "domain"
    SOLUTION_PATTERNS = "patterns"
    FAILURE_KNOWLEDGE = "failures"
    METHODOLOGIES = "methods"


@dataclass
class KnowledgePartition:
    """Definition of a knowledge partition"""
    id: str
    name: str
    category: DataCategory
    criteria: Dict[str, Any]
    size_limit: int  # Maximum entries before splitting
    current_size: int
    last_updated: datetime
    related_partitions: List[str]
    sme_agent_id: Optional[str] = None


@dataclass
class ConsolidationPlan:
    """Plan for consolidating knowledge bases"""
    source_paths: List[str]
    target_structure: Dict[str, Any]
    migration_steps: List[Dict[str, Any]]
    partitioning_strategy: PartitionStrategy
    estimated_time: int
    backup_location: str


class KBConsolidationSystem:
    """System for consolidating and partitioning knowledge bases"""
    
    def __init__(self, 
                 target_base_path: str = "/mnt/c/Users/misley/Documents/Projects/kb-brain",
                 partition_size_limit: int = 500):
        """
        Initialize KB Consolidation System
        
        Args:
            target_base_path: Target path for unified KB system
            partition_size_limit: Maximum entries per partition
        """
        self.target_base_path = Path(target_base_path)
        self.partition_size_limit = partition_size_limit
        
        # Source paths to consolidate
        self.source_paths = [
            Path("/mnt/c/Users/misley/Documents/Projects/kb_system"),
            Path("/mnt/c/Users/misley/Documents/Projects/kb-brain/kb_brain")
        ]
        
        # Consolidation tracking
        self.partitions: Dict[str, KnowledgePartition] = {}
        self.migration_log: List[Dict[str, Any]] = []
        self.knowledge_map: Dict[str, str] = {}  # original_id -> partition_id
        
        # Analytics
        self.domain_clusters: Dict[str, List[str]] = {}
        self.complexity_analysis: Dict[str, int] = {}
        self.frequency_stats: Dict[str, int] = {}
        
        logger.info("KB Consolidation System initialized")
    
    def analyze_existing_knowledge(self) -> Dict[str, Any]:
        """Analyze existing knowledge bases for consolidation planning"""
        
        logger.info("Analyzing existing knowledge bases")
        
        analysis = {
            "total_files": 0,
            "total_entries": 0,
            "domains_found": set(),
            "projects_found": set(),
            "file_sizes": {},
            "domain_distribution": {},
            "complexity_distribution": {},
            "recommendations": []
        }
        
        for source_path in self.source_paths:
            if source_path.exists():
                path_analysis = self._analyze_source_path(source_path)
                
                # Merge analysis
                analysis["total_files"] += path_analysis["file_count"]
                analysis["total_entries"] += path_analysis["entry_count"]
                analysis["domains_found"].update(path_analysis["domains"])
                analysis["projects_found"].update(path_analysis["projects"])
                analysis["file_sizes"].update(path_analysis["file_sizes"])
                
                # Merge distributions
                for domain, count in path_analysis["domain_dist"].items():
                    analysis["domain_distribution"][domain] = analysis["domain_distribution"].get(domain, 0) + count
                
                for complexity, count in path_analysis["complexity_dist"].items():
                    analysis["complexity_distribution"][complexity] = analysis["complexity_distribution"].get(complexity, 0) + count
        
        # Convert sets to lists for JSON serialization
        analysis["domains_found"] = list(analysis["domains_found"])
        analysis["projects_found"] = list(analysis["projects_found"])
        
        # Generate recommendations
        analysis["recommendations"] = self._generate_consolidation_recommendations(analysis)
        
        logger.info(f"Analysis complete: {analysis['total_files']} files, {analysis['total_entries']} entries")
        
        return analysis
    
    def _analyze_source_path(self, source_path: Path) -> Dict[str, Any]:
        """Analyze a single source path"""
        
        analysis = {
            "file_count": 0,
            "entry_count": 0,
            "domains": set(),
            "projects": set(),
            "file_sizes": {},
            "domain_dist": {},
            "complexity_dist": {}
        }
        
        # Analyze JSON files
        for json_file in source_path.rglob("*.json"):
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                analysis["file_count"] += 1
                analysis["file_sizes"][str(json_file)] = json_file.stat().st_size
                
                # Analyze content
                file_analysis = self._analyze_json_content(data, json_file)
                
                analysis["entry_count"] += file_analysis["entries"]
                analysis["domains"].update(file_analysis["domains"])
                analysis["projects"].update(file_analysis["projects"])
                
                # Merge distributions
                for domain, count in file_analysis["domain_dist"].items():
                    analysis["domain_dist"][domain] = analysis["domain_dist"].get(domain, 0) + count
                
                for complexity, count in file_analysis["complexity_dist"].items():
                    analysis["complexity_dist"][complexity] = analysis["complexity_dist"].get(complexity, 0) + count
                    
            except Exception as e:
                logger.error(f"Error analyzing {json_file}: {e}")
        
        return analysis
    
    def _analyze_json_content(self, data: Any, file_path: Path) -> Dict[str, Any]:
        """Analyze JSON content for knowledge patterns"""
        
        analysis = {
            "entries": 0,
            "domains": set(),
            "projects": set(),
            "domain_dist": {},
            "complexity_dist": {}
        }
        
        # Detect domains and projects from file path
        path_str = str(file_path).lower()
        
        # Project detection
        project_indicators = ["dunes", "whsa", "mspc", "chdn", "wildlife", "springs"]
        for indicator in project_indicators:
            if indicator in path_str:
                analysis["projects"].add(indicator)
        
        # Domain detection from filename
        domain_indicators = {
            "troubleshooting": "technical",
            "dunes": "geospatial", 
            "mspc": "data_science",
            "core": "general",
            "research": "research"
        }
        
        for indicator, domain in domain_indicators.items():
            if indicator in path_str:
                analysis["domains"].add(domain)
        
        # Analyze data structure
        if isinstance(data, dict):
            analysis["entries"] += self._count_knowledge_entries(data)
            
            # Extract domains from content
            content_domains = self._extract_domains_from_content(data)
            analysis["domains"].update(content_domains)
            
            # Update distributions
            for domain in content_domains:
                analysis["domain_dist"][domain] = analysis["domain_dist"].get(domain, 0) + 1
        
        return analysis
    
    def _count_knowledge_entries(self, data: Dict[str, Any]) -> int:
        """Count knowledge entries in data structure"""
        
        count = 0
        
        # Common KB structure patterns
        entry_sections = [
            "solved_problems", "active_problems", "archived_problems",
            "failed_approaches", "lessons_learned", "insights",
            "patterns", "solutions", "knowledge_entries"
        ]
        
        for section in entry_sections:
            if section in data and isinstance(data[section], list):
                count += len(data[section])
        
        # Count top-level entries if no sections found
        if count == 0 and isinstance(data, dict):
            count = len([v for v in data.values() if isinstance(v, (dict, list))])
        
        return count
    
    def _extract_domains_from_content(self, data: Dict[str, Any]) -> Set[str]:
        """Extract domain information from content"""
        
        domains = set()
        
        # Convert data to searchable text
        content_text = json.dumps(data).lower()
        
        # Domain keyword mapping
        domain_keywords = {
            "geospatial": ["gis", "landsat", "satellite", "boundary", "ndwi", "remote sensing", "spatial"],
            "data_science": ["ml", "model", "analysis", "pandas", "numpy", "scikit", "jupyter"],
            "technical": ["ssl", "certificate", "server", "network", "error", "debug", "config"],
            "programming": ["python", "javascript", "code", "function", "class", "api", "framework"],
            "research": ["publication", "paper", "study", "findings", "methodology", "validation"],
            "environmental": ["ecosystem", "monitoring", "water", "environmental", "species"]
        }
        
        for domain, keywords in domain_keywords.items():
            if any(keyword in content_text for keyword in keywords):
                domains.add(domain)
        
        return domains
    
    def _generate_consolidation_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate consolidation recommendations based on analysis"""
        
        recommendations = []
        
        # Size-based recommendations
        total_entries = analysis["total_entries"]
        if total_entries > 1000:
            recommendations.append("Large knowledge base detected - recommend domain-based partitioning")
        
        # Domain distribution recommendations
        domains = analysis["domain_distribution"]
        if len(domains) > 5:
            recommendations.append("Multiple domains detected - recommend SME agent specialization")
        
        # Largest domains for SME creation
        sorted_domains = sorted(domains.items(), key=lambda x: x[1], reverse=True)
        if sorted_domains and sorted_domains[0][1] > 100:
            recommendations.append(f"Domain '{sorted_domains[0][0]}' has {sorted_domains[0][1]} entries - ready for SME specialization")
        
        # Project separation recommendations
        projects = analysis["projects_found"]
        if len(projects) > 3:
            recommendations.append("Multiple projects detected - recommend project-based knowledge organization")
        
        return recommendations
    
    def create_consolidation_plan(self, 
                                analysis: Dict[str, Any],
                                strategy: PartitionStrategy = PartitionStrategy.HYBRID) -> ConsolidationPlan:
        """Create a detailed consolidation plan"""
        
        logger.info(f"Creating consolidation plan with {strategy.value} strategy")
        
        # Define target structure
        target_structure = {
            "unified_kb": {
                "system": "System-wide knowledge",
                "domains": "Domain-specific partitions",
                "projects": "Project-specific knowledge",
                "patterns": "Solution patterns and methodologies",
                "archive": "Historical and low-priority knowledge"
            },
            "sme_agents": "SME agent knowledge bases",
            "backup": "Backup of original structures"
        }
        
        # Create migration steps
        migration_steps = []
        
        # Step 1: Backup original structures
        migration_steps.append({
            "step": 1,
            "action": "backup_original",
            "description": "Create backup of original KB structures",
            "estimated_time": 2
        })
        
        # Step 2: Create unified structure
        migration_steps.append({
            "step": 2,
            "action": "create_structure",
            "description": "Create unified KB directory structure",
            "estimated_time": 1
        })
        
        # Step 3: Migrate and partition data
        migration_steps.append({
            "step": 3,
            "action": "migrate_partition",
            "description": "Migrate data with intelligent partitioning",
            "estimated_time": analysis["total_files"] * 2  # 2 minutes per file
        })
        
        # Step 4: Create SME-ready partitions
        migration_steps.append({
            "step": 4,
            "action": "prepare_sme",
            "description": "Prepare partitions for SME agent creation",
            "estimated_time": len(analysis["domains_found"]) * 5
        })
        
        # Step 5: Validation and testing
        migration_steps.append({
            "step": 5,
            "action": "validate",
            "description": "Validate consolidated structure",
            "estimated_time": 10
        })
        
        # Estimate total time
        total_time = sum(step["estimated_time"] for step in migration_steps)
        
        # Create backup location
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_location = str(self.target_base_path / "backups" / f"kb_backup_{timestamp}")
        
        plan = ConsolidationPlan(
            source_paths=[str(p) for p in self.source_paths],
            target_structure=target_structure,
            migration_steps=migration_steps,
            partitioning_strategy=strategy,
            estimated_time=total_time,
            backup_location=backup_location
        )
        
        logger.info(f"Consolidation plan created: {total_time} minutes estimated")
        
        return plan
    
    def execute_consolidation(self, plan: ConsolidationPlan) -> Dict[str, Any]:
        """Execute the consolidation plan"""
        
        logger.info("Executing KB consolidation plan")
        
        execution_log = {
            "start_time": datetime.now(),
            "plan": asdict(plan),
            "steps_completed": [],
            "errors": [],
            "final_structure": {},
            "migration_stats": {}
        }
        
        try:
            # Execute each step
            for step in plan.migration_steps:
                logger.info(f"Executing step {step['step']}: {step['description']}")
                
                step_result = self._execute_migration_step(step, plan)
                
                execution_log["steps_completed"].append({
                    "step": step["step"],
                    "action": step["action"],
                    "result": step_result,
                    "completed_at": datetime.now().isoformat()
                })
                
                if step_result.get("error"):
                    execution_log["errors"].append(step_result["error"])
            
            # Generate final structure report
            execution_log["final_structure"] = self._analyze_final_structure()
            execution_log["migration_stats"] = self._generate_migration_stats()
            
        except Exception as e:
            logger.error(f"Consolidation failed: {e}")
            execution_log["errors"].append(str(e))
        
        execution_log["end_time"] = datetime.now()
        execution_log["total_time"] = (execution_log["end_time"] - execution_log["start_time"]).total_seconds()
        
        # Save execution log
        log_file = self.target_base_path / "consolidation_log.json"
        with open(log_file, 'w') as f:
            json.dump(execution_log, f, indent=2, default=str)
        
        logger.info(f"Consolidation completed in {execution_log['total_time']:.1f} seconds")
        
        return execution_log
    
    def _execute_migration_step(self, step: Dict[str, Any], plan: ConsolidationPlan) -> Dict[str, Any]:
        """Execute a single migration step"""
        
        action = step["action"]
        result = {"success": False, "details": ""}
        
        try:
            if action == "backup_original":
                result = self._backup_original_structures(plan.backup_location)
            
            elif action == "create_structure":
                result = self._create_unified_structure(plan.target_structure)
            
            elif action == "migrate_partition":
                result = self._migrate_and_partition_data(plan.partitioning_strategy)
            
            elif action == "prepare_sme":
                result = self._prepare_sme_partitions()
            
            elif action == "validate":
                result = self._validate_consolidated_structure()
            
            else:
                result = {"success": False, "error": f"Unknown action: {action}"}
                
        except Exception as e:
            result = {"success": False, "error": str(e)}
        
        return result
    
    def _backup_original_structures(self, backup_location: str) -> Dict[str, Any]:
        """Backup original KB structures"""
        
        backup_path = Path(backup_location)
        backup_path.mkdir(parents=True, exist_ok=True)
        
        backed_up = []
        
        for source_path in self.source_paths:
            if source_path.exists():
                target_backup = backup_path / source_path.name
                shutil.copytree(source_path, target_backup, dirs_exist_ok=True)
                backed_up.append(str(source_path))
        
        return {
            "success": True,
            "details": f"Backed up {len(backed_up)} directories to {backup_location}",
            "backed_up_paths": backed_up
        }
    
    def _create_unified_structure(self, target_structure: Dict[str, Any]) -> Dict[str, Any]:
        """Create unified directory structure"""
        
        unified_path = self.target_base_path / "unified_kb"
        unified_path.mkdir(exist_ok=True)
        
        created_dirs = []
        
        for dir_name, description in target_structure["unified_kb"].items():
            dir_path = unified_path / dir_name
            dir_path.mkdir(exist_ok=True)
            created_dirs.append(dir_name)
            
            # Create metadata file
            metadata = {
                "directory": dir_name,
                "description": description,
                "created_at": datetime.now().isoformat(),
                "purpose": "Unified KB consolidation"
            }
            
            with open(dir_path / "metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
        
        # Create SME agents directory
        sme_dir = self.target_base_path / "sme_agents"
        sme_dir.mkdir(exist_ok=True)
        created_dirs.append("sme_agents")
        
        return {
            "success": True,
            "details": f"Created {len(created_dirs)} directories",
            "created_directories": created_dirs
        }
    
    def _migrate_and_partition_data(self, strategy: PartitionStrategy) -> Dict[str, Any]:
        """Migrate and partition knowledge data"""
        
        migrated_files = []
        partition_stats = {}
        
        for source_path in self.source_paths:
            if source_path.exists():
                for json_file in source_path.rglob("*.json"):
                    try:
                        # Load and analyze file
                        with open(json_file, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                        
                        # Determine partition based on strategy
                        partition_id = self._determine_partition(json_file, data, strategy)
                        
                        # Migrate to appropriate partition
                        target_file = self._get_partition_file_path(partition_id, json_file.name)
                        
                        # Enhance data with migration metadata
                        if isinstance(data, dict):
                            data["_migration_metadata"] = {
                                "original_path": str(json_file),
                                "migrated_at": datetime.now().isoformat(),
                                "partition_id": partition_id,
                                "strategy": strategy.value
                            }
                        
                        # Save to new location
                        target_file.parent.mkdir(parents=True, exist_ok=True)
                        with open(target_file, 'w') as f:
                            json.dump(data, f, indent=2)
                        
                        migrated_files.append(str(json_file))
                        partition_stats[partition_id] = partition_stats.get(partition_id, 0) + 1
                        
                    except Exception as e:
                        logger.error(f"Error migrating {json_file}: {e}")
        
        return {
            "success": True,
            "details": f"Migrated {len(migrated_files)} files across {len(partition_stats)} partitions",
            "migrated_files": migrated_files,
            "partition_distribution": partition_stats
        }
    
    def _determine_partition(self, file_path: Path, data: Any, strategy: PartitionStrategy) -> str:
        """Determine which partition a file should go to"""
        
        if strategy == PartitionStrategy.BY_DOMAIN:
            return self._determine_domain_partition(file_path, data)
        elif strategy == PartitionStrategy.BY_PROJECT:
            return self._determine_project_partition(file_path, data)
        elif strategy == PartitionStrategy.HYBRID:
            return self._determine_hybrid_partition(file_path, data)
        else:
            return "general"
    
    def _determine_domain_partition(self, file_path: Path, data: Any) -> str:
        """Determine domain-based partition"""
        
        path_str = str(file_path).lower()
        
        # Domain mapping
        domain_map = {
            "dunes": "geospatial",
            "troubleshooting": "technical", 
            "mspc": "data_science",
            "research": "research",
            "system": "system"
        }
        
        for indicator, domain in domain_map.items():
            if indicator in path_str:
                return domain
        
        # Analyze content for domain
        if isinstance(data, dict):
            content_text = json.dumps(data).lower()
            
            if any(word in content_text for word in ["landsat", "satellite", "boundary"]):
                return "geospatial"
            elif any(word in content_text for word in ["ssl", "certificate", "error"]):
                return "technical"
            elif any(word in content_text for word in ["ml", "model", "analysis"]):
                return "data_science"
        
        return "general"
    
    def _determine_project_partition(self, file_path: Path, data: Any) -> str:
        """Determine project-based partition"""
        
        path_str = str(file_path).lower()
        
        project_indicators = ["dunes", "whsa", "mspc", "chdn", "wildlife", "springs"]
        
        for project in project_indicators:
            if project in path_str:
                return f"project_{project}"
        
        return "project_general"
    
    def _determine_hybrid_partition(self, file_path: Path, data: Any) -> str:
        """Determine partition using hybrid strategy"""
        
        # Combine domain and project information
        domain = self._determine_domain_partition(file_path, data)
        project = self._determine_project_partition(file_path, data)
        
        # Create hybrid partition ID
        if project != "project_general":
            return f"{domain}_{project}"
        else:
            return domain
    
    def _get_partition_file_path(self, partition_id: str, filename: str) -> Path:
        """Get target file path for a partition"""
        
        # Determine partition directory based on partition ID
        if "project_" in partition_id:
            partition_dir = self.target_base_path / "unified_kb" / "projects" / partition_id
        elif partition_id in ["geospatial", "technical", "data_science", "research"]:
            partition_dir = self.target_base_path / "unified_kb" / "domains" / partition_id
        elif partition_id == "system":
            partition_dir = self.target_base_path / "unified_kb" / "system"
        else:
            partition_dir = self.target_base_path / "unified_kb" / "general"
        
        return partition_dir / filename
    
    def _prepare_sme_partitions(self) -> Dict[str, Any]:
        """Prepare partitions for SME agent creation"""
        
        unified_path = self.target_base_path / "unified_kb"
        prepared_partitions = []
        
        # Analyze domain partitions for SME readiness
        domains_path = unified_path / "domains"
        if domains_path.exists():
            for domain_dir in domains_path.iterdir():
                if domain_dir.is_dir():
                    # Count knowledge entries
                    entry_count = 0
                    for json_file in domain_dir.glob("*.json"):
                        try:
                            with open(json_file, 'r') as f:
                                data = json.load(f)
                            entry_count += self._count_knowledge_entries(data)
                        except Exception:
                            continue
                    
                    # Create partition metadata
                    if entry_count >= 50:  # Threshold for SME creation
                        partition = KnowledgePartition(
                            id=f"sme_ready_{domain_dir.name}",
                            name=domain_dir.name.title(),
                            category=DataCategory.DOMAIN_EXPERTISE,
                            criteria={"domain": domain_dir.name, "min_entries": 50},
                            size_limit=self.partition_size_limit,
                            current_size=entry_count,
                            last_updated=datetime.now(),
                            related_partitions=[]
                        )
                        
                        self.partitions[partition.id] = partition
                        prepared_partitions.append(partition.id)
                        
                        # Save partition metadata
                        metadata_file = domain_dir / "sme_metadata.json"
                        with open(metadata_file, 'w') as f:
                            json.dump(asdict(partition), f, indent=2, default=str)
        
        return {
            "success": True,
            "details": f"Prepared {len(prepared_partitions)} partitions for SME creation",
            "sme_ready_partitions": prepared_partitions
        }
    
    def _validate_consolidated_structure(self) -> Dict[str, Any]:
        """Validate the consolidated structure"""
        
        validation_results = {
            "structure_valid": True,
            "missing_directories": [],
            "file_integrity": True,
            "partition_stats": {},
            "recommendations": []
        }
        
        # Check required directories
        required_dirs = [
            "unified_kb/system",
            "unified_kb/domains", 
            "unified_kb/projects",
            "unified_kb/patterns",
            "sme_agents"
        ]
        
        for dir_path in required_dirs:
            full_path = self.target_base_path / dir_path
            if not full_path.exists():
                validation_results["missing_directories"].append(dir_path)
                validation_results["structure_valid"] = False
        
        # Validate file integrity
        unified_path = self.target_base_path / "unified_kb"
        total_files = 0
        valid_files = 0
        
        for json_file in unified_path.rglob("*.json"):
            total_files += 1
            try:
                with open(json_file, 'r') as f:
                    json.load(f)
                valid_files += 1
            except Exception:
                validation_results["file_integrity"] = False
        
        validation_results["partition_stats"] = {
            "total_files": total_files,
            "valid_files": valid_files,
            "integrity_rate": valid_files / total_files if total_files > 0 else 1.0
        }
        
        # Generate recommendations
        if validation_results["structure_valid"] and validation_results["file_integrity"]:
            validation_results["recommendations"].append("Consolidation successful - ready for SME system initialization")
        else:
            validation_results["recommendations"].append("Issues found - review missing directories and file integrity")
        
        return validation_results
    
    def _analyze_final_structure(self) -> Dict[str, Any]:
        """Analyze the final consolidated structure"""
        
        unified_path = self.target_base_path / "unified_kb"
        
        structure_analysis = {
            "total_directories": 0,
            "total_files": 0,
            "partition_sizes": {},
            "domain_distribution": {},
            "project_distribution": {}
        }
        
        if unified_path.exists():
            for item in unified_path.rglob("*"):
                if item.is_dir():
                    structure_analysis["total_directories"] += 1
                elif item.is_file() and item.suffix == ".json":
                    structure_analysis["total_files"] += 1
                    
                    # Analyze partition sizes
                    partition_name = item.parent.name
                    structure_analysis["partition_sizes"][partition_name] = structure_analysis["partition_sizes"].get(partition_name, 0) + 1
        
        return structure_analysis
    
    def _generate_migration_stats(self) -> Dict[str, Any]:
        """Generate migration statistics"""
        
        return {
            "partitions_created": len(self.partitions),
            "knowledge_entries_migrated": sum(p.current_size for p in self.partitions.values()),
            "sme_ready_domains": len([p for p in self.partitions.values() if p.current_size >= 50]),
            "migration_log_entries": len(self.migration_log)
        }
    
    def get_consolidation_status(self) -> Dict[str, Any]:
        """Get current consolidation status"""
        
        return {
            "target_path": str(self.target_base_path),
            "source_paths": [str(p) for p in self.source_paths],
            "partitions": {pid: asdict(partition) for pid, partition in self.partitions.items()},
            "migration_log_size": len(self.migration_log),
            "system_ready": len(self.partitions) > 0
        }


def test_consolidation_system():
    """Test the KB consolidation system"""
    
    consolidation = KBConsolidationSystem()
    
    # Analyze existing knowledge
    print("=== Analyzing Existing Knowledge ===")
    analysis = consolidation.analyze_existing_knowledge()
    print(json.dumps(analysis, indent=2, default=str))
    
    # Create consolidation plan
    print("\n=== Creating Consolidation Plan ===")
    plan = consolidation.create_consolidation_plan(analysis)
    print(f"Plan created: {plan.estimated_time} minutes estimated")
    print(f"Steps: {len(plan.migration_steps)}")
    
    # Execute consolidation (commented out for safety)
    # print("\n=== Executing Consolidation ===")
    # execution_log = consolidation.execute_consolidation(plan)
    # print(f"Execution completed: {execution_log['total_time']:.1f} seconds")
    
    # Get status
    status = consolidation.get_consolidation_status()
    print(f"\nConsolidation status: {status}")


if __name__ == "__main__":
    test_consolidation_system()