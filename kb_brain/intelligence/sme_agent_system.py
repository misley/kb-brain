"""
Subject Matter Expert (SME) Agent System
Hierarchical knowledge architecture with automatic specialization and parent-child relationships
"""

import json
import uuid
import hashlib
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import asyncio

from .kb_integration_engine import KBIntegrationEngine, KnowledgeResponse
from .prompt_classifier import PromptClassifier, PromptClassification
from .knowledge_ingestion import KnowledgeIngestionSystem

logger = logging.getLogger(__name__)


class SMEStatus(Enum):
    """SME Agent Status"""
    SPAWNING = "spawning"
    ACTIVE = "active"
    SPECIALIZED = "specialized"
    MATURE = "mature"
    DORMANT = "dormant"


class ExpertiseLevel(Enum):
    """Level of expertise in a domain"""
    NOVICE = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


@dataclass
class DomainExpertise:
    """Domain expertise metrics"""
    domain: str
    knowledge_count: int
    solution_success_rate: float
    query_volume: int
    complexity_handled: int
    last_activity: datetime
    expertise_level: ExpertiseLevel
    confidence_score: float
    
    def calculate_specialization_score(self) -> float:
        """Calculate if this domain warrants SME specialization"""
        # Factors: volume, success rate, complexity, recency
        volume_score = min(1.0, self.knowledge_count / 100)  # 100 knowledge entries = full score
        success_score = self.solution_success_rate
        complexity_score = min(1.0, self.complexity_handled / 10)  # 10 complex solutions = full score
        recency_days = (datetime.now() - self.last_activity).days
        recency_score = max(0.0, 1.0 - (recency_days / 30))  # 30 days = 0 score
        
        return (volume_score * 0.4 + success_score * 0.3 + 
                complexity_score * 0.2 + recency_score * 0.1)


@dataclass
class SMEAgent:
    """Subject Matter Expert Agent"""
    id: str
    domain: str
    parent_id: Optional[str]
    children_ids: List[str]
    status: SMEStatus
    expertise: DomainExpertise
    knowledge_base_path: str
    creation_date: datetime
    last_summary_update: datetime
    specialization_triggers: List[str]
    
    # Agent capabilities
    kb_engine: Optional[KBIntegrationEngine] = None
    prompt_classifier: Optional[PromptClassifier] = None
    ingestion_system: Optional[KnowledgeIngestionSystem] = None
    
    def __post_init__(self):
        if not self.children_ids:
            self.children_ids = []
        if not self.specialization_triggers:
            self.specialization_triggers = []


@dataclass
class SMESummary:
    """Summary report from SME to parent"""
    sme_id: str
    domain: str
    period_start: datetime
    period_end: datetime
    queries_handled: int
    solutions_provided: int
    success_rate: float
    new_knowledge_entries: int
    key_insights: List[str]
    problem_areas: List[str]
    recommended_actions: List[str]
    expertise_evolution: Dict[str, Any]


class SMEAgentSystem:
    """Hierarchical SME Agent Management System"""
    
    def __init__(self, 
                 base_kb_path: str = "/mnt/c/Users/misley/Documents/Projects/kb-brain",
                 specialization_threshold: float = 0.7,
                 max_sme_depth: int = 3):
        """
        Initialize SME Agent System
        
        Args:
            base_kb_path: Base path for knowledge bases
            specialization_threshold: Threshold for creating new SME
            max_sme_depth: Maximum depth of SME hierarchy
        """
        self.base_kb_path = Path(base_kb_path)
        self.specialization_threshold = specialization_threshold
        self.max_sme_depth = max_sme_depth
        
        # Agent tracking
        self.agents: Dict[str, SMEAgent] = {}
        self.domain_map: Dict[str, str] = {}  # domain -> agent_id
        self.parent_agent_id: Optional[str] = None
        
        # System components
        self.kb_engine = KBIntegrationEngine()
        self.prompt_classifier = PromptClassifier()
        self.ingestion_system = KnowledgeIngestionSystem()
        
        # Metrics and monitoring
        self.specialization_history: List[Dict[str, Any]] = []
        self.query_routing_stats: Dict[str, int] = {}
        
        self._initialize_system()
        
        logger.info("SME Agent System initialized")
    
    def _initialize_system(self):
        """Initialize the SME system"""
        
        # Create SME directories
        sme_dir = self.base_kb_path / "sme_agents"
        sme_dir.mkdir(exist_ok=True)
        
        # Create parent agent (generalist)
        self.parent_agent_id = self._create_parent_agent()
        
        # Load existing SME agents
        self._load_existing_smes()
        
        # Analyze current knowledge for potential specializations
        self._analyze_specialization_opportunities()
    
    def _create_parent_agent(self) -> str:
        """Create the parent (generalist) agent"""
        
        parent_id = "parent_agent_" + str(uuid.uuid4())[:8]
        
        parent_expertise = DomainExpertise(
            domain="general",
            knowledge_count=0,
            solution_success_rate=0.8,
            query_volume=0,
            complexity_handled=0,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.EXPERT,
            confidence_score=0.9
        )
        
        parent_agent = SMEAgent(
            id=parent_id,
            domain="general",
            parent_id=None,
            children_ids=[],
            status=SMEStatus.ACTIVE,
            expertise=parent_expertise,
            knowledge_base_path=str(self.base_kb_path / "kb_system"),
            creation_date=datetime.now(),
            last_summary_update=datetime.now(),
            specialization_triggers=["knowledge_volume", "query_complexity", "domain_clustering"],
            kb_engine=self.kb_engine,
            prompt_classifier=self.prompt_classifier,
            ingestion_system=self.ingestion_system
        )
        
        self.agents[parent_id] = parent_agent
        
        logger.info(f"Created parent agent: {parent_id}")
        
        return parent_id
    
    def _load_existing_smes(self):
        """Load existing SME agents from disk"""
        
        sme_dir = self.base_kb_path / "sme_agents"
        
        for sme_file in sme_dir.glob("sme_*.json"):
            try:
                with open(sme_file, 'r') as f:
                    sme_data = json.load(f)
                
                # Reconstruct SME agent
                sme_agent = self._reconstruct_sme_from_data(sme_data)
                self.agents[sme_agent.id] = sme_agent
                self.domain_map[sme_agent.domain] = sme_agent.id
                
                logger.info(f"Loaded SME agent: {sme_agent.domain}")
                
            except Exception as e:
                logger.error(f"Error loading SME {sme_file}: {e}")
    
    def _reconstruct_sme_from_data(self, sme_data: Dict[str, Any]) -> SMEAgent:
        """Reconstruct SME agent from saved data"""
        
        # Reconstruct expertise
        expertise_data = sme_data["expertise"]
        expertise = DomainExpertise(
            domain=expertise_data["domain"],
            knowledge_count=expertise_data["knowledge_count"],
            solution_success_rate=expertise_data["solution_success_rate"],
            query_volume=expertise_data["query_volume"],
            complexity_handled=expertise_data["complexity_handled"],
            last_activity=datetime.fromisoformat(expertise_data["last_activity"]),
            expertise_level=ExpertiseLevel(expertise_data["expertise_level"]),
            confidence_score=expertise_data["confidence_score"]
        )
        
        # Create specialized components for this SME
        sme_kb_engine = KBIntegrationEngine(kb_system_path=sme_data["knowledge_base_path"])
        sme_ingestion = KnowledgeIngestionSystem(kb_system_path=sme_data["knowledge_base_path"])
        
        # Reconstruct SME agent
        return SMEAgent(
            id=sme_data["id"],
            domain=sme_data["domain"],
            parent_id=sme_data["parent_id"],
            children_ids=sme_data["children_ids"],
            status=SMEStatus(sme_data["status"]),
            expertise=expertise,
            knowledge_base_path=sme_data["knowledge_base_path"],
            creation_date=datetime.fromisoformat(sme_data["creation_date"]),
            last_summary_update=datetime.fromisoformat(sme_data["last_summary_update"]),
            specialization_triggers=sme_data["specialization_triggers"],
            kb_engine=sme_kb_engine,
            prompt_classifier=self.prompt_classifier,
            ingestion_system=sme_ingestion
        )
    
    def _analyze_specialization_opportunities(self):
        """Analyze current knowledge for SME specialization opportunities"""
        
        # Get domain expertise analysis from parent agent
        parent_agent = self.agents[self.parent_agent_id]
        
        # Analyze knowledge domains
        domain_analysis = self._analyze_knowledge_domains(parent_agent)
        
        # Check for specialization candidates
        for domain, metrics in domain_analysis.items():
            if metrics.calculate_specialization_score() >= self.specialization_threshold:
                if domain not in self.domain_map:
                    logger.info(f"Domain {domain} ready for SME specialization")
                    self._create_sme_agent(domain, metrics, parent_agent.id)
    
    def _analyze_knowledge_domains(self, agent: SMEAgent) -> Dict[str, DomainExpertise]:
        """Analyze knowledge domains in an agent's knowledge base"""
        
        domain_stats = {}
        
        # Search agent's knowledge base for domain patterns
        if agent.kb_engine:
            kb_status = agent.kb_engine.get_system_status()
            
            # Analyze by domain from KB sources
            for domain, source_count in kb_status.get("sources_by_domain", {}).items():
                if domain not in domain_stats:
                    domain_stats[domain] = DomainExpertise(
                        domain=domain,
                        knowledge_count=source_count * 10,  # Estimate
                        solution_success_rate=0.7,  # Default
                        query_volume=source_count * 5,  # Estimate
                        complexity_handled=source_count // 2,  # Estimate
                        last_activity=datetime.now(),
                        expertise_level=ExpertiseLevel.INTERMEDIATE,
                        confidence_score=0.6
                    )
        
        # Enhance with actual usage data if available
        # TODO: Integrate with query routing stats and solution success tracking
        
        return domain_stats
    
    async def route_query(self, 
                         prompt: str, 
                         context: Optional[Dict[str, Any]] = None) -> Tuple[str, Any]:
        """
        Route query to appropriate SME agent
        
        Args:
            prompt: The query prompt
            context: Optional context
        
        Returns:
            Tuple of (agent_id, response)
        """
        
        # Classify prompt to determine domain
        classification = self.prompt_classifier.classify_prompt(prompt, context)
        domain = classification.domain.value
        
        # Update routing stats
        self.query_routing_stats[domain] = self.query_routing_stats.get(domain, 0) + 1
        
        # Route to appropriate agent
        if domain in self.domain_map:
            # Route to specialized SME
            sme_id = self.domain_map[domain]
            sme_agent = self.agents[sme_id]
            
            logger.info(f"Routing {domain} query to SME: {sme_id}")
            response = await self._process_with_sme(sme_agent, prompt, context, classification)
            
            # Update SME metrics
            self._update_sme_metrics(sme_agent, classification, response)
            
            return sme_id, response
        else:
            # Route to parent agent
            parent_agent = self.agents[self.parent_agent_id]
            
            logger.info(f"Routing {domain} query to parent agent")
            response = await self._process_with_parent(parent_agent, prompt, context, classification)
            
            # Check if this interaction suggests SME creation
            await self._check_for_sme_creation(domain, classification, response)
            
            return self.parent_agent_id, response
    
    async def _process_with_sme(self, 
                              sme_agent: SMEAgent, 
                              prompt: str, 
                              context: Dict[str, Any], 
                              classification: PromptClassification) -> Dict[str, Any]:
        """Process query with specialized SME agent"""
        
        # Use SME's specialized knowledge base
        kb_response = sme_agent.kb_engine.search_knowledge(
            query=prompt,
            domain_filter=sme_agent.domain,
            max_results=10
        )
        
        # Generate specialized response
        response = {
            "response": f"SME {sme_agent.domain} response: {kb_response.results[0].content if kb_response.results else 'No specific knowledge found'}",
            "sme_domain": sme_agent.domain,
            "expertise_level": sme_agent.expertise.expertise_level.value,
            "confidence": kb_response.confidence,
            "kb_results": kb_response.results[:3],  # Top 3 results
            "specialized": True
        }
        
        # If SME can't handle complexity, escalate to parent
        if classification.complexity.value >= 3 and sme_agent.expertise.expertise_level.value < 4:
            response["escalated_to_parent"] = True
            parent_response = await self._escalate_to_parent(prompt, context, sme_agent.id)
            response["parent_insights"] = parent_response
        
        return response
    
    async def _process_with_parent(self, 
                                 parent_agent: SMEAgent, 
                                 prompt: str, 
                                 context: Dict[str, Any], 
                                 classification: PromptClassification) -> Dict[str, Any]:
        """Process query with parent (generalist) agent"""
        
        # Use parent's comprehensive knowledge base
        kb_response = parent_agent.kb_engine.search_knowledge(
            query=prompt,
            search_depth=self._determine_search_depth(classification),
            max_results=5
        )
        
        response = {
            "response": f"General analysis: {kb_response.results[0].content if kb_response.results else 'Analysis based on general knowledge'}",
            "domain": "general",
            "kb_results": kb_response.results,
            "cross_repo_insights": kb_response.cross_repo_insights,
            "specialized": False
        }
        
        # Check if any SME children should be consulted
        relevant_smes = self._find_relevant_child_smes(classification.domain.value)
        if relevant_smes:
            sme_consultations = await self._consult_child_smes(relevant_smes, prompt, context)
            response["sme_consultations"] = sme_consultations
        
        return response
    
    async def _escalate_to_parent(self, prompt: str, context: Dict[str, Any], sme_id: str) -> Dict[str, Any]:
        """Escalate complex query from SME to parent"""
        
        parent_agent = self.agents[self.parent_agent_id]
        escalation_context = context.copy()
        escalation_context["escalated_from_sme"] = sme_id
        escalation_context["escalation_reason"] = "complexity_exceeded"
        
        # Parent processes with full context
        kb_response = parent_agent.kb_engine.search_knowledge(
            query=prompt,
            search_depth="comprehensive",
            max_results=8
        )
        
        return {
            "escalation_response": f"Parent analysis of escalated query: {kb_response.results[0].content if kb_response.results else 'Complex analysis required'}",
            "escalation_insights": kb_response.cross_repo_insights,
            "recommended_sme_learning": self._generate_sme_learning_recommendations(sme_id, kb_response)
        }
    
    def _find_relevant_child_smes(self, domain: str) -> List[SMEAgent]:
        """Find child SMEs relevant to a domain"""
        
        relevant_smes = []
        
        for agent_id, agent in self.agents.items():
            if (agent.parent_id == self.parent_agent_id and 
                (agent.domain == domain or domain in agent.domain or agent.domain in domain)):
                relevant_smes.append(agent)
        
        return relevant_smes
    
    async def _consult_child_smes(self, 
                                smes: List[SMEAgent], 
                                prompt: str, 
                                context: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Consult multiple child SMEs for their perspectives"""
        
        consultations = []
        
        for sme in smes:
            try:
                consultation = await self._get_sme_consultation(sme, prompt, context)
                consultations.append({
                    "sme_id": sme.id,
                    "domain": sme.domain,
                    "consultation": consultation
                })
            except Exception as e:
                logger.error(f"Error consulting SME {sme.id}: {e}")
        
        return consultations
    
    async def _get_sme_consultation(self, 
                                  sme: SMEAgent, 
                                  prompt: str, 
                                  context: Dict[str, Any]) -> Dict[str, Any]:
        """Get consultation from a specific SME"""
        
        # Quick domain-specific search
        kb_response = sme.kb_engine.search_knowledge(
            query=prompt,
            domain_filter=sme.domain,
            max_results=3
        )
        
        return {
            "has_relevant_knowledge": len(kb_response.results) > 0,
            "confidence": kb_response.confidence,
            "key_insights": [r.content for r in kb_response.results[:2]],
            "recommendations": kb_response.recommendations[:2]
        }
    
    async def _check_for_sme_creation(self, 
                                    domain: str, 
                                    classification: PromptClassification, 
                                    response: Dict[str, Any]):
        """Check if this interaction suggests creating a new SME"""
        
        # Analyze domain activity
        domain_activity = self.query_routing_stats.get(domain, 0)
        
        # Check complexity and frequency thresholds
        if (domain_activity >= 10 and  # Minimum query volume
            classification.complexity.value >= 2 and  # Moderate complexity
            domain not in self.domain_map):  # No existing SME
            
            # Analyze potential for specialization
            potential_metrics = DomainExpertise(
                domain=domain,
                knowledge_count=domain_activity * 2,  # Estimate
                solution_success_rate=0.7,  # Default
                query_volume=domain_activity,
                complexity_handled=domain_activity // 2,
                last_activity=datetime.now(),
                expertise_level=ExpertiseLevel.INTERMEDIATE,
                confidence_score=0.6
            )
            
            if potential_metrics.calculate_specialization_score() >= self.specialization_threshold:
                logger.info(f"Triggering SME creation for domain: {domain}")
                await self._create_sme_agent(domain, potential_metrics, self.parent_agent_id)
    
    async def _create_sme_agent(self, 
                              domain: str, 
                              expertise: DomainExpertise, 
                              parent_id: str) -> str:
        """Create a new SME agent for a domain"""
        
        sme_id = f"sme_{domain}_{str(uuid.uuid4())[:8]}"
        
        # Create SME-specific knowledge base directory
        sme_kb_path = self.base_kb_path / "sme_agents" / sme_id
        sme_kb_path.mkdir(exist_ok=True)
        
        # Create SME agent
        sme_agent = SMEAgent(
            id=sme_id,
            domain=domain,
            parent_id=parent_id,
            children_ids=[],
            status=SMEStatus.SPAWNING,
            expertise=expertise,
            knowledge_base_path=str(sme_kb_path),
            creation_date=datetime.now(),
            last_summary_update=datetime.now(),
            specialization_triggers=["sub_domain_clustering", "advanced_complexity"]
        )
        
        # Initialize SME components
        await self._initialize_sme_components(sme_agent)
        
        # Migrate relevant knowledge from parent
        await self._migrate_knowledge_to_sme(sme_agent, parent_id)
        
        # Register SME
        self.agents[sme_id] = sme_agent
        self.domain_map[domain] = sme_id
        
        # Update parent's children
        if parent_id in self.agents:
            self.agents[parent_id].children_ids.append(sme_id)
        
        # Save SME to disk
        self._save_sme_agent(sme_agent)
        
        # Record specialization event
        self.specialization_history.append({
            "sme_id": sme_id,
            "domain": domain,
            "parent_id": parent_id,
            "created_at": datetime.now().isoformat(),
            "trigger_metrics": asdict(expertise)
        })
        
        sme_agent.status = SMEStatus.ACTIVE
        
        logger.info(f"Created SME agent {sme_id} for domain {domain}")
        
        return sme_id
    
    async def _initialize_sme_components(self, sme_agent: SMEAgent):
        """Initialize components for a new SME agent"""
        
        # Create specialized KB engine
        sme_agent.kb_engine = KBIntegrationEngine(kb_system_path=sme_agent.knowledge_base_path)
        
        # Create specialized ingestion system
        sme_agent.ingestion_system = KnowledgeIngestionSystem(kb_system_path=sme_agent.knowledge_base_path)
        
        # Share prompt classifier (can be global)
        sme_agent.prompt_classifier = self.prompt_classifier
    
    async def _migrate_knowledge_to_sme(self, sme_agent: SMEAgent, parent_id: str):
        """Migrate relevant knowledge from parent to new SME"""
        
        parent_agent = self.agents[parent_id]
        
        # Search parent's knowledge for domain-relevant content
        domain_knowledge = parent_agent.kb_engine.search_knowledge(
            query=sme_agent.domain,
            domain_filter=sme_agent.domain,
            max_results=50
        )
        
        # Create initial knowledge base for SME
        sme_kb_file = Path(sme_agent.knowledge_base_path) / f"{sme_agent.domain}_kb.json"
        
        initial_kb = {
            "metadata": {
                "sme_id": sme_agent.id,
                "domain": sme_agent.domain,
                "parent_id": parent_id,
                "created_at": datetime.now().isoformat(),
                "migrated_entries": len(domain_knowledge.results)
            },
            "knowledge_entries": [asdict(result) for result in domain_knowledge.results],
            "domain_patterns": [],
            "specialization_areas": []
        }
        
        with open(sme_kb_file, 'w') as f:
            json.dump(initial_kb, f, indent=2)
        
        logger.info(f"Migrated {len(domain_knowledge.results)} knowledge entries to SME {sme_agent.id}")
    
    def _save_sme_agent(self, sme_agent: SMEAgent):
        """Save SME agent to disk"""
        
        sme_file = self.base_kb_path / "sme_agents" / f"sme_{sme_agent.id}.json"
        
        # Prepare serializable data
        sme_data = {
            "id": sme_agent.id,
            "domain": sme_agent.domain,
            "parent_id": sme_agent.parent_id,
            "children_ids": sme_agent.children_ids,
            "status": sme_agent.status.value,
            "expertise": {
                "domain": sme_agent.expertise.domain,
                "knowledge_count": sme_agent.expertise.knowledge_count,
                "solution_success_rate": sme_agent.expertise.solution_success_rate,
                "query_volume": sme_agent.expertise.query_volume,
                "complexity_handled": sme_agent.expertise.complexity_handled,
                "last_activity": sme_agent.expertise.last_activity.isoformat(),
                "expertise_level": sme_agent.expertise.expertise_level.value,
                "confidence_score": sme_agent.expertise.confidence_score
            },
            "knowledge_base_path": sme_agent.knowledge_base_path,
            "creation_date": sme_agent.creation_date.isoformat(),
            "last_summary_update": sme_agent.last_summary_update.isoformat(),
            "specialization_triggers": sme_agent.specialization_triggers
        }
        
        with open(sme_file, 'w') as f:
            json.dump(sme_data, f, indent=2)
    
    def _update_sme_metrics(self, 
                          sme_agent: SMEAgent, 
                          classification: PromptClassification, 
                          response: Dict[str, Any]):
        """Update SME performance metrics"""
        
        # Update query volume
        sme_agent.expertise.query_volume += 1
        
        # Update complexity handled
        if classification.complexity.value >= 3:
            sme_agent.expertise.complexity_handled += 1
        
        # Update success rate (simplified - would need actual feedback)
        if response.get("confidence", 0) > 0.7:
            current_rate = sme_agent.expertise.solution_success_rate
            sme_agent.expertise.solution_success_rate = (current_rate * 0.9 + 0.8 * 0.1)
        
        # Update last activity
        sme_agent.expertise.last_activity = datetime.now()
        
        # Update expertise level based on performance
        self._assess_expertise_evolution(sme_agent)
    
    def _assess_expertise_evolution(self, sme_agent: SMEAgent):
        """Assess if SME should evolve to higher expertise level"""
        
        metrics = sme_agent.expertise
        
        # Criteria for expertise advancement
        if (metrics.query_volume >= 100 and 
            metrics.solution_success_rate >= 0.9 and 
            metrics.complexity_handled >= 20):
            
            if metrics.expertise_level == ExpertiseLevel.INTERMEDIATE:
                metrics.expertise_level = ExpertiseLevel.ADVANCED
                sme_agent.status = SMEStatus.SPECIALIZED
            elif metrics.expertise_level == ExpertiseLevel.ADVANCED:
                metrics.expertise_level = ExpertiseLevel.EXPERT
                sme_agent.status = SMEStatus.MATURE
    
    def _determine_search_depth(self, classification: PromptClassification) -> str:
        """Determine search depth based on classification"""
        
        if classification.requires_cross_repo:
            return "cross_repo"
        elif classification.complexity.value >= 3:
            return "deep"
        else:
            return "shallow"
    
    def _generate_sme_learning_recommendations(self, 
                                             sme_id: str, 
                                             parent_response: Any) -> List[str]:
        """Generate learning recommendations for SME based on escalation"""
        
        recommendations = [
            f"SME {sme_id} should study escalated knowledge patterns",
            "Consider expanding domain expertise to handle higher complexity",
            "Review parent agent's cross-repository analysis methods"
        ]
        
        return recommendations
    
    def get_sme_system_status(self) -> Dict[str, Any]:
        """Get comprehensive SME system status"""
        
        return {
            "total_agents": len(self.agents),
            "parent_agent": self.parent_agent_id,
            "specialized_agents": len([a for a in self.agents.values() if a.status == SMEStatus.SPECIALIZED]),
            "active_domains": list(self.domain_map.keys()),
            "query_routing_stats": self.query_routing_stats,
            "specialization_events": len(self.specialization_history),
            "system_metrics": {
                "specialization_threshold": self.specialization_threshold,
                "max_depth": self.max_sme_depth,
                "avg_sme_expertise": self._calculate_avg_expertise()
            }
        }
    
    def _calculate_avg_expertise(self) -> float:
        """Calculate average expertise level across SMEs"""
        
        if not self.agents:
            return 0.0
        
        total_expertise = sum(agent.expertise.expertise_level.value for agent in self.agents.values())
        return total_expertise / len(self.agents)
    
    async def consolidate_kb_systems(self):
        """Consolidate existing kb_system into unified kb_brain structure"""
        
        logger.info("Starting KB system consolidation")
        
        # Create unified structure
        unified_kb_dir = self.base_kb_path / "unified_kb"
        unified_kb_dir.mkdir(exist_ok=True)
        
        # Migrate existing knowledge bases
        original_kb_path = Path("/mnt/c/Users/misley/Documents/Projects/kb_system")
        
        if original_kb_path.exists():
            import shutil
            
            # Copy all KB files to unified location
            for kb_file in original_kb_path.glob("*.json"):
                target_file = unified_kb_dir / kb_file.name
                shutil.copy2(kb_file, target_file)
                logger.info(f"Migrated {kb_file.name} to unified KB")
            
            # Update all agent paths to use unified KB
            for agent in self.agents.values():
                if agent.kb_engine:
                    agent.kb_engine = KBIntegrationEngine(kb_system_path=str(unified_kb_dir))
                if agent.ingestion_system:
                    agent.ingestion_system = KnowledgeIngestionSystem(kb_system_path=str(unified_kb_dir))
        
        logger.info("KB system consolidation completed")


async def test_sme_system():
    """Test the SME Agent System"""
    
    sme_system = SMEAgentSystem()
    
    # Test query routing
    test_queries = [
        ("How do I fix SSL certificate issues in Dunes?", {"project": "dunes"}),
        ("Implement GPU acceleration for similarity search", {"domain": "programming"}),
        ("Analyze WHSA boundary detection methods", {"project": "research"}),
        ("General Python debugging help", {})
    ]
    
    for prompt, context in test_queries:
        print(f"\n=== Testing Query: {prompt} ===")
        
        agent_id, response = await sme_system.route_query(prompt, context)
        
        print(f"Routed to: {agent_id}")
        print(f"Specialized: {response.get('specialized', False)}")
        print(f"Domain: {response.get('domain', 'unknown')}")
        print(f"Response: {response.get('response', '')[:100]}...")
    
    # System status
    print(f"\n=== SME System Status ===")
    status = sme_system.get_sme_system_status()
    print(json.dumps(status, indent=2))
    
    # Test consolidation
    print(f"\n=== Testing KB Consolidation ===")
    await sme_system.consolidate_kb_systems()


if __name__ == "__main__":
    asyncio.run(test_sme_system())