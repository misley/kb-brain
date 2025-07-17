"""
KB Brain Intelligence System - Main Integration Module
Unified system integrating all intelligence components with SME agent hierarchy
"""

import asyncio
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

from .prompt_classifier import PromptClassifier
from .kb_integration_engine import KBIntegrationEngine
from .processing_orchestrator import ProcessingOrchestrator
from .ai_screen_manager import AIScreenManager
from .knowledge_ingestion import KnowledgeIngestionSystem
from .sme_agent_system import SMEAgentSystem
from .sme_communication_protocol import SMECommunicationProtocol
from .kb_consolidation_system import KBConsolidationSystem

logger = logging.getLogger(__name__)


class KBBrainIntelligence:
    """
    Unified KB Brain Intelligence System
    Integrates all intelligence components with SME agent hierarchy
    """
    
    def __init__(self, 
                 base_path: str = "/mnt/c/Users/misley/Documents/Projects/kb-brain",
                 auto_initialize: bool = True):
        """
        Initialize KB Brain Intelligence System
        
        Args:
            base_path: Base path for the system
            auto_initialize: Whether to auto-initialize all components
        """
        self.base_path = Path(base_path)
        self.initialized = False
        
        # Core components
        self.prompt_classifier: Optional[PromptClassifier] = None
        self.kb_engine: Optional[KBIntegrationEngine] = None
        self.orchestrator: Optional[ProcessingOrchestrator] = None
        self.screen_manager: Optional[AIScreenManager] = None
        self.ingestion_system: Optional[KnowledgeIngestionSystem] = None
        
        # SME system components
        self.sme_system: Optional[SMEAgentSystem] = None
        self.communication_protocol: Optional[SMECommunicationProtocol] = None
        self.consolidation_system: Optional[KBConsolidationSystem] = None
        
        # System state
        self.system_stats = {
            "total_queries_processed": 0,
            "sme_agents_active": 0,
            "knowledge_entries_ingested": 0,
            "avg_response_time": 0.0,
            "system_uptime": datetime.now()
        }
        
        if auto_initialize:
            asyncio.create_task(self.initialize())
        
        logger.info("KB Brain Intelligence System created")
    
    async def initialize(self) -> bool:
        """
        Initialize all system components
        
        Returns:
            True if initialization successful
        """
        try:
            logger.info("Initializing KB Brain Intelligence System")
            
            # Initialize core components
            await self._initialize_core_components()
            
            # Initialize SME system
            await self._initialize_sme_system()
            
            # Perform system consolidation if needed
            await self._check_and_consolidate()
            
            # Set up component integrations
            await self._setup_integrations()
            
            self.initialized = True
            logger.info("KB Brain Intelligence System initialized successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize KB Brain Intelligence System: {e}")
            return False
    
    async def _initialize_core_components(self):
        """Initialize core intelligence components"""
        
        # Prompt classifier
        self.prompt_classifier = PromptClassifier()
        
        # KB integration engine
        kb_path = str(self.base_path / "unified_kb")
        self.kb_engine = KBIntegrationEngine(kb_system_path=kb_path)
        
        # Knowledge ingestion system
        self.ingestion_system = KnowledgeIngestionSystem(kb_system_path=kb_path)
        
        # Screen manager for background processing
        self.screen_manager = AIScreenManager()
        
        # Processing orchestrator
        self.orchestrator = ProcessingOrchestrator(
            kb_engine=self.kb_engine,
            screen_manager=self.screen_manager
        )
        
        logger.info("Core components initialized")
    
    async def _initialize_sme_system(self):
        """Initialize SME agent system"""
        
        # SME agent system
        self.sme_system = SMEAgentSystem(
            base_kb_path=str(self.base_path),
            specialization_threshold=0.7
        )
        
        # Communication protocol
        self.communication_protocol = SMECommunicationProtocol()
        
        # KB consolidation system
        self.consolidation_system = KBConsolidationSystem(
            target_base_path=str(self.base_path)
        )
        
        logger.info("SME system components initialized")
    
    async def _check_and_consolidate(self):
        """Check if consolidation is needed and perform if necessary"""
        
        # Check if unified KB exists
        unified_kb_path = self.base_path / "unified_kb"
        
        if not unified_kb_path.exists():
            logger.info("Unified KB not found, performing consolidation")
            
            # Analyze existing knowledge
            analysis = self.consolidation_system.analyze_existing_knowledge()
            
            # Create consolidation plan
            plan = self.consolidation_system.create_consolidation_plan(analysis)
            
            # Execute consolidation
            result = self.consolidation_system.execute_consolidation(plan)
            
            if result.get("errors"):
                logger.warning(f"Consolidation completed with errors: {result['errors']}")
            else:
                logger.info("Knowledge base consolidation completed successfully")
    
    async def _setup_integrations(self):
        """Set up integrations between components"""
        
        # Register SME agents with communication protocol
        if self.sme_system and self.communication_protocol:
            for agent_id, agent in self.sme_system.agents.items():
                self.communication_protocol.register_agent(
                    agent_id=agent_id,
                    domain=agent.domain,
                    parent_id=agent.parent_id,
                    children_ids=agent.children_ids
                )
        
        # Set up ingestion feedback loop
        if self.ingestion_system and self.orchestrator:
            # Connect ingestion system to orchestrator for automatic knowledge capture
            pass  # This would be implemented based on specific integration patterns
        
        logger.info("Component integrations configured")
    
    async def process_query(self, 
                          query: str, 
                          context: Optional[Dict[str, Any]] = None,
                          use_sme_routing: bool = True) -> Dict[str, Any]:
        """
        Process a query through the intelligent system
        
        Args:
            query: The query to process
            context: Optional context information
            use_sme_routing: Whether to use SME agent routing
        
        Returns:
            Comprehensive response with all processing details
        """
        
        if not self.initialized:
            await self.initialize()
        
        start_time = datetime.now()
        
        try:
            # Route through SME system if enabled
            if use_sme_routing and self.sme_system:
                agent_id, sme_response = await self.sme_system.route_query(query, context)
                
                # Enhance with orchestrator processing if needed
                if sme_response.get("escalated_to_parent"):
                    orchestrator_result = await self.orchestrator.process_prompt(query, context)
                    sme_response["orchestrator_enhancement"] = orchestrator_result
                
                response = {
                    "query": query,
                    "processing_method": "sme_routing",
                    "assigned_agent": agent_id,
                    "response": sme_response,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            else:
                # Process through orchestrator
                orchestrator_result = await self.orchestrator.process_prompt(query, context)
                
                response = {
                    "query": query,
                    "processing_method": "orchestrator",
                    "response": orchestrator_result,
                    "processing_time": (datetime.now() - start_time).total_seconds()
                }
            
            # Ingest results for learning
            await self._ingest_query_results(query, response, context)
            
            # Update system stats
            self._update_system_stats(response)
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "query": query,
                "error": str(e),
                "processing_time": (datetime.now() - start_time).total_seconds()
            }
    
    async def create_sme_agent(self, 
                             domain: str, 
                             knowledge_threshold: int = 50) -> Optional[str]:
        """
        Manually create an SME agent for a domain
        
        Args:
            domain: Domain for the SME agent
            knowledge_threshold: Minimum knowledge entries for creation
        
        Returns:
            SME agent ID if created successfully
        """
        
        if not self.sme_system:
            logger.error("SME system not initialized")
            return None
        
        # Check if domain has enough knowledge
        domain_analysis = self.sme_system._analyze_knowledge_domains(
            self.sme_system.agents[self.sme_system.parent_agent_id]
        )
        
        if domain in domain_analysis:
            expertise = domain_analysis[domain]
            if expertise.knowledge_count >= knowledge_threshold:
                sme_id = await self.sme_system._create_sme_agent(
                    domain, expertise, self.sme_system.parent_agent_id
                )
                logger.info(f"Created SME agent {sme_id} for domain {domain}")
                return sme_id
        
        logger.warning(f"Insufficient knowledge for SME creation in domain {domain}")
        return None
    
    async def get_sme_summary_report(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """
        Get summary report from an SME agent
        
        Args:
            agent_id: SME agent ID
        
        Returns:
            Summary report data
        """
        
        if not self.sme_system or agent_id not in self.sme_system.agents:
            return None
        
        agent = self.sme_system.agents[agent_id]
        
        # Generate summary report
        summary = {
            "agent_id": agent_id,
            "domain": agent.domain,
            "expertise_level": agent.expertise.expertise_level.value,
            "knowledge_count": agent.expertise.knowledge_count,
            "query_volume": agent.expertise.query_volume,
            "success_rate": agent.expertise.solution_success_rate,
            "last_activity": agent.expertise.last_activity.isoformat(),
            "status": agent.status.value
        }
        
        return summary
    
    async def _ingest_query_results(self, 
                                  query: str, 
                                  response: Dict[str, Any], 
                                  context: Optional[Dict[str, Any]]):
        """Ingest query results for learning"""
        
        if not self.ingestion_system:
            return
        
        try:
            # Extract relevant information for ingestion
            processing_method = response.get("processing_method")
            response_data = response.get("response", {})
            
            # Ingest as user interaction
            self.ingestion_system.ingest_user_interaction(
                prompt=query,
                response=str(response_data),
                context=context or {},
                user_feedback={"helpful": True}  # Default positive feedback
            )
            
        except Exception as e:
            logger.error(f"Error ingesting query results: {e}")
    
    def _update_system_stats(self, response: Dict[str, Any]):
        """Update system statistics"""
        
        self.system_stats["total_queries_processed"] += 1
        
        # Update average response time
        processing_time = response.get("processing_time", 0)
        current_avg = self.system_stats["avg_response_time"]
        total_queries = self.system_stats["total_queries_processed"]
        
        self.system_stats["avg_response_time"] = (
            (current_avg * (total_queries - 1) + processing_time) / total_queries
        )
        
        # Update SME agent count
        if self.sme_system:
            self.system_stats["sme_agents_active"] = len(self.sme_system.agents)
    
    async def consolidate_knowledge_bases(self) -> Dict[str, Any]:
        """
        Manually trigger knowledge base consolidation
        
        Returns:
            Consolidation results
        """
        
        if not self.consolidation_system:
            return {"error": "Consolidation system not initialized"}
        
        # Analyze existing knowledge
        analysis = self.consolidation_system.analyze_existing_knowledge()
        
        # Create and execute consolidation plan
        plan = self.consolidation_system.create_consolidation_plan(analysis)
        result = self.consolidation_system.execute_consolidation(plan)
        
        return result
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        status = {
            "system_initialized": self.initialized,
            "system_stats": self.system_stats,
            "components": {
                "prompt_classifier": self.prompt_classifier is not None,
                "kb_engine": self.kb_engine is not None,
                "orchestrator": self.orchestrator is not None,
                "screen_manager": self.screen_manager is not None,
                "ingestion_system": self.ingestion_system is not None,
                "sme_system": self.sme_system is not None,
                "communication_protocol": self.communication_protocol is not None,
                "consolidation_system": self.consolidation_system is not None
            }
        }
        
        # Add component-specific status
        if self.sme_system:
            status["sme_system_status"] = self.sme_system.get_sme_system_status()
        
        if self.communication_protocol:
            status["communication_stats"] = self.communication_protocol.get_communication_stats()
        
        if self.kb_engine:
            status["kb_engine_status"] = self.kb_engine.get_system_status()
        
        return status
    
    async def shutdown(self):
        """Gracefully shutdown the system"""
        
        logger.info("Shutting down KB Brain Intelligence System")
        
        # Clean up screen sessions
        if self.screen_manager:
            self.screen_manager.cleanup_completed_sessions()
        
        # Process any pending ingestion
        if self.ingestion_system:
            self.ingestion_system.force_process_pending()
        
        # Save SME agent states
        if self.sme_system:
            for agent in self.sme_system.agents.values():
                self.sme_system._save_sme_agent(agent)
        
        self.initialized = False
        logger.info("System shutdown completed")


# Global system instance
_intelligence_system: Optional[KBBrainIntelligence] = None


def get_intelligence_system(base_path: Optional[str] = None) -> KBBrainIntelligence:
    """
    Get or create the global intelligence system instance
    
    Args:
        base_path: Base path for the system (only used on first creation)
    
    Returns:
        KBBrainIntelligence instance
    """
    global _intelligence_system
    
    if _intelligence_system is None:
        _intelligence_system = KBBrainIntelligence(
            base_path=base_path or "/mnt/c/Users/misley/Documents/Projects/kb-brain"
        )
    
    return _intelligence_system


async def test_intelligence_system():
    """Test the integrated intelligence system"""
    
    system = KBBrainIntelligence()
    await system.initialize()
    
    # Test queries
    test_queries = [
        "How do I fix SSL certificate issues in WSL?",
        "What's the status of the Dunes boundary analysis project?",
        "Implement GPU acceleration for similarity search",
        "Help me debug this Python import error"
    ]
    
    print("=== Testing KB Brain Intelligence System ===")
    
    for query in test_queries:
        print(f"\nProcessing: {query}")
        
        response = await system.process_query(query)
        
        print(f"Method: {response.get('processing_method')}")
        print(f"Time: {response.get('processing_time', 0):.3f}s")
        
        if "assigned_agent" in response:
            print(f"SME Agent: {response['assigned_agent']}")
    
    # System status
    print("\n=== System Status ===")
    status = system.get_system_status()
    print(f"Queries processed: {status['system_stats']['total_queries_processed']}")
    print(f"SME agents active: {status['system_stats']['sme_agents_active']}")
    print(f"Average response time: {status['system_stats']['avg_response_time']:.3f}s")
    
    await system.shutdown()


if __name__ == "__main__":
    asyncio.run(test_intelligence_system())