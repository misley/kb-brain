"""
KB Brain Intelligence System
Advanced intelligent prompt processing with SME agent hierarchy
"""

from .prompt_classifier import PromptClassifier, PromptClassification, ComplexityLevel, PromptIntent, Domain
from .kb_integration_engine import KBIntegrationEngine, KnowledgeResponse, SearchDepth
from .processing_orchestrator import ProcessingOrchestrator, ProcessingRoute, ProcessingResult
from .ai_screen_manager import AIScreenManager, TaskType, TaskStatus
from .knowledge_ingestion import KnowledgeIngestionSystem, KnowledgeEntry, IngestionSource
from .sme_agent_system import SMEAgentSystem, SMEAgent, ExpertiseLevel, DomainExpertise
from .sme_communication_protocol import SMECommunicationProtocol, SMEMessage, MessageType
from .kb_consolidation_system import KBConsolidationSystem, PartitionStrategy, ConsolidationPlan

__all__ = [
    # Core classification and processing
    'PromptClassifier', 'PromptClassification', 'ComplexityLevel', 'PromptIntent', 'Domain',
    'KBIntegrationEngine', 'KnowledgeResponse', 'SearchDepth',
    'ProcessingOrchestrator', 'ProcessingRoute', 'ProcessingResult',
    
    # Background processing
    'AIScreenManager', 'TaskType', 'TaskStatus',
    
    # Knowledge management
    'KnowledgeIngestionSystem', 'KnowledgeEntry', 'IngestionSource',
    'KBConsolidationSystem', 'PartitionStrategy', 'ConsolidationPlan',
    
    # SME agent system
    'SMEAgentSystem', 'SMEAgent', 'ExpertiseLevel', 'DomainExpertise',
    'SMECommunicationProtocol', 'SMEMessage', 'MessageType'
]

__version__ = "1.0.0"
__author__ = "misley"