"""
SME Agent Communication Protocol
Defines parent-child and peer communication patterns for SME agent hierarchy
"""

import json
import uuid
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in SME communication"""
    SUMMARY_REPORT = "summary_report"
    DETAIL_REQUEST = "detail_request"
    DETAIL_RESPONSE = "detail_response"
    ESCALATION = "escalation"
    CONSULTATION = "consultation"
    COLLABORATION = "collaboration"
    STATUS_UPDATE = "status_update"
    KNOWLEDGE_SHARE = "knowledge_share"
    LEARNING_UPDATE = "learning_update"


class MessagePriority(Enum):
    """Message priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    URGENT = 4
    CRITICAL = 5


class MessageStatus(Enum):
    """Message processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class SMEMessage:
    """Message between SME agents"""
    id: str
    type: MessageType
    priority: MessagePriority
    sender_id: str
    recipient_id: str
    subject: str
    content: Dict[str, Any]
    created_at: datetime
    expires_at: Optional[datetime] = None
    status: MessageStatus = MessageStatus.PENDING
    response_id: Optional[str] = None
    correlation_id: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class SMESummaryReport:
    """Periodic summary report from SME to parent"""
    agent_id: str
    domain: str
    reporting_period: Dict[str, str]  # start_date, end_date
    activity_metrics: Dict[str, Any]
    key_insights: List[str]
    problem_areas: List[str]
    success_stories: List[str]
    knowledge_growth: Dict[str, Any]
    resource_requests: List[str]
    collaboration_opportunities: List[str]
    confidence_level: float


@dataclass
class SMEDetailRequest:
    """Request for detailed information from parent to child"""
    query: str
    context: Dict[str, Any]
    requested_detail_level: str  # summary, detailed, comprehensive
    requested_sections: List[str]  # specific sections of knowledge
    urgency: MessagePriority
    deadline: Optional[datetime] = None


@dataclass
class SMEConsultation:
    """Consultation request between peer SMEs"""
    requesting_agent: str
    consulting_domain: str
    query: str
    current_approach: str
    seeking_perspective: List[str]  # what kind of input is needed
    collaboration_type: str  # advice, joint_analysis, knowledge_share


class SMECommunicationProtocol:
    """Communication protocol for SME agent hierarchy"""
    
    def __init__(self, message_timeout: int = 300):  # 5 minutes default timeout
        """
        Initialize SME Communication Protocol
        
        Args:
            message_timeout: Default message timeout in seconds
        """
        self.message_timeout = message_timeout
        
        # Message tracking
        self.active_messages: Dict[str, SMEMessage] = {}
        self.message_history: List[SMEMessage] = []
        self.agent_registry: Dict[str, Dict[str, Any]] = {}
        
        # Communication patterns
        self.message_handlers: Dict[MessageType, Callable] = {
            MessageType.SUMMARY_REPORT: self._handle_summary_report,
            MessageType.DETAIL_REQUEST: self._handle_detail_request,
            MessageType.ESCALATION: self._handle_escalation,
            MessageType.CONSULTATION: self._handle_consultation,
            MessageType.COLLABORATION: self._handle_collaboration,
            MessageType.KNOWLEDGE_SHARE: self._handle_knowledge_share
        }
        
        # Performance metrics
        self.communication_stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "avg_response_time": 0.0,
            "success_rate": 0.0,
            "collaboration_count": 0
        }
        
        logger.info("SME Communication Protocol initialized")
    
    def register_agent(self, 
                      agent_id: str, 
                      domain: str, 
                      parent_id: Optional[str] = None,
                      children_ids: List[str] = None,
                      capabilities: List[str] = None) -> bool:
        """
        Register an SME agent in the communication protocol
        
        Args:
            agent_id: Unique agent identifier
            domain: Agent's domain of expertise
            parent_id: Parent agent ID (None for root agent)
            children_ids: List of child agent IDs
            capabilities: List of agent capabilities
        
        Returns:
            True if registration successful
        """
        
        if children_ids is None:
            children_ids = []
        if capabilities is None:
            capabilities = []
        
        self.agent_registry[agent_id] = {
            "domain": domain,
            "parent_id": parent_id,
            "children_ids": children_ids,
            "capabilities": capabilities,
            "registered_at": datetime.now(),
            "last_activity": datetime.now(),
            "message_count": 0,
            "status": "active"
        }
        
        logger.info(f"Registered SME agent {agent_id} for domain {domain}")
        
        return True
    
    async def send_summary_report(self, 
                                agent_id: str, 
                                summary_report: SMESummaryReport) -> str:
        """
        Send summary report from SME to parent
        
        Args:
            agent_id: Sending agent ID
            summary_report: Summary report content
        
        Returns:
            Message ID
        """
        
        # Get parent ID
        agent_info = self.agent_registry.get(agent_id)
        if not agent_info or not agent_info.get("parent_id"):
            raise ValueError(f"Agent {agent_id} has no parent to report to")
        
        parent_id = agent_info["parent_id"]
        
        # Create message
        message = SMEMessage(
            id=self._generate_message_id(),
            type=MessageType.SUMMARY_REPORT,
            priority=MessagePriority.NORMAL,
            sender_id=agent_id,
            recipient_id=parent_id,
            subject=f"Summary Report: {summary_report.domain}",
            content=asdict(summary_report),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=24)  # Reports expire in 24 hours
        )
        
        # Send message
        return await self._send_message(message)
    
    async def request_details(self, 
                            parent_id: str, 
                            child_id: str, 
                            detail_request: SMEDetailRequest) -> str:
        """
        Request detailed information from parent to child SME
        
        Args:
            parent_id: Requesting parent agent ID
            child_id: Target child agent ID
            detail_request: Detail request content
        
        Returns:
            Message ID
        """
        
        # Validate parent-child relationship
        if not self._validate_parent_child_relationship(parent_id, child_id):
            raise ValueError(f"Invalid parent-child relationship: {parent_id} -> {child_id}")
        
        # Create message
        message = SMEMessage(
            id=self._generate_message_id(),
            type=MessageType.DETAIL_REQUEST,
            priority=detail_request.urgency,
            sender_id=parent_id,
            recipient_id=child_id,
            subject=f"Detail Request: {detail_request.query[:50]}...",
            content=asdict(detail_request),
            created_at=datetime.now(),
            expires_at=detail_request.deadline or (datetime.now() + timedelta(hours=2))
        )
        
        # Send message
        return await self._send_message(message)
    
    async def escalate_to_parent(self, 
                               agent_id: str, 
                               query: str, 
                               context: Dict[str, Any],
                               escalation_reason: str) -> str:
        """
        Escalate complex query to parent agent
        
        Args:
            agent_id: Escalating agent ID
            query: Original query
            context: Query context
            escalation_reason: Reason for escalation
        
        Returns:
            Message ID
        """
        
        # Get parent ID
        agent_info = self.agent_registry.get(agent_id)
        if not agent_info or not agent_info.get("parent_id"):
            raise ValueError(f"Agent {agent_id} has no parent to escalate to")
        
        parent_id = agent_info["parent_id"]
        
        escalation_content = {
            "original_query": query,
            "context": context,
            "escalation_reason": escalation_reason,
            "agent_domain": agent_info["domain"],
            "attempted_approaches": context.get("attempted_approaches", []),
            "complexity_level": context.get("complexity", "unknown")
        }
        
        # Create escalation message
        message = SMEMessage(
            id=self._generate_message_id(),
            type=MessageType.ESCALATION,
            priority=MessagePriority.HIGH,
            sender_id=agent_id,
            recipient_id=parent_id,
            subject=f"Escalation: {query[:50]}...",
            content=escalation_content,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=1)  # Escalations are urgent
        )
        
        return await self._send_message(message)
    
    async def request_consultation(self, 
                                 requesting_agent: str, 
                                 consulting_agent: str, 
                                 consultation: SMEConsultation) -> str:
        """
        Request consultation between peer SMEs
        
        Args:
            requesting_agent: Agent requesting consultation
            consulting_agent: Agent being consulted
            consultation: Consultation details
        
        Returns:
            Message ID
        """
        
        # Validate both agents exist
        if not all(agent_id in self.agent_registry for agent_id in [requesting_agent, consulting_agent]):
            raise ValueError("One or both agents not registered")
        
        # Create consultation message
        message = SMEMessage(
            id=self._generate_message_id(),
            type=MessageType.CONSULTATION,
            priority=MessagePriority.NORMAL,
            sender_id=requesting_agent,
            recipient_id=consulting_agent,
            subject=f"Consultation: {consultation.consulting_domain}",
            content=asdict(consultation),
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(hours=4)
        )
        
        return await self._send_message(message)
    
    async def initiate_collaboration(self, 
                                   initiating_agent: str, 
                                   collaborating_agents: List[str], 
                                   collaboration_topic: str,
                                   collaboration_goal: str) -> str:
        """
        Initiate collaboration between multiple SMEs
        
        Args:
            initiating_agent: Agent initiating collaboration
            collaborating_agents: List of agents to collaborate with
            collaboration_topic: Topic of collaboration
            collaboration_goal: Goal of collaboration
        
        Returns:
            Collaboration session ID
        """
        
        collaboration_id = f"collab_{str(uuid.uuid4())[:8]}"
        
        collaboration_content = {
            "collaboration_id": collaboration_id,
            "topic": collaboration_topic,
            "goal": collaboration_goal,
            "participants": [initiating_agent] + collaborating_agents,
            "initiated_by": initiating_agent,
            "status": "initiating"
        }
        
        # Send collaboration invitations to all participants
        message_ids = []
        
        for agent_id in collaborating_agents:
            message = SMEMessage(
                id=self._generate_message_id(),
                type=MessageType.COLLABORATION,
                priority=MessagePriority.NORMAL,
                sender_id=initiating_agent,
                recipient_id=agent_id,
                subject=f"Collaboration Invitation: {collaboration_topic}",
                content=collaboration_content,
                created_at=datetime.now(),
                expires_at=datetime.now() + timedelta(hours=8),
                correlation_id=collaboration_id
            )
            
            message_id = await self._send_message(message)
            message_ids.append(message_id)
        
        self.communication_stats["collaboration_count"] += 1
        
        return collaboration_id
    
    async def share_knowledge(self, 
                            sender_id: str, 
                            recipient_id: str, 
                            knowledge_type: str,
                            knowledge_content: Dict[str, Any]) -> str:
        """
        Share knowledge between SME agents
        
        Args:
            sender_id: Agent sharing knowledge
            recipient_id: Agent receiving knowledge
            knowledge_type: Type of knowledge (solution, pattern, insight)
            knowledge_content: Knowledge content
        
        Returns:
            Message ID
        """
        
        share_content = {
            "knowledge_type": knowledge_type,
            "content": knowledge_content,
            "sender_domain": self.agent_registry[sender_id]["domain"],
            "relevance_score": knowledge_content.get("relevance", 0.8),
            "confidence_score": knowledge_content.get("confidence", 0.7)
        }
        
        message = SMEMessage(
            id=self._generate_message_id(),
            type=MessageType.KNOWLEDGE_SHARE,
            priority=MessagePriority.LOW,
            sender_id=sender_id,
            recipient_id=recipient_id,
            subject=f"Knowledge Share: {knowledge_type}",
            content=share_content,
            created_at=datetime.now(),
            expires_at=datetime.now() + timedelta(days=7)  # Knowledge shares don't expire quickly
        )
        
        return await self._send_message(message)
    
    async def _send_message(self, message: SMEMessage) -> str:
        """Internal method to send a message"""
        
        # Add to active messages
        self.active_messages[message.id] = message
        
        # Update sender statistics
        if message.sender_id in self.agent_registry:
            self.agent_registry[message.sender_id]["message_count"] += 1
            self.agent_registry[message.sender_id]["last_activity"] = datetime.now()
        
        # Update communication stats
        self.communication_stats["messages_sent"] += 1
        
        # Trigger message processing
        asyncio.create_task(self._process_message(message))
        
        logger.info(f"Sent {message.type.value} message {message.id} from {message.sender_id} to {message.recipient_id}")
        
        return message.id
    
    async def _process_message(self, message: SMEMessage):
        """Process a message asynchronously"""
        
        try:
            message.status = MessageStatus.PROCESSING
            
            # Get message handler
            handler = self.message_handlers.get(message.type)
            
            if handler:
                response = await handler(message)
                
                if response:
                    # Send response message
                    response_message = SMEMessage(
                        id=self._generate_message_id(),
                        type=MessageType.DETAIL_RESPONSE,
                        priority=message.priority,
                        sender_id=message.recipient_id,
                        recipient_id=message.sender_id,
                        subject=f"Re: {message.subject}",
                        content=response,
                        created_at=datetime.now(),
                        correlation_id=message.id
                    )
                    
                    await self._send_message(response_message)
                    message.response_id = response_message.id
            
            message.status = MessageStatus.COMPLETED
            
            # Update communication stats
            self.communication_stats["messages_received"] += 1
            self._update_response_time_stats(message)
            
        except Exception as e:
            logger.error(f"Error processing message {message.id}: {e}")
            message.status = MessageStatus.FAILED
        
        finally:
            # Move to history
            self._archive_message(message)
    
    async def _handle_summary_report(self, message: SMEMessage) -> Optional[Dict[str, Any]]:
        """Handle summary report from child SME"""
        
        summary_data = message.content
        sender_id = message.sender_id
        
        logger.info(f"Received summary report from {sender_id} for domain {summary_data.get('domain')}")
        
        # Process summary report
        # Parent can analyze trends, identify issues, provide guidance
        
        # Generate acknowledgment response
        response = {
            "acknowledgment": "Summary report received and processed",
            "feedback": self._generate_summary_feedback(summary_data),
            "recommendations": self._generate_parent_recommendations(summary_data),
            "next_reporting_period": (datetime.now() + timedelta(days=7)).isoformat()
        }
        
        return response
    
    async def _handle_detail_request(self, message: SMEMessage) -> Optional[Dict[str, Any]]:
        """Handle detail request from parent"""
        
        request_data = message.content
        sender_id = message.sender_id  # parent
        recipient_id = message.recipient_id  # child SME
        
        logger.info(f"Processing detail request from parent {sender_id}")
        
        # Extract requested details based on request
        query = request_data.get("query", "")
        detail_level = request_data.get("requested_detail_level", "summary")
        sections = request_data.get("requested_sections", [])
        
        # Generate detailed response
        response = {
            "query": query,
            "detail_level": detail_level,
            "detailed_analysis": self._generate_detailed_analysis(query, detail_level),
            "supporting_evidence": self._gather_supporting_evidence(query, sections),
            "confidence_assessment": self._assess_response_confidence(query),
            "additional_resources": self._suggest_additional_resources(query)
        }
        
        return response
    
    async def _handle_escalation(self, message: SMEMessage) -> Optional[Dict[str, Any]]:
        """Handle escalation from child SME"""
        
        escalation_data = message.content
        child_id = message.sender_id
        
        logger.info(f"Handling escalation from child SME {child_id}")
        
        # Analyze escalation
        query = escalation_data.get("original_query", "")
        reason = escalation_data.get("escalation_reason", "")
        context = escalation_data.get("context", {})
        
        # Generate escalation response
        response = {
            "escalation_accepted": True,
            "parent_analysis": self._perform_parent_analysis(query, context),
            "guidance": self._provide_child_guidance(child_id, reason),
            "resources": self._allocate_additional_resources(child_id, query),
            "follow_up_required": True
        }
        
        return response
    
    async def _handle_consultation(self, message: SMEMessage) -> Optional[Dict[str, Any]]:
        """Handle consultation request between peers"""
        
        consultation_data = message.content
        requesting_agent = message.sender_id
        
        logger.info(f"Handling consultation request from {requesting_agent}")
        
        # Generate consultation response
        response = {
            "consultation_accepted": True,
            "domain_perspective": self._provide_domain_perspective(consultation_data),
            "recommended_approaches": self._suggest_approaches(consultation_data),
            "potential_collaborations": self._identify_collaboration_opportunities(consultation_data),
            "follow_up_suggestions": self._suggest_follow_up_actions(consultation_data)
        }
        
        return response
    
    async def _handle_collaboration(self, message: SMEMessage) -> Optional[Dict[str, Any]]:
        """Handle collaboration invitation"""
        
        collaboration_data = message.content
        collaboration_id = collaboration_data.get("collaboration_id")
        
        logger.info(f"Handling collaboration invitation {collaboration_id}")
        
        # Accept/decline collaboration
        response = {
            "collaboration_id": collaboration_id,
            "participation_status": "accepted",
            "available_resources": self._assess_collaboration_resources(),
            "contribution_areas": self._identify_contribution_areas(collaboration_data),
            "availability_window": self._determine_availability_window()
        }
        
        return response
    
    async def _handle_knowledge_share(self, message: SMEMessage) -> Optional[Dict[str, Any]]:
        """Handle knowledge sharing between agents"""
        
        knowledge_data = message.content
        sender_id = message.sender_id
        
        logger.info(f"Handling knowledge share from {sender_id}")
        
        # Process shared knowledge
        knowledge_type = knowledge_data.get("knowledge_type")
        content = knowledge_data.get("content", {})
        
        # Generate acknowledgment
        response = {
            "knowledge_received": True,
            "relevance_assessment": self._assess_knowledge_relevance(content),
            "integration_status": self._integrate_shared_knowledge(content),
            "reciprocal_knowledge": self._identify_reciprocal_knowledge(sender_id)
        }
        
        return response
    
    def _validate_parent_child_relationship(self, parent_id: str, child_id: str) -> bool:
        """Validate parent-child relationship"""
        
        if parent_id not in self.agent_registry or child_id not in self.agent_registry:
            return False
        
        parent_info = self.agent_registry[parent_id]
        child_info = self.agent_registry[child_id]
        
        return (child_id in parent_info.get("children_ids", []) and 
                child_info.get("parent_id") == parent_id)
    
    def _generate_message_id(self) -> str:
        """Generate unique message ID"""
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        random_part = str(uuid.uuid4())[:8]
        return f"msg_{timestamp}_{random_part}"
    
    def _archive_message(self, message: SMEMessage):
        """Archive completed message"""
        
        # Remove from active messages
        if message.id in self.active_messages:
            del self.active_messages[message.id]
        
        # Add to history
        self.message_history.append(message)
        
        # Keep history manageable
        if len(self.message_history) > 1000:
            self.message_history = self.message_history[-500:]
    
    def _update_response_time_stats(self, message: SMEMessage):
        """Update response time statistics"""
        
        if message.status == MessageStatus.COMPLETED:
            processing_time = (datetime.now() - message.created_at).total_seconds()
            
            current_avg = self.communication_stats["avg_response_time"]
            message_count = self.communication_stats["messages_received"]
            
            # Calculate running average
            self.communication_stats["avg_response_time"] = (
                (current_avg * (message_count - 1) + processing_time) / message_count
            )
    
    # Helper methods for message processing (simplified implementations)
    
    def _generate_summary_feedback(self, summary_data: Dict[str, Any]) -> str:
        """Generate feedback on summary report"""
        return "Summary report shows good progress in domain expertise"
    
    def _generate_parent_recommendations(self, summary_data: Dict[str, Any]) -> List[str]:
        """Generate recommendations from parent to child"""
        return ["Continue current approach", "Consider expanding to related domains"]
    
    def _generate_detailed_analysis(self, query: str, detail_level: str) -> str:
        """Generate detailed analysis for parent request"""
        return f"Detailed analysis for: {query} at {detail_level} level"
    
    def _gather_supporting_evidence(self, query: str, sections: List[str]) -> Dict[str, Any]:
        """Gather supporting evidence for response"""
        return {"evidence": f"Supporting evidence for {query}"}
    
    def _assess_response_confidence(self, query: str) -> float:
        """Assess confidence in response"""
        return 0.8
    
    def _suggest_additional_resources(self, query: str) -> List[str]:
        """Suggest additional resources"""
        return ["Additional resource suggestions"]
    
    def _perform_parent_analysis(self, query: str, context: Dict[str, Any]) -> str:
        """Perform parent-level analysis of escalated query"""
        return f"Parent analysis of escalated query: {query}"
    
    def _provide_child_guidance(self, child_id: str, reason: str) -> str:
        """Provide guidance to child SME"""
        return f"Guidance for {child_id} regarding {reason}"
    
    def _allocate_additional_resources(self, child_id: str, query: str) -> Dict[str, Any]:
        """Allocate additional resources for escalated query"""
        return {"resources": "Additional computational resources allocated"}
    
    def _provide_domain_perspective(self, consultation_data: Dict[str, Any]) -> str:
        """Provide domain perspective for consultation"""
        return "Domain-specific perspective on the consultation request"
    
    def _suggest_approaches(self, consultation_data: Dict[str, Any]) -> List[str]:
        """Suggest approaches for consultation"""
        return ["Approach 1", "Approach 2", "Approach 3"]
    
    def _identify_collaboration_opportunities(self, consultation_data: Dict[str, Any]) -> List[str]:
        """Identify collaboration opportunities"""
        return ["Joint research opportunity", "Knowledge sharing session"]
    
    def _suggest_follow_up_actions(self, consultation_data: Dict[str, Any]) -> List[str]:
        """Suggest follow-up actions"""
        return ["Schedule follow-up meeting", "Share relevant resources"]
    
    def _assess_collaboration_resources(self) -> Dict[str, Any]:
        """Assess available resources for collaboration"""
        return {"available": True, "capacity": "high"}
    
    def _identify_contribution_areas(self, collaboration_data: Dict[str, Any]) -> List[str]:
        """Identify areas where agent can contribute"""
        return ["Domain expertise", "Analytical capabilities"]
    
    def _determine_availability_window(self) -> Dict[str, str]:
        """Determine availability window for collaboration"""
        return {"start": datetime.now().isoformat(), "duration": "1 week"}
    
    def _assess_knowledge_relevance(self, content: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge"""
        return 0.7
    
    def _integrate_shared_knowledge(self, content: Dict[str, Any]) -> str:
        """Integrate shared knowledge into local knowledge base"""
        return "Knowledge integrated successfully"
    
    def _identify_reciprocal_knowledge(self, sender_id: str) -> Dict[str, Any]:
        """Identify knowledge that could be shared in return"""
        return {"reciprocal": "Relevant knowledge identified for sharing"}
    
    def get_communication_stats(self) -> Dict[str, Any]:
        """Get communication statistics"""
        
        # Calculate success rate
        completed_messages = len([m for m in self.message_history if m.status == MessageStatus.COMPLETED])
        total_messages = len(self.message_history)
        success_rate = completed_messages / total_messages if total_messages > 0 else 0.0
        
        self.communication_stats["success_rate"] = success_rate
        
        return {
            "communication_stats": self.communication_stats,
            "active_messages": len(self.active_messages),
            "message_history_size": len(self.message_history),
            "registered_agents": len(self.agent_registry),
            "agent_activity": {
                agent_id: info["message_count"] 
                for agent_id, info in self.agent_registry.items()
            }
        }


async def test_communication_protocol():
    """Test the SME communication protocol"""
    
    protocol = SMECommunicationProtocol()
    
    # Register test agents
    protocol.register_agent("parent_001", "general", None, ["child_001", "child_002"])
    protocol.register_agent("child_001", "geospatial", "parent_001")
    protocol.register_agent("child_002", "technical", "parent_001")
    
    # Test summary report
    summary = SMESummaryReport(
        agent_id="child_001",
        domain="geospatial",
        reporting_period={"start_date": "2025-01-01", "end_date": "2025-01-07"},
        activity_metrics={"queries_handled": 25, "success_rate": 0.85},
        key_insights=["Improved boundary detection accuracy"],
        problem_areas=["Need more satellite data"],
        success_stories=["Solved WHSA boundary issue"],
        knowledge_growth={"new_patterns": 3},
        resource_requests=["More computational power"],
        collaboration_opportunities=["Work with data science team"],
        confidence_level=0.8
    )
    
    print("=== Testing Summary Report ===")
    message_id = await protocol.send_summary_report("child_001", summary)
    print(f"Sent summary report: {message_id}")
    
    # Test detail request
    detail_request = SMEDetailRequest(
        query="How to improve boundary detection accuracy?",
        context={"project": "WHSA", "urgency": "high"},
        requested_detail_level="comprehensive",
        requested_sections=["methodology", "validation"],
        urgency=MessagePriority.HIGH
    )
    
    print("\n=== Testing Detail Request ===")
    detail_msg_id = await protocol.request_details("parent_001", "child_001", detail_request)
    print(f"Sent detail request: {detail_msg_id}")
    
    # Test consultation
    consultation = SMEConsultation(
        requesting_agent="child_001",
        consulting_domain="technical",
        query="SSL certificate issues affecting data download",
        current_approach="Using corporate certificates",
        seeking_perspective=["alternative approaches", "security implications"],
        collaboration_type="advice"
    )
    
    print("\n=== Testing Consultation ===")
    consult_msg_id = await protocol.request_consultation("child_001", "child_002", consultation)
    print(f"Sent consultation request: {consult_msg_id}")
    
    # Wait for message processing
    await asyncio.sleep(2)
    
    # Get communication stats
    print("\n=== Communication Statistics ===")
    stats = protocol.get_communication_stats()
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_communication_protocol())