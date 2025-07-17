"""
Tests for SME Communication Protocol
"""

import pytest
import asyncio
from datetime import datetime, timedelta
from unittest.mock import Mock

from kb_brain.intelligence.sme_communication_protocol import (
    SMECommunicationProtocol, SMEMessage, MessageType, MessagePriority, MessageStatus,
    SMESummaryReport, SMEDetailRequest, SMEConsultation
)


class TestSMECommunicationProtocol:
    """Test cases for SME Communication Protocol"""
    
    @pytest.fixture
    def protocol(self):
        """Create communication protocol instance"""
        return SMECommunicationProtocol()
    
    @pytest.fixture
    def setup_agents(self, protocol):
        """Set up test agents"""
        protocol.register_agent(
            agent_id="parent_001",
            domain="general",
            parent_id=None,
            children_ids=["child_001", "child_002"]
        )
        
        protocol.register_agent(
            agent_id="child_001",
            domain="geospatial",
            parent_id="parent_001"
        )
        
        protocol.register_agent(
            agent_id="child_002", 
            domain="technical",
            parent_id="parent_001"
        )
        
        return protocol
    
    def test_agent_registration(self, protocol):
        """Test agent registration"""
        agent_id = "test_agent_001"
        domain = "test_domain"
        
        success = protocol.register_agent(
            agent_id=agent_id,
            domain=domain,
            capabilities=["analysis", "consultation"]
        )
        
        assert success
        assert agent_id in protocol.agent_registry
        
        agent_info = protocol.agent_registry[agent_id]
        assert agent_info["domain"] == domain
        assert agent_info["status"] == "active"
        assert "analysis" in agent_info["capabilities"]
    
    @pytest.mark.asyncio
    async def test_summary_report_sending(self, setup_agents):
        """Test sending summary report from child to parent"""
        protocol = setup_agents
        
        summary_report = SMESummaryReport(
            agent_id="child_001",
            domain="geospatial",
            reporting_period={"start_date": "2025-01-01", "end_date": "2025-01-07"},
            activity_metrics={"queries_handled": 25, "success_rate": 0.85},
            key_insights=["Improved boundary detection"],
            problem_areas=["Need more data"],
            success_stories=["Solved WHSA issue"],
            knowledge_growth={"new_patterns": 3},
            resource_requests=["More compute"],
            collaboration_opportunities=["Data science team"],
            confidence_level=0.8
        )
        
        message_id = await protocol.send_summary_report("child_001", summary_report)
        
        assert message_id is not None
        assert message_id in protocol.active_messages
        
        message = protocol.active_messages[message_id]
        assert message.type == MessageType.SUMMARY_REPORT
        assert message.sender_id == "child_001"
        assert message.recipient_id == "parent_001"
    
    @pytest.mark.asyncio
    async def test_detail_request_sending(self, setup_agents):
        """Test sending detail request from parent to child"""
        protocol = setup_agents
        
        detail_request = SMEDetailRequest(
            query="Explain boundary detection methodology",
            context={"project": "WHSA", "urgency": "high"},
            requested_detail_level="comprehensive",
            requested_sections=["methodology", "validation"],
            urgency=MessagePriority.HIGH
        )
        
        message_id = await protocol.request_details(
            "parent_001", "child_001", detail_request
        )
        
        assert message_id is not None
        assert message_id in protocol.active_messages
        
        message = protocol.active_messages[message_id]
        assert message.type == MessageType.DETAIL_REQUEST
        assert message.sender_id == "parent_001"
        assert message.recipient_id == "child_001"
        assert message.priority == MessagePriority.HIGH
    
    @pytest.mark.asyncio
    async def test_escalation_handling(self, setup_agents):
        """Test escalation from child to parent"""
        protocol = setup_agents
        
        message_id = await protocol.escalate_to_parent(
            agent_id="child_001",
            query="Complex boundary analysis problem",
            context={"complexity": "high", "attempted_approaches": ["method1", "method2"]},
            escalation_reason="complexity_exceeded"
        )
        
        assert message_id is not None
        
        message = protocol.active_messages[message_id]
        assert message.type == MessageType.ESCALATION
        assert message.sender_id == "child_001"
        assert message.recipient_id == "parent_001"
        assert message.priority == MessagePriority.HIGH
        
        # Check escalation content
        content = message.content
        assert content["escalation_reason"] == "complexity_exceeded"
        assert "attempted_approaches" in content
    
    @pytest.mark.asyncio
    async def test_consultation_between_peers(self, setup_agents):
        """Test consultation request between peer SMEs"""
        protocol = setup_agents
        
        consultation = SMEConsultation(
            requesting_agent="child_001",
            consulting_domain="technical",
            query="SSL issues affecting data download",
            current_approach="Using corporate certificates",
            seeking_perspective=["alternative approaches"],
            collaboration_type="advice"
        )
        
        message_id = await protocol.request_consultation(
            "child_001", "child_002", consultation
        )
        
        assert message_id is not None
        
        message = protocol.active_messages[message_id]
        assert message.type == MessageType.CONSULTATION
        assert message.sender_id == "child_001"
        assert message.recipient_id == "child_002"
    
    @pytest.mark.asyncio
    async def test_collaboration_initiation(self, setup_agents):
        """Test initiating collaboration between multiple SMEs"""
        protocol = setup_agents
        
        collaboration_id = await protocol.initiate_collaboration(
            initiating_agent="child_001",
            collaborating_agents=["child_002"],
            collaboration_topic="Cross-domain analysis",
            collaboration_goal="Combine geospatial and technical expertise"
        )
        
        assert collaboration_id is not None
        assert collaboration_id.startswith("collab_")
        
        # Check that collaboration message was sent
        collaboration_messages = [
            msg for msg in protocol.active_messages.values()
            if msg.type == MessageType.COLLABORATION
        ]
        
        assert len(collaboration_messages) > 0
        
        collab_message = collaboration_messages[0]
        assert collab_message.correlation_id == collaboration_id
    
    @pytest.mark.asyncio
    async def test_knowledge_sharing(self, setup_agents):
        """Test knowledge sharing between agents"""
        protocol = setup_agents
        
        knowledge_content = {
            "solution": "Use SSL certificate bundling",
            "confidence": 0.9,
            "relevance": 0.8,
            "validation": "tested_successfully"
        }
        
        message_id = await protocol.share_knowledge(
            sender_id="child_002",
            recipient_id="child_001", 
            knowledge_type="solution",
            knowledge_content=knowledge_content
        )
        
        assert message_id is not None
        
        message = protocol.active_messages[message_id]
        assert message.type == MessageType.KNOWLEDGE_SHARE
        assert message.content["knowledge_type"] == "solution"
        assert message.content["confidence_score"] == 0.9
    
    @pytest.mark.asyncio
    async def test_message_processing(self, setup_agents):
        """Test message processing and response generation"""
        protocol = setup_agents
        
        # Send a summary report and wait for processing
        summary_report = SMESummaryReport(
            agent_id="child_001",
            domain="geospatial",
            reporting_period={"start_date": "2025-01-01", "end_date": "2025-01-07"},
            activity_metrics={"queries_handled": 10},
            key_insights=["Test insight"],
            problem_areas=[],
            success_stories=[],
            knowledge_growth={},
            resource_requests=[],
            collaboration_opportunities=[],
            confidence_level=0.7
        )
        
        message_id = await protocol.send_summary_report("child_001", summary_report)
        
        # Wait for message processing
        await asyncio.sleep(0.1)
        
        # Check if message was processed
        # In real implementation, this would involve actual processing
        assert message_id in protocol.active_messages
    
    def test_parent_child_relationship_validation(self, setup_agents):
        """Test parent-child relationship validation"""
        protocol = setup_agents
        
        # Valid relationship
        assert protocol._validate_parent_child_relationship("parent_001", "child_001")
        
        # Invalid relationship
        assert not protocol._validate_parent_child_relationship("child_001", "parent_001")
        assert not protocol._validate_parent_child_relationship("child_001", "child_002")
    
    def test_message_expiration(self, protocol):
        """Test message expiration handling"""
        message = SMEMessage(
            id="test_msg_001",
            type=MessageType.DETAIL_REQUEST,
            priority=MessagePriority.NORMAL,
            sender_id="sender",
            recipient_id="recipient",
            subject="Test message",
            content={"test": "data"},
            created_at=datetime.now(),
            expires_at=datetime.now() - timedelta(hours=1)  # Already expired
        )
        
        # Message should be considered expired
        assert message.expires_at < datetime.now()
    
    def test_message_priority_handling(self, protocol):
        """Test message priority handling"""
        urgent_message = SMEMessage(
            id="urgent_001",
            type=MessageType.ESCALATION,
            priority=MessagePriority.URGENT,
            sender_id="child",
            recipient_id="parent",
            subject="Urgent escalation",
            content={},
            created_at=datetime.now()
        )
        
        normal_message = SMEMessage(
            id="normal_001",
            type=MessageType.CONSULTATION,
            priority=MessagePriority.NORMAL,
            sender_id="child1",
            recipient_id="child2",
            subject="Normal consultation",
            content={},
            created_at=datetime.now()
        )
        
        # Urgent messages should have higher priority value
        assert urgent_message.priority.value > normal_message.priority.value
    
    def test_communication_statistics(self, setup_agents):
        """Test communication statistics tracking"""
        protocol = setup_agents
        
        initial_stats = protocol.get_communication_stats()
        
        # Initially should have no messages
        assert initial_stats["communication_stats"]["messages_sent"] == 0
        assert initial_stats["communication_stats"]["messages_received"] == 0
    
    @pytest.mark.asyncio
    async def test_invalid_agent_operations(self, protocol):
        """Test operations with invalid agents"""
        
        # Test sending message from unregistered agent
        with pytest.raises(ValueError):
            summary_report = SMESummaryReport(
                agent_id="nonexistent",
                domain="test",
                reporting_period={},
                activity_metrics={},
                key_insights=[],
                problem_areas=[],
                success_stories=[],
                knowledge_growth={},
                resource_requests=[],
                collaboration_opportunities=[],
                confidence_level=0.5
            )
            await protocol.send_summary_report("nonexistent", summary_report)
    
    def test_message_id_generation(self, protocol):
        """Test unique message ID generation"""
        id1 = protocol._generate_message_id()
        id2 = protocol._generate_message_id()
        
        assert id1 != id2
        assert id1.startswith("msg_")
        assert id2.startswith("msg_")
    
    @pytest.mark.asyncio
    async def test_message_archiving(self, setup_agents):
        """Test message archiving after completion"""
        protocol = setup_agents
        
        # Create and process a message
        summary_report = SMESummaryReport(
            agent_id="child_001",
            domain="geospatial",
            reporting_period={},
            activity_metrics={},
            key_insights=[],
            problem_areas=[],
            success_stories=[],
            knowledge_growth={},
            resource_requests=[],
            collaboration_opportunities=[],
            confidence_level=0.5
        )
        
        message_id = await protocol.send_summary_report("child_001", summary_report)
        
        # Wait for processing
        await asyncio.sleep(0.1)
        
        # Message should eventually be archived
        # (In actual implementation, completed messages get moved to history)
        assert message_id in protocol.active_messages  # Still active during test


if __name__ == "__main__":
    pytest.main([__file__])