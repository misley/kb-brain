"""
Tests for SME Agent System
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
from datetime import datetime

from kb_brain.intelligence.sme_agent_system import (
    SMEAgentSystem, SMEAgent, DomainExpertise, ExpertiseLevel, SMEStatus
)


class TestSMEAgentSystem:
    """Test cases for SME Agent System"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def sme_system(self, temp_dir):
        """Create SME system instance for testing"""
        return SMEAgentSystem(
            base_kb_path=temp_dir,
            specialization_threshold=0.5  # Lower threshold for testing
        )
    
    def test_system_initialization(self, sme_system):
        """Test system initialization"""
        assert sme_system.parent_agent_id is not None
        assert sme_system.parent_agent_id in sme_system.agents
        
        parent_agent = sme_system.agents[sme_system.parent_agent_id]
        assert parent_agent.domain == "general"
        assert parent_agent.parent_id is None
        assert parent_agent.status == SMEStatus.ACTIVE
    
    def test_domain_expertise_scoring(self):
        """Test domain expertise specialization scoring"""
        expertise = DomainExpertise(
            domain="test_domain",
            knowledge_count=100,
            solution_success_rate=0.9,
            query_volume=50,
            complexity_handled=15,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            confidence_score=0.8
        )
        
        score = expertise.calculate_specialization_score()
        assert 0.0 <= score <= 1.0
        
        # High activity should result in high score
        assert score > 0.7  # Should be high enough for specialization
    
    @pytest.mark.asyncio
    async def test_query_routing_to_parent(self, sme_system):
        """Test query routing to parent agent"""
        query = "What is Python programming?"
        context = {"domain": "general"}
        
        agent_id, response = await sme_system.route_query(query, context)
        
        assert agent_id == sme_system.parent_agent_id
        assert response["specialized"] == False
        assert "response" in response
    
    @pytest.mark.asyncio
    async def test_sme_creation_trigger(self, sme_system):
        """Test SME creation when domain reaches threshold"""
        domain = "test_domain"
        
        # Simulate multiple queries in same domain to trigger SME creation
        for i in range(15):  # Exceed threshold
            query = f"Test query {i} for {domain}"
            context = {"domain": domain}
            await sme_system.route_query(query, context)
        
        # Check if SME was created
        assert domain in sme_system.domain_map
        sme_id = sme_system.domain_map[domain]
        assert sme_id in sme_system.agents
        
        sme_agent = sme_system.agents[sme_id]
        assert sme_agent.domain == domain
        assert sme_agent.parent_id == sme_system.parent_agent_id
    
    @pytest.mark.asyncio
    async def test_sme_query_routing(self, sme_system):
        """Test query routing to specialized SME"""
        # First create an SME agent
        domain = "specialized_domain"
        expertise = DomainExpertise(
            domain=domain,
            knowledge_count=100,
            solution_success_rate=0.8,
            query_volume=50,
            complexity_handled=10,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.ADVANCED,
            confidence_score=0.9
        )
        
        sme_id = await sme_system._create_sme_agent(
            domain, expertise, sme_system.parent_agent_id
        )
        
        # Test routing to SME
        query = f"Specialized question about {domain}"
        context = {"domain": domain}
        
        agent_id, response = await sme_system.route_query(query, context)
        
        assert agent_id == sme_id
        assert response["specialized"] == True
        assert response["sme_domain"] == domain
    
    @pytest.mark.asyncio
    async def test_escalation_to_parent(self, sme_system):
        """Test escalation from SME to parent"""
        # Create SME with lower expertise
        domain = "test_escalation"
        expertise = DomainExpertise(
            domain=domain,
            knowledge_count=50,
            solution_success_rate=0.7,
            query_volume=25,
            complexity_handled=5,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.INTERMEDIATE,  # Lower level
            confidence_score=0.6
        )
        
        sme_id = await sme_system._create_sme_agent(
            domain, expertise, sme_system.parent_agent_id
        )
        
        # Test complex query that should trigger escalation
        complex_query = "Extremely complex analysis requiring advanced expertise"
        context = {"domain": domain, "complexity": 4}  # High complexity
        
        agent_id, response = await sme_system.route_query(complex_query, context)
        
        # Should route to SME but escalate to parent
        assert agent_id == sme_id
        assert response.get("escalated_to_parent") == True
        assert "parent_insights" in response
    
    def test_agent_registry_operations(self, sme_system):
        """Test agent registration and lookup"""
        # Test parent agent is registered
        parent_id = sme_system.parent_agent_id
        assert parent_id in sme_system.agents
        
        parent_agent = sme_system.agents[parent_id]
        assert parent_agent.domain == "general"
        assert parent_agent.status == SMEStatus.ACTIVE
    
    def test_expertise_evolution(self, sme_system):
        """Test SME expertise level evolution"""
        # Create SME with intermediate level
        expertise = DomainExpertise(
            domain="evolving_domain",
            knowledge_count=50,
            solution_success_rate=0.8,
            query_volume=75,
            complexity_handled=15,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            confidence_score=0.7
        )
        
        sme_agent = SMEAgent(
            id="test_sme_001",
            domain="evolving_domain",
            parent_id=sme_system.parent_agent_id,
            children_ids=[],
            status=SMEStatus.ACTIVE,
            expertise=expertise,
            knowledge_base_path="/tmp/test",
            creation_date=datetime.now(),
            last_summary_update=datetime.now(),
            specialization_triggers=[]
        )
        
        # Simulate successful performance
        sme_agent.expertise.query_volume = 100
        sme_agent.expertise.solution_success_rate = 0.9
        sme_agent.expertise.complexity_handled = 20
        
        # Test expertise evolution
        sme_system._assess_expertise_evolution(sme_agent)
        
        # Should have evolved to advanced level
        assert sme_agent.expertise.expertise_level == ExpertiseLevel.ADVANCED
        assert sme_agent.status == SMEStatus.SPECIALIZED
    
    def test_system_status_reporting(self, sme_system):
        """Test system status reporting"""
        status = sme_system.get_sme_system_status()
        
        assert "total_agents" in status
        assert "parent_agent" in status
        assert "specialized_agents" in status
        assert "active_domains" in status
        assert "system_metrics" in status
        
        assert status["total_agents"] >= 1  # At least parent agent
        assert status["parent_agent"] == sme_system.parent_agent_id
    
    @pytest.mark.asyncio
    async def test_cross_sme_consultation(self, sme_system):
        """Test consultation between SME agents"""
        # Create two SME agents
        domain1 = "domain_one"
        domain2 = "domain_two"
        
        expertise1 = DomainExpertise(
            domain=domain1,
            knowledge_count=75,
            solution_success_rate=0.8,
            query_volume=40,
            complexity_handled=12,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.ADVANCED,
            confidence_score=0.8
        )
        
        expertise2 = DomainExpertise(
            domain=domain2,
            knowledge_count=60,
            solution_success_rate=0.7,
            query_volume=35,
            complexity_handled=10,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            confidence_score=0.7
        )
        
        sme_id1 = await sme_system._create_sme_agent(
            domain1, expertise1, sme_system.parent_agent_id
        )
        sme_id2 = await sme_system._create_sme_agent(
            domain2, expertise2, sme_system.parent_agent_id
        )
        
        # Test finding relevant child SMEs
        relevant_smes = sme_system._find_relevant_child_smes(domain1)
        
        # Should find the SME for domain1
        assert any(sme.id == sme_id1 for sme in relevant_smes)
    
    def test_knowledge_migration(self, sme_system):
        """Test knowledge migration to new SME"""
        # This is tested implicitly in SME creation, but we can verify
        # the migration process creates appropriate knowledge files
        domain = "migration_test"
        expertise = DomainExpertise(
            domain=domain,
            knowledge_count=25,
            solution_success_rate=0.8,
            query_volume=15,
            complexity_handled=5,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.INTERMEDIATE,
            confidence_score=0.7
        )
        
        # Mock the knowledge migration process
        with patch.object(sme_system, '_migrate_knowledge_to_sme') as mock_migrate:
            asyncio.run(sme_system._create_sme_agent(
                domain, expertise, sme_system.parent_agent_id
            ))
            
            # Verify migration was called
            mock_migrate.assert_called_once()
    
    def test_specialization_threshold_enforcement(self, sme_system):
        """Test specialization threshold enforcement"""
        # Test with expertise below threshold
        low_expertise = DomainExpertise(
            domain="low_domain",
            knowledge_count=10,  # Low count
            solution_success_rate=0.6,  # Low success rate
            query_volume=5,  # Low volume
            complexity_handled=2,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.NOVICE,
            confidence_score=0.4
        )
        
        score = low_expertise.calculate_specialization_score()
        assert score < sme_system.specialization_threshold
        
        # Test with expertise above threshold
        high_expertise = DomainExpertise(
            domain="high_domain",
            knowledge_count=100,
            solution_success_rate=0.9,
            query_volume=75,
            complexity_handled=20,
            last_activity=datetime.now(),
            expertise_level=ExpertiseLevel.ADVANCED,
            confidence_score=0.9
        )
        
        score = high_expertise.calculate_specialization_score()
        assert score >= sme_system.specialization_threshold


if __name__ == "__main__":
    pytest.main([__file__])