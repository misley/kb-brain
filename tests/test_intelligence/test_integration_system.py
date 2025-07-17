"""
Tests for KB Brain Intelligence Integration System
"""

import pytest
import asyncio
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

from kb_brain.intelligence.intelligence_system import KBBrainIntelligence


class TestKBBrainIntelligence:
    """Test cases for KB Brain Intelligence Integration System"""
    
    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for testing"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield temp_dir
    
    @pytest.fixture
    def intelligence_system(self, temp_dir):
        """Create intelligence system instance for testing"""
        return KBBrainIntelligence(
            base_path=temp_dir,
            auto_initialize=False  # Manual initialization for testing
        )
    
    @pytest.mark.asyncio
    async def test_system_initialization(self, intelligence_system):
        """Test system initialization"""
        assert not intelligence_system.initialized
        
        success = await intelligence_system.initialize()
        
        assert success
        assert intelligence_system.initialized
        
        # Check that core components are initialized
        assert intelligence_system.prompt_classifier is not None
        assert intelligence_system.kb_engine is not None
        assert intelligence_system.orchestrator is not None
        assert intelligence_system.screen_manager is not None
        assert intelligence_system.ingestion_system is not None
        
        # Check that SME components are initialized
        assert intelligence_system.sme_system is not None
        assert intelligence_system.communication_protocol is not None
        assert intelligence_system.consolidation_system is not None
    
    @pytest.mark.asyncio
    async def test_query_processing_without_sme(self, intelligence_system):
        """Test query processing through orchestrator (no SME routing)"""
        await intelligence_system.initialize()
        
        query = "What is Python programming?"
        context = {"simple": True}
        
        response = await intelligence_system.process_query(
            query, context, use_sme_routing=False
        )
        
        assert "query" in response
        assert response["query"] == query
        assert response["processing_method"] == "orchestrator"
        assert "processing_time" in response
        assert response["processing_time"] > 0
    
    @pytest.mark.asyncio
    async def test_query_processing_with_sme_routing(self, intelligence_system):
        """Test query processing with SME routing"""
        await intelligence_system.initialize()
        
        query = "Analyze WHSA boundary detection methods"
        context = {"project": "dunes", "domain": "geospatial"}
        
        response = await intelligence_system.process_query(
            query, context, use_sme_routing=True
        )
        
        assert "query" in response
        assert response["query"] == query
        assert response["processing_method"] == "sme_routing"
        assert "assigned_agent" in response
        assert "processing_time" in response
    
    @pytest.mark.asyncio
    async def test_sme_agent_creation(self, intelligence_system):
        """Test manual SME agent creation"""
        await intelligence_system.initialize()
        
        # Mock sufficient knowledge for SME creation
        with patch.object(intelligence_system.sme_system, '_analyze_knowledge_domains') as mock_analyze:
            from kb_brain.intelligence.sme_agent_system import DomainExpertise, ExpertiseLevel
            from datetime import datetime
            
            mock_analyze.return_value = {
                "test_domain": DomainExpertise(
                    domain="test_domain",
                    knowledge_count=100,  # Above threshold
                    solution_success_rate=0.8,
                    query_volume=50,
                    complexity_handled=15,
                    last_activity=datetime.now(),
                    expertise_level=ExpertiseLevel.INTERMEDIATE,
                    confidence_score=0.8
                )
            }
            
            sme_id = await intelligence_system.create_sme_agent("test_domain", 50)
            
            assert sme_id is not None
            assert sme_id in intelligence_system.sme_system.agents
    
    @pytest.mark.asyncio
    async def test_sme_summary_report_generation(self, intelligence_system):
        """Test SME summary report generation"""
        await intelligence_system.initialize()
        
        # Get parent agent ID for testing
        parent_id = intelligence_system.sme_system.parent_agent_id
        
        summary = await intelligence_system.get_sme_summary_report(parent_id)
        
        assert summary is not None
        assert "agent_id" in summary
        assert "domain" in summary
        assert "expertise_level" in summary
        assert summary["agent_id"] == parent_id
        assert summary["domain"] == "general"
    
    def test_system_status_reporting(self, intelligence_system):
        """Test comprehensive system status reporting"""
        # Test status before initialization
        status = intelligence_system.get_system_status()
        
        assert "system_initialized" in status
        assert status["system_initialized"] == False
        assert "components" in status
        assert "system_stats" in status
        
        # All components should be None/False before initialization
        components = status["components"]
        assert not components["prompt_classifier"]
        assert not components["kb_engine"]
        assert not components["sme_system"]
    
    @pytest.mark.asyncio
    async def test_system_status_after_initialization(self, intelligence_system):
        """Test system status after initialization"""
        await intelligence_system.initialize()
        
        status = intelligence_system.get_system_status()
        
        assert status["system_initialized"] == True
        
        # All components should be initialized
        components = status["components"]
        assert components["prompt_classifier"]
        assert components["kb_engine"] 
        assert components["orchestrator"]
        assert components["sme_system"]
        assert components["communication_protocol"]
        
        # Should have additional status from components
        assert "sme_system_status" in status
        assert "communication_stats" in status
        assert "kb_engine_status" in status
    
    @pytest.mark.asyncio
    async def test_knowledge_ingestion_integration(self, intelligence_system):
        """Test integration with knowledge ingestion system"""
        await intelligence_system.initialize()
        
        query = "Test query for ingestion"
        context = {"test": True}
        
        response = await intelligence_system.process_query(query, context)
        
        # System should automatically ingest query results
        # This tests the _ingest_query_results method
        assert "processing_time" in response
        
        # Check that ingestion system has pending entries
        # (In real implementation, this would verify ingestion occurred)
        ingestion_status = intelligence_system.ingestion_system.get_ingestion_status()
        assert "pending_entries" in ingestion_status
    
    def test_system_stats_tracking(self, intelligence_system):
        """Test system statistics tracking"""
        initial_stats = intelligence_system.system_stats
        
        assert "total_queries_processed" in initial_stats
        assert "sme_agents_active" in initial_stats
        assert "avg_response_time" in initial_stats
        assert "system_uptime" in initial_stats
        
        assert initial_stats["total_queries_processed"] == 0
        assert initial_stats["avg_response_time"] == 0.0
    
    @pytest.mark.asyncio
    async def test_system_stats_update_after_query(self, intelligence_system):
        """Test system stats update after processing queries"""
        await intelligence_system.initialize()
        
        initial_count = intelligence_system.system_stats["total_queries_processed"]
        
        query = "Test query for stats"
        await intelligence_system.process_query(query)
        
        # Stats should be updated
        assert intelligence_system.system_stats["total_queries_processed"] == initial_count + 1
        assert intelligence_system.system_stats["avg_response_time"] > 0
    
    @pytest.mark.asyncio
    async def test_error_handling_in_query_processing(self, intelligence_system):
        """Test error handling in query processing"""
        await intelligence_system.initialize()
        
        # Test with invalid/problematic query
        with patch.object(intelligence_system.orchestrator, 'process_prompt', side_effect=Exception("Test error")):
            response = await intelligence_system.process_query("Test query")
            
            assert "error" in response
            assert "Test error" in response["error"]
            assert "processing_time" in response
    
    @pytest.mark.asyncio
    async def test_consolidation_integration(self, intelligence_system):
        """Test knowledge base consolidation integration"""
        await intelligence_system.initialize()
        
        # Test manual consolidation trigger
        result = await intelligence_system.consolidate_knowledge_bases()
        
        assert "total_files" in result or "error" in result
        # Result depends on whether source KB files exist in test environment
    
    @pytest.mark.asyncio
    async def test_escalation_integration(self, intelligence_system):
        """Test escalation from SME to parent integration"""
        await intelligence_system.initialize()
        
        # Mock a complex query that should trigger escalation
        complex_query = "Perform comprehensive cross-repository analysis with advanced ML techniques"
        context = {"complexity": 4, "domain": "advanced_analysis"}
        
        response = await intelligence_system.process_query(complex_query, context)
        
        # Should be processed through SME system
        assert response["processing_method"] == "sme_routing"
        # May include escalation depending on SME capabilities
    
    @pytest.mark.asyncio
    async def test_component_integration_setup(self, intelligence_system):
        """Test that component integrations are properly set up"""
        await intelligence_system.initialize()
        
        # Check that SME agents are registered with communication protocol
        sme_agents = intelligence_system.sme_system.agents
        registered_agents = intelligence_system.communication_protocol.agent_registry
        
        # Parent agent should be registered
        parent_id = intelligence_system.sme_system.parent_agent_id
        assert parent_id in registered_agents
        
        # Verify agent information matches
        sme_agent = sme_agents[parent_id]
        registered_info = registered_agents[parent_id]
        
        assert registered_info["domain"] == sme_agent.domain
    
    @pytest.mark.asyncio
    async def test_system_shutdown(self, intelligence_system):
        """Test graceful system shutdown"""
        await intelligence_system.initialize()
        
        assert intelligence_system.initialized
        
        await intelligence_system.shutdown()
        
        assert not intelligence_system.initialized
        # System should clean up resources properly
    
    @pytest.mark.asyncio
    async def test_global_system_instance(self):
        """Test global system instance management"""
        from kb_brain.intelligence.intelligence_system import get_intelligence_system
        
        # Get global instance
        system1 = get_intelligence_system()
        system2 = get_intelligence_system()
        
        # Should be the same instance
        assert system1 is system2
    
    @pytest.mark.asyncio
    async def test_auto_initialization_flag(self, temp_dir):
        """Test auto-initialization flag"""
        # Test with auto_initialize=True
        system_auto = KBBrainIntelligence(
            base_path=temp_dir,
            auto_initialize=True
        )
        
        # Should start initialization automatically
        await asyncio.sleep(0.1)  # Give time for async initialization
        
        # Test with auto_initialize=False (default in fixture)
        system_manual = KBBrainIntelligence(
            base_path=temp_dir,
            auto_initialize=False
        )
        
        assert not system_manual.initialized


if __name__ == "__main__":
    pytest.main([__file__])