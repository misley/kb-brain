"""
Tests for Prompt Classifier
"""

import pytest
from kb_brain.intelligence.prompt_classifier import (
    PromptClassifier, PromptIntent, ComplexityLevel, Domain
)


class TestPromptClassifier:
    """Test cases for PromptClassifier"""
    
    @pytest.fixture
    def classifier(self):
        """Create a prompt classifier instance"""
        return PromptClassifier()
    
    def test_simple_question_classification(self, classifier):
        """Test classification of simple questions"""
        prompts = [
            "What is Python?",
            "How do I install numpy?", 
            "Define machine learning"
        ]
        
        for prompt in prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.intent == PromptIntent.SIMPLE_QUESTION
            assert classification.complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
    
    def test_code_request_classification(self, classifier):
        """Test classification of code requests"""
        prompts = [
            "Write a function to calculate fibonacci numbers",
            "Implement a binary search algorithm",
            "Create a REST API endpoint"
        ]
        
        for prompt in prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.intent == PromptIntent.CODE_REQUEST
            assert classification.domain == Domain.PROGRAMMING
    
    def test_debug_help_classification(self, classifier):
        """Test classification of debugging requests"""
        prompts = [
            "Fix this SSL certificate error",
            "Why is my code throwing a TypeError?",
            "Debug connection timeout issue"
        ]
        
        for prompt in prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.intent == PromptIntent.DEBUG_HELP
            assert classification.requires_kb_search
    
    def test_project_status_classification(self, classifier):
        """Test classification of project status requests"""
        prompts = [
            "What's the status of the Dunes project?",
            "Show me current progress on WHSA analysis",
            "What are the next steps for research?"
        ]
        
        for prompt in prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.intent == PromptIntent.PROJECT_STATUS
            assert classification.requires_kb_search
    
    def test_domain_classification(self, classifier):
        """Test domain classification"""
        test_cases = [
            ("Analyze satellite imagery with NDWI", Domain.GEOSPATIAL),
            ("Train a machine learning model", Domain.DATA_SCIENCE),
            ("Fix SSL certificate issues", Domain.SYSTEM_ADMIN),
            ("Implement a Python function", Domain.PROGRAMMING),
            ("Research boundary detection methods", Domain.RESEARCH)
        ]
        
        for prompt, expected_domain in test_cases:
            classification = classifier.classify_prompt(prompt)
            assert classification.domain == expected_domain
    
    def test_complexity_assessment(self, classifier):
        """Test complexity level assessment"""
        simple_prompts = [
            "What is git?",
            "Quick help with imports"
        ]
        
        complex_prompts = [
            "Perform comprehensive cross-repository analysis of boundary detection methods",
            "Implement advanced GPU-accelerated similarity search with fallback mechanisms"
        ]
        
        for prompt in simple_prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.complexity in [ComplexityLevel.SIMPLE, ComplexityLevel.MODERATE]
        
        for prompt in complex_prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ITERATIVE]
    
    def test_context_integration(self, classifier):
        """Test integration of context information"""
        prompt = "Analyze this data"
        context = {
            "file_path": "/projects/dunes/analysis.py",
            "project": "whsa"
        }
        
        classification = classifier.classify_prompt(prompt, context)
        
        # Context should influence domain classification
        assert classification.domain in [Domain.GEOSPATIAL, Domain.DATA_SCIENCE]
        assert "dunes" in classification.context_hints.get("detected_projects", [])
    
    def test_processing_recommendations(self, classifier):
        """Test processing recommendations"""
        simple_prompt = "What is pandas?"
        complex_prompt = "Analyze all boundary detection methods across repositories"
        
        simple_classification = classifier.classify_prompt(simple_prompt)
        complex_classification = classifier.classify_prompt(complex_prompt)
        
        simple_recs = classifier.get_processing_recommendations(simple_classification)
        complex_recs = classifier.get_processing_recommendations(complex_classification)
        
        assert simple_recs["route_to"] == "direct_response"
        assert complex_recs["route_to"] in ["kb_enhanced_processing", "screen_processing"]
        assert complex_recs["kb_search_depth"] in ["deep", "cross_repo"]
    
    def test_confidence_scoring(self, classifier):
        """Test confidence scoring"""
        clear_prompt = "What is Python programming language?"
        ambiguous_prompt = "Help me with this thing"
        
        clear_classification = classifier.classify_prompt(clear_prompt)
        ambiguous_classification = classifier.classify_prompt(ambiguous_prompt)
        
        assert clear_classification.confidence > ambiguous_classification.confidence
        assert clear_classification.confidence > 0.8
    
    def test_keyword_extraction(self, classifier):
        """Test keyword extraction"""
        prompt = "Fix SSL certificate issues in Docker container using Python"
        classification = classifier.classify_prompt(prompt)
        
        expected_keywords = ["ssl", "certificate", "docker", "python"]
        assert any(keyword in classification.keywords for keyword in expected_keywords)
    
    def test_cross_repo_detection(self, classifier):
        """Test cross-repository requirement detection"""
        cross_repo_prompts = [
            "Compare methods across all WHSA repositories",
            "Analyze Dunes project across multiple repos"
        ]
        
        for prompt in cross_repo_prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.requires_cross_repo
    
    def test_iterative_processing_detection(self, classifier):
        """Test iterative processing requirement detection"""
        iterative_prompts = [
            "Perform step by step comprehensive analysis",
            "Run iterative improvement on the algorithm"
        ]
        
        for prompt in iterative_prompts:
            classification = classifier.classify_prompt(prompt)
            assert classification.requires_iterative


if __name__ == "__main__":
    pytest.main([__file__])