"""
Processing Orchestrator for Intelligent Prompt Routing and Workflow Management
Coordinates between prompt classification, KB search, AI processing, and response synthesis
"""

import asyncio
import json
import time
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, asdict
from enum import Enum
import logging
from datetime import datetime

from .prompt_classifier import PromptClassifier, PromptClassification, ComplexityLevel
from .kb_integration_engine import KBIntegrationEngine, SearchDepth, KnowledgeResponse

logger = logging.getLogger(__name__)


class ProcessingRoute(Enum):
    """Processing route options"""
    DIRECT_RESPONSE = "direct_response"
    KB_ENHANCED = "kb_enhanced"
    SCREEN_PROCESSING = "screen_processing"
    ITERATIVE_ANALYSIS = "iterative_analysis"
    CROSS_REPO_RESEARCH = "cross_repo_research"


class ProcessingStatus(Enum):
    """Processing status tracking"""
    PENDING = "pending"
    CLASSIFYING = "classifying"
    KB_SEARCHING = "kb_searching"
    AI_PROCESSING = "ai_processing"
    SYNTHESIZING = "synthesizing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class ProcessingRequest:
    """Complete processing request with metadata"""
    id: str
    prompt: str
    context: Dict[str, Any]
    classification: Optional[PromptClassification] = None
    route: Optional[ProcessingRoute] = None
    priority: str = "normal"
    created_at: datetime = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class ProcessingResult:
    """Complete processing result"""
    request_id: str
    status: ProcessingStatus
    response: str
    kb_results: Optional[KnowledgeResponse] = None
    ai_insights: Dict[str, Any] = None
    processing_time: float = 0.0
    route_taken: Optional[ProcessingRoute] = None
    screen_sessions: List[str] = None
    confidence: float = 0.0
    recommendations: List[str] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.ai_insights is None:
            self.ai_insights = {}
        if self.screen_sessions is None:
            self.screen_sessions = []
        if self.recommendations is None:
            self.recommendations = []
        if self.metadata is None:
            self.metadata = {}


class ProcessingOrchestrator:
    """Central orchestrator for intelligent prompt processing"""
    
    def __init__(self, 
                 kb_engine: Optional[KBIntegrationEngine] = None,
                 screen_manager = None,
                 ai_processor: Optional[Callable] = None):
        """
        Initialize the processing orchestrator
        
        Args:
            kb_engine: Knowledge base integration engine
            screen_manager: Screen session manager for background processing
            ai_processor: AI processing function/service
        """
        self.prompt_classifier = PromptClassifier()
        self.kb_engine = kb_engine or KBIntegrationEngine()
        self.screen_manager = screen_manager
        self.ai_processor = ai_processor
        
        # Processing tracking
        self.active_requests: Dict[str, ProcessingRequest] = {}
        self.processing_history: List[ProcessingResult] = []
        
        # Route handlers
        self.route_handlers = {
            ProcessingRoute.DIRECT_RESPONSE: self._handle_direct_response,
            ProcessingRoute.KB_ENHANCED: self._handle_kb_enhanced,
            ProcessingRoute.SCREEN_PROCESSING: self._handle_screen_processing,
            ProcessingRoute.ITERATIVE_ANALYSIS: self._handle_iterative_analysis,
            ProcessingRoute.CROSS_REPO_RESEARCH: self._handle_cross_repo_research
        }
        
        # Performance metrics
        self.metrics = {
            'total_requests': 0,
            'avg_processing_time': 0.0,
            'route_usage': {route.value: 0 for route in ProcessingRoute},
            'success_rate': 0.0
        }
        
        logger.info("Processing Orchestrator initialized")
    
    async def process_prompt(self, 
                           prompt: str, 
                           context: Optional[Dict[str, Any]] = None,
                           priority: str = "normal") -> ProcessingResult:
        """
        Main entry point for intelligent prompt processing
        
        Args:
            prompt: The input prompt to process
            context: Optional context (file paths, project info, etc.)
            priority: Processing priority (low, normal, high, urgent)
        
        Returns:
            ProcessingResult with complete response and metadata
        """
        
        start_time = time.time()
        request_id = str(uuid.uuid4())
        
        # Create processing request
        request = ProcessingRequest(
            id=request_id,
            prompt=prompt,
            context=context or {},
            priority=priority
        )
        
        self.active_requests[request_id] = request
        self.metrics['total_requests'] += 1
        
        try:
            # Step 1: Classify the prompt
            result = ProcessingResult(request_id=request_id, status=ProcessingStatus.CLASSIFYING, response="")
            classification = self._classify_prompt(prompt, context)
            request.classification = classification
            
            # Step 2: Determine processing route
            route = self._determine_route(classification)
            request.route = route
            result.route_taken = route
            self.metrics['route_usage'][route.value] += 1
            
            logger.info(f"Request {request_id}: Route={route.value}, Complexity={classification.complexity.value}")
            
            # Step 3: Execute processing route
            result = await self.route_handlers[route](request, result)
            
            # Step 4: Finalize result
            result.processing_time = time.time() - start_time
            result.status = ProcessingStatus.COMPLETED
            
            # Update metrics
            self._update_metrics(result)
            
        except Exception as e:
            logger.error(f"Processing failed for request {request_id}: {e}")
            result = ProcessingResult(
                request_id=request_id,
                status=ProcessingStatus.FAILED,
                response=f"Processing failed: {str(e)}",
                error=str(e),
                processing_time=time.time() - start_time
            )
        
        finally:
            # Clean up
            if request_id in self.active_requests:
                del self.active_requests[request_id]
            
            self.processing_history.append(result)
            
            # Keep history manageable
            if len(self.processing_history) > 1000:
                self.processing_history = self.processing_history[-500:]
        
        return result
    
    def _classify_prompt(self, prompt: str, context: Optional[Dict[str, Any]]) -> PromptClassification:
        """Classify the prompt for processing requirements"""
        
        return self.prompt_classifier.classify_prompt(prompt, context)
    
    def _determine_route(self, classification: PromptClassification) -> ProcessingRoute:
        """Determine the optimal processing route based on classification"""
        
        # Get routing recommendations
        recommendations = self.prompt_classifier.get_processing_recommendations(classification)
        route_recommendation = recommendations.get('route_to', 'direct_response')
        
        # Map recommendations to routes
        route_map = {
            'direct_response': ProcessingRoute.DIRECT_RESPONSE,
            'kb_enhanced_processing': ProcessingRoute.KB_ENHANCED,
            'screen_processing': ProcessingRoute.SCREEN_PROCESSING
        }
        
        base_route = route_map.get(route_recommendation, ProcessingRoute.DIRECT_RESPONSE)
        
        # Override based on specific conditions
        if classification.requires_cross_repo:
            return ProcessingRoute.CROSS_REPO_RESEARCH
        elif classification.requires_iterative:
            return ProcessingRoute.ITERATIVE_ANALYSIS
        elif classification.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ITERATIVE]:
            if self.screen_manager:
                return ProcessingRoute.SCREEN_PROCESSING
            else:
                return ProcessingRoute.KB_ENHANCED
        
        return base_route
    
    async def _handle_direct_response(self, request: ProcessingRequest, result: ProcessingResult) -> ProcessingResult:
        """Handle simple direct response processing"""
        
        result.status = ProcessingStatus.AI_PROCESSING
        
        # For simple prompts, minimal processing
        if self.ai_processor:
            ai_response = await self._call_ai_processor(
                request.prompt, 
                context=request.context,
                mode="simple"
            )
            result.response = ai_response.get('response', 'Simple response generated')
            result.confidence = ai_response.get('confidence', 0.8)
        else:
            result.response = f"Direct response for: {request.prompt}"
            result.confidence = 0.7
        
        result.recommendations = ["Consider using KB search for more detailed information"]
        
        return result
    
    async def _handle_kb_enhanced(self, request: ProcessingRequest, result: ProcessingResult) -> ProcessingResult:
        """Handle KB-enhanced processing with knowledge base search"""
        
        # Step 1: Search knowledge base
        result.status = ProcessingStatus.KB_SEARCHING
        
        classification = request.classification
        search_depth = SearchDepth.DEEP if classification.complexity == ComplexityLevel.COMPLEX else SearchDepth.SHALLOW
        
        kb_response = self.kb_engine.search_knowledge(
            query=request.prompt,
            search_depth=search_depth,
            domain_filter=classification.domain.value if classification.domain else None,
            max_results=5
        )
        
        result.kb_results = kb_response
        
        # Step 2: AI processing with KB context
        result.status = ProcessingStatus.AI_PROCESSING
        
        if self.ai_processor:
            ai_context = {
                'kb_results': [asdict(r) for r in kb_response.results],
                'kb_insights': kb_response.cross_repo_insights,
                'original_context': request.context
            }
            
            ai_response = await self._call_ai_processor(
                request.prompt,
                context=ai_context,
                mode="kb_enhanced"
            )
            
            result.response = ai_response.get('response', 'KB-enhanced response generated')
            result.ai_insights = ai_response.get('insights', {})
            result.confidence = ai_response.get('confidence', 0.85)
        else:
            # Fallback: synthesize KB results
            result.response = self._synthesize_kb_results(request.prompt, kb_response)
            result.confidence = kb_response.confidence
        
        # Combine recommendations
        result.recommendations = kb_response.recommendations + result.recommendations
        
        return result
    
    async def _handle_screen_processing(self, request: ProcessingRequest, result: ProcessingResult) -> ProcessingResult:
        """Handle background screen-based processing"""
        
        if not self.screen_manager:
            # Fallback to KB enhanced if no screen manager
            return await self._handle_kb_enhanced(request, result)
        
        # Create screen session for background processing
        session_name = f"process_{request.id[:8]}"
        
        try:
            # Start background processing
            screen_info = self.screen_manager.create_processing_session(
                session_name=session_name,
                task_type="prompt_processing",
                prompt=request.prompt,
                context=request.context
            )
            
            result.screen_sessions = [session_name]
            result.status = ProcessingStatus.AI_PROCESSING
            
            # Monitor progress
            await self._monitor_screen_processing(session_name, result)
            
        except Exception as e:
            logger.error(f"Screen processing failed: {e}")
            # Fallback to KB enhanced
            return await self._handle_kb_enhanced(request, result)
        
        return result
    
    async def _handle_iterative_analysis(self, request: ProcessingRequest, result: ProcessingResult) -> ProcessingResult:
        """Handle complex iterative analysis with multiple AI rounds"""
        
        result.status = ProcessingStatus.AI_PROCESSING
        
        # Multi-stage processing
        stages = [
            "initial_analysis",
            "deep_research", 
            "cross_validation",
            "synthesis"
        ]
        
        cumulative_insights = {}
        cumulative_response = []
        
        for stage in stages:
            stage_result = await self._process_stage(
                request.prompt,
                stage,
                request.context,
                cumulative_insights
            )
            
            cumulative_insights[stage] = stage_result
            cumulative_response.append(f"Stage {stage}: {stage_result.get('summary', 'Completed')}")
        
        result.response = "\n".join(cumulative_response)
        result.ai_insights = cumulative_insights
        result.confidence = 0.9  # High confidence from thorough analysis
        
        return result
    
    async def _handle_cross_repo_research(self, request: ProcessingRequest, result: ProcessingResult) -> ProcessingResult:
        """Handle cross-repository research and analysis"""
        
        # Step 1: Comprehensive KB search across repositories
        result.status = ProcessingStatus.KB_SEARCHING
        
        kb_response = self.kb_engine.search_knowledge(
            query=request.prompt,
            search_depth=SearchDepth.CROSS_REPO,
            max_results=15  # More results for comprehensive analysis
        )
        
        result.kb_results = kb_response
        
        # Step 2: Cross-repository analysis
        result.status = ProcessingStatus.AI_PROCESSING
        
        if self.ai_processor:
            research_context = {
                'all_kb_results': [asdict(r) for r in kb_response.results],
                'cross_repo_insights': kb_response.cross_repo_insights,
                'repositories': kb_response.cross_repo_insights.get('repositories_found', []),
                'research_mode': True
            }
            
            ai_response = await self._call_ai_processor(
                request.prompt,
                context=research_context,
                mode="cross_repo_research"
            )
            
            result.response = ai_response.get('response', 'Cross-repository research completed')
            result.ai_insights = ai_response.get('insights', {})
            result.confidence = ai_response.get('confidence', 0.9)
        else:
            # Fallback: comprehensive KB synthesis
            result.response = self._synthesize_cross_repo_results(request.prompt, kb_response)
            result.confidence = kb_response.confidence
        
        result.recommendations = kb_response.recommendations
        
        return result
    
    async def _call_ai_processor(self, 
                                prompt: str, 
                                context: Dict[str, Any], 
                                mode: str) -> Dict[str, Any]:
        """Call the AI processor with appropriate context"""
        
        if not self.ai_processor:
            return {
                'response': f"AI processing not available for mode: {mode}",
                'confidence': 0.5
            }
        
        try:
            # Call the AI processor (could be local model, API call, etc.)
            if asyncio.iscoroutinefunction(self.ai_processor):
                return await self.ai_processor(prompt, context, mode)
            else:
                return self.ai_processor(prompt, context, mode)
        except Exception as e:
            logger.error(f"AI processor call failed: {e}")
            return {
                'response': f"AI processing failed: {str(e)}",
                'confidence': 0.3,
                'error': str(e)
            }
    
    async def _monitor_screen_processing(self, session_name: str, result: ProcessingResult):
        """Monitor screen-based background processing"""
        
        max_wait = 300  # 5 minutes max
        check_interval = 10  # Check every 10 seconds
        waited = 0
        
        while waited < max_wait:
            try:
                status = self.screen_manager.get_session_status(session_name)
                
                if status.get('completed', False):
                    # Processing completed
                    output = self.screen_manager.get_session_output(session_name)
                    result.response = output.get('response', 'Background processing completed')
                    result.ai_insights = output.get('insights', {})
                    result.confidence = output.get('confidence', 0.8)
                    break
                
                await asyncio.sleep(check_interval)
                waited += check_interval
                
            except Exception as e:
                logger.error(f"Error monitoring screen session {session_name}: {e}")
                break
        
        if waited >= max_wait:
            result.response = "Background processing timed out"
            result.confidence = 0.3
    
    async def _process_stage(self, 
                           prompt: str, 
                           stage: str, 
                           context: Dict[str, Any], 
                           previous_insights: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single stage of iterative analysis"""
        
        stage_context = {
            'stage': stage,
            'previous_insights': previous_insights,
            'original_context': context
        }
        
        if self.ai_processor:
            return await self._call_ai_processor(prompt, stage_context, f"iterative_{stage}")
        else:
            return {
                'summary': f"Stage {stage} completed",
                'insights': f"Insights from {stage}",
                'confidence': 0.7
            }
    
    def _synthesize_kb_results(self, prompt: str, kb_response: KnowledgeResponse) -> str:
        """Synthesize knowledge base results into a response"""
        
        if not kb_response.results:
            return "No relevant knowledge found in the knowledge base."
        
        synthesis = [f"Found {len(kb_response.results)} relevant knowledge entries:"]
        
        for i, result in enumerate(kb_response.results[:3], 1):
            confidence_str = f"({result.confidence:.1f} confidence)"
            synthesis.append(f"{i}. {result.source.domain} knowledge {confidence_str}")
            
            if isinstance(result.content, dict) and 'solution' in result.content:
                synthesis.append(f"   Solution: {result.content['solution']}")
        
        if kb_response.recommendations:
            synthesis.append("\nRecommendations:")
            synthesis.extend(f"- {rec}" for rec in kb_response.recommendations[:3])
        
        return "\n".join(synthesis)
    
    def _synthesize_cross_repo_results(self, prompt: str, kb_response: KnowledgeResponse) -> str:
        """Synthesize cross-repository research results"""
        
        synthesis = [f"Cross-repository analysis for: {prompt}"]
        synthesis.append(f"Searched {kb_response.total_sources_searched} knowledge sources")
        
        if kb_response.cross_repo_insights['repositories_found']:
            repos = kb_response.cross_repo_insights['repositories_found']
            synthesis.append(f"Found information across {len(repos)} repositories: {', '.join(repos)}")
        
        if kb_response.cross_repo_insights['common_patterns']:
            patterns = kb_response.cross_repo_insights['common_patterns']
            synthesis.append(f"Common patterns identified: {', '.join(patterns[:3])}")
        
        synthesis.append(f"\nFound {len(kb_response.results)} relevant knowledge entries")
        synthesis.append(f"Overall confidence: {kb_response.confidence:.1f}")
        
        return "\n".join(synthesis)
    
    def _update_metrics(self, result: ProcessingResult):
        """Update performance metrics"""
        
        # Calculate running average processing time
        current_avg = self.metrics['avg_processing_time']
        total_requests = self.metrics['total_requests']
        
        self.metrics['avg_processing_time'] = (
            (current_avg * (total_requests - 1) + result.processing_time) / total_requests
        )
        
        # Update success rate
        successful_requests = len([r for r in self.processing_history if r.status == ProcessingStatus.COMPLETED])
        self.metrics['success_rate'] = successful_requests / len(self.processing_history) if self.processing_history else 0.0
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            'orchestrator_status': {
                'active_requests': len(self.active_requests),
                'total_processed': len(self.processing_history),
                'metrics': self.metrics
            },
            'kb_engine_status': self.kb_engine.get_system_status() if self.kb_engine else None,
            'screen_manager_status': self.screen_manager.get_status() if self.screen_manager else None,
            'ai_processor_available': self.ai_processor is not None
        }
    
    def get_processing_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent processing history"""
        
        recent_history = self.processing_history[-limit:] if self.processing_history else []
        return [asdict(result) for result in recent_history]


async def test_orchestrator():
    """Test the processing orchestrator"""
    
    # Mock AI processor
    async def mock_ai_processor(prompt: str, context: Dict[str, Any], mode: str) -> Dict[str, Any]:
        await asyncio.sleep(0.1)  # Simulate processing time
        return {
            'response': f"AI response for '{prompt}' in {mode} mode",
            'confidence': 0.85,
            'insights': {'processing_mode': mode, 'context_size': len(context)}
        }
    
    # Initialize orchestrator
    orchestrator = ProcessingOrchestrator(ai_processor=mock_ai_processor)
    
    # Test different types of prompts
    test_prompts = [
        ("What is Python?", {}, "Simple question"),
        ("Debug SSL certificate error in Dunes project", {'project': 'dunes'}, "Debug with context"),
        ("Analyze boundary detection methods across all repositories", {}, "Complex analysis"),
        ("Create comprehensive publication timeline for research themes", {}, "Iterative processing")
    ]
    
    for prompt, context, description in test_prompts:
        print(f"\n=== Testing: {description} ===")
        print(f"Prompt: {prompt}")
        
        result = await orchestrator.process_prompt(prompt, context)
        
        print(f"Route: {result.route_taken.value if result.route_taken else 'None'}")
        print(f"Status: {result.status.value}")
        print(f"Processing time: {result.processing_time:.3f}s")
        print(f"Confidence: {result.confidence:.2f}")
        print(f"Response: {result.response[:100]}...")
        
        if result.recommendations:
            print(f"Recommendations: {len(result.recommendations)}")
    
    # System status
    print(f"\n=== System Status ===")
    status = orchestrator.get_system_status()
    print(json.dumps(status, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(test_orchestrator())