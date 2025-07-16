#!/usr/bin/env python3
"""
KB Brain Continue Extension Adapter
Provides integration with the Continue VSCode extension
"""

import json
import asyncio
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import re

@dataclass
class ContinueContext:
    """Context information from Continue extension"""
    file_path: str
    language: str
    code_snippet: str
    cursor_position: int
    selected_text: str
    project_root: str
    error_message: Optional[str] = None
    intent: Optional[str] = None  # "completion", "explanation", "debug", "refactor"

@dataclass
class ContinueResponse:
    """Response formatted for Continue extension"""
    suggestions: List[str]
    explanations: List[str]
    confidence: float
    source: str
    context_used: bool
    kb_matches: int

class ContinueAdapter:
    """Adapter for Continue extension integration"""
    
    def __init__(self, kb_brain=None):
        """Initialize Continue adapter with KB Brain instance"""
        if kb_brain is None:
            # Import here to avoid circular imports
            try:
                from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
                self.kb_brain = HybridGPUKBBrain()
            except ImportError:
                raise ImportError("KB Brain not available. Install with: pip install kb-brain")
        else:
            self.kb_brain = kb_brain
            
        self.context_cache = {}
        self.response_cache = {}
    
    async def process_continue_request(self, request: Dict[str, Any]) -> ContinueResponse:
        """Process request from Continue extension"""
        
        # Parse Continue request
        context = self._parse_continue_context(request)
        
        # Check cache first
        cache_key = self._generate_cache_key(context)
        if cache_key in self.response_cache:
            return self.response_cache[cache_key]
        
        # Generate KB Brain query from context
        query = self._generate_kb_query(context)
        
        # Search KB Brain
        solutions = await self._search_kb_brain(query, context)
        
        # Format response for Continue
        response = self._format_continue_response(solutions, context)
        
        # Cache response
        self.response_cache[cache_key] = response
        
        return response
    
    def _parse_continue_context(self, request: Dict[str, Any]) -> ContinueContext:
        """Parse Continue extension request into context"""
        
        return ContinueContext(
            file_path=request.get('filePath', ''),
            language=request.get('language', 'python'),
            code_snippet=request.get('codeSnippet', ''),
            cursor_position=request.get('cursorPosition', 0),
            selected_text=request.get('selectedText', ''),
            project_root=request.get('projectRoot', ''),
            error_message=request.get('errorMessage'),
            intent=request.get('intent', 'completion')
        )
    
    def _generate_kb_query(self, context: ContinueContext) -> str:
        """Generate KB Brain query from Continue context"""
        
        query_parts = []
        
        # Add language context
        if context.language:
            query_parts.append(f"{context.language} programming")
        
        # Add intent-specific context
        if context.intent == "debug" and context.error_message:
            query_parts.append(f"error: {context.error_message}")
        elif context.intent == "completion":
            query_parts.append("code completion")
        elif context.intent == "explanation":
            query_parts.append("code explanation")
        elif context.intent == "refactor":
            query_parts.append("code refactoring")
        
        # Add code context
        if context.selected_text:
            query_parts.append(f"code: {context.selected_text[:100]}")
        elif context.code_snippet:
            query_parts.append(f"code: {context.code_snippet[:100]}")
        
        # Add file context
        if context.file_path:
            file_type = Path(context.file_path).suffix
            if file_type:
                query_parts.append(f"file type: {file_type}")
        
        return " ".join(query_parts)
    
    async def _search_kb_brain(self, query: str, context: ContinueContext) -> List[Any]:
        """Search KB Brain with context-aware query"""
        
        # Create context dictionary for KB Brain
        kb_context = {
            'language': context.language,
            'intent': context.intent,
            'file_path': context.file_path,
            'project_root': context.project_root
        }
        
        # Search KB Brain
        try:
            solutions = self.kb_brain.find_best_solution_hybrid(
                query, 
                context=kb_context,
                top_k=5
            )
            return solutions
        except Exception as e:
            print(f"KB Brain search error: {e}")
            return []
    
    def _format_continue_response(self, solutions: List[Any], context: ContinueContext) -> ContinueResponse:
        """Format KB Brain solutions for Continue extension"""
        
        suggestions = []
        explanations = []
        total_confidence = 0
        
        for solution in solutions:
            # Extract suggestions based on intent
            if context.intent == "completion":
                suggestions.append(self._extract_code_suggestion(solution))
            elif context.intent == "explanation":
                explanations.append(self._extract_explanation(solution))
            elif context.intent == "debug":
                suggestions.append(self._extract_debug_suggestion(solution))
            elif context.intent == "refactor":
                suggestions.append(self._extract_refactor_suggestion(solution))
            
            total_confidence += solution.confidence
        
        avg_confidence = total_confidence / len(solutions) if solutions else 0
        
        return ContinueResponse(
            suggestions=suggestions,
            explanations=explanations,
            confidence=avg_confidence,
            source="KB Brain",
            context_used=True,
            kb_matches=len(solutions)
        )
    
    def _extract_code_suggestion(self, solution: Any) -> str:
        """Extract code suggestion from KB Brain solution"""
        
        # Look for code patterns in solution text
        code_patterns = [
            r'```[\w]*\n(.*?)\n```',  # Code blocks
            r'`([^`]+)`',  # Inline code
            r'def\s+\w+\([^)]*\):',  # Function definitions
            r'class\s+\w+\([^)]*\):',  # Class definitions
        ]
        
        solution_text = solution.solution_text
        
        for pattern in code_patterns:
            matches = re.findall(pattern, solution_text, re.DOTALL)
            if matches:
                return matches[0].strip()
        
        # If no code pattern found, return relevant text
        return solution_text[:200] + "..." if len(solution_text) > 200 else solution_text
    
    def _extract_explanation(self, solution: Any) -> str:
        """Extract explanation from KB Brain solution"""
        
        # Look for explanation patterns
        explanation_patterns = [
            r'explanation[:\s]+(.*?)(?:\n|$)',
            r'this\s+(?:means|does|is)\s+(.*?)(?:\n|$)',
            r'(?:because|since|as)\s+(.*?)(?:\n|$)',
        ]
        
        solution_text = solution.solution_text.lower()
        
        for pattern in explanation_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return solution.solution_text[:300] + "..." if len(solution.solution_text) > 300 else solution.solution_text
    
    def _extract_debug_suggestion(self, solution: Any) -> str:
        """Extract debugging suggestion from KB Brain solution"""
        
        # Look for debug-related patterns
        debug_patterns = [
            r'(?:fix|solution|resolve)[:\s]+(.*?)(?:\n|$)',
            r'(?:try|use|change)[:\s]+(.*?)(?:\n|$)',
            r'(?:error|issue|problem)[:\s]+(.*?)(?:\n|$)',
        ]
        
        solution_text = solution.solution_text.lower()
        
        for pattern in debug_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return solution.solution_text[:200] + "..." if len(solution.solution_text) > 200 else solution.solution_text
    
    def _extract_refactor_suggestion(self, solution: Any) -> str:
        """Extract refactoring suggestion from KB Brain solution"""
        
        # Look for refactoring patterns
        refactor_patterns = [
            r'(?:refactor|improve|optimize)[:\s]+(.*?)(?:\n|$)',
            r'(?:better|cleaner|more efficient)[:\s]+(.*?)(?:\n|$)',
            r'(?:consider|suggestion|recommendation)[:\s]+(.*?)(?:\n|$)',
        ]
        
        solution_text = solution.solution_text.lower()
        
        for pattern in refactor_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                return matches[0].strip()
        
        return solution.solution_text[:200] + "..." if len(solution.solution_text) > 200 else solution.solution_text
    
    def _generate_cache_key(self, context: ContinueContext) -> str:
        """Generate cache key for context"""
        
        key_parts = [
            context.language,
            context.intent,
            context.selected_text[:50] if context.selected_text else "",
            context.error_message[:50] if context.error_message else "",
        ]
        
        return "|".join(key_parts)
    
    def get_continue_config(self) -> Dict[str, Any]:
        """Get Continue extension configuration for KB Brain"""
        
        return {
            "models": [
                {
                    "title": "KB Brain",
                    "provider": "custom",
                    "model": "kb-brain-hybrid",
                    "apiBase": "http://localhost:8080/kb-brain",
                    "apiKey": "kb-brain-api-key",
                    "systemMessage": "You are an AI assistant with access to a knowledge base. Use the KB Brain system to find relevant solutions and context for coding problems.",
                }
            ],
            "customCommands": [
                {
                    "name": "kb-search",
                    "description": "Search KB Brain knowledge base",
                    "prompt": "Search the knowledge base for: {input}"
                },
                {
                    "name": "kb-debug",
                    "description": "Find debugging solutions in KB Brain",
                    "prompt": "Find debugging solutions for this error: {input}"
                },
                {
                    "name": "kb-explain",
                    "description": "Get explanation from KB Brain",
                    "prompt": "Explain this code using knowledge base: {input}"
                }
            ],
            "contextProviders": [
                {
                    "name": "kb-brain",
                    "description": "KB Brain Knowledge Base",
                    "type": "custom",
                    "config": {
                        "serverUrl": "http://localhost:8080/kb-brain",
                        "apiKey": "kb-brain-api-key"
                    }
                }
            ]
        }
    
    async def start_continue_server(self, port: int = 8080):
        """Start HTTP server for Continue extension integration"""
        
        from aiohttp import web
        
        async def handle_search(request):
            """Handle search requests from Continue"""
            
            try:
                data = await request.json()
                context = self._parse_continue_context(data)
                response = await self.process_continue_request(data)
                
                return web.json_response(asdict(response))
            
            except Exception as e:
                return web.json_response({
                    "error": str(e),
                    "suggestions": [],
                    "explanations": [],
                    "confidence": 0,
                    "source": "KB Brain",
                    "context_used": False,
                    "kb_matches": 0
                })
        
        async def handle_status(request):
            """Handle status requests"""
            
            status = self.kb_brain.get_system_status()
            return web.json_response({
                "status": "online",
                "kb_brain_status": status,
                "continue_adapter": "ready"
            })
        
        app = web.Application()
        app.router.add_post('/kb-brain/search', handle_search)
        app.router.add_get('/kb-brain/status', handle_status)
        
        print(f"🔌 Starting KB Brain Continue server on port {port}")
        
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, 'localhost', port)
        await site.start()
        
        print(f"✅ KB Brain Continue server running on http://localhost:{port}")
        print(f"📋 Configure Continue extension with: {self.get_continue_config()}")
        
        # Keep server running
        try:
            await asyncio.Event().wait()
        except KeyboardInterrupt:
            print("🛑 Shutting down KB Brain Continue server")
            await runner.cleanup()

def main():
    """Main entry point for Continue adapter"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="KB Brain Continue Extension Adapter")
    parser.add_argument('--port', type=int, default=8080, help='Server port')
    parser.add_argument('--config', action='store_true', help='Print Continue configuration')
    
    args = parser.parse_args()
    
    adapter = ContinueAdapter()
    
    if args.config:
        print("Continue Extension Configuration:")
        print(json.dumps(adapter.get_continue_config(), indent=2))
    else:
        asyncio.run(adapter.start_continue_server(args.port))

if __name__ == "__main__":
    main()