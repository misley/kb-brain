"""
Intelligent Prompt Classifier for KB Brain
Analyzes prompts to determine intent, complexity, and processing requirements
"""

import re
import json
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


class PromptIntent(Enum):
    """Classification of prompt intent types"""
    SIMPLE_QUESTION = "simple_question"
    CODE_REQUEST = "code_request"  
    DEBUG_HELP = "debug_help"
    PROJECT_STATUS = "project_status"
    RESEARCH_ANALYSIS = "research_analysis"
    SYSTEM_INTEGRATION = "system_integration"
    KNOWLEDGE_LOOKUP = "knowledge_lookup"
    COMPLEX_ANALYSIS = "complex_analysis"
    ITERATIVE_PROCESSING = "iterative_processing"


class ComplexityLevel(Enum):
    """Processing complexity assessment"""
    SIMPLE = 1      # Direct KB lookup or simple response
    MODERATE = 2    # KB search + basic analysis
    COMPLEX = 3     # Multi-stage processing required
    ITERATIVE = 4   # Screen workers + multiple AI rounds


class Domain(Enum):
    """Knowledge domain classification"""
    GENERAL = "general"
    PROGRAMMING = "programming"
    DATA_SCIENCE = "data_science"
    RESEARCH = "research"
    SYSTEM_ADMIN = "system_admin"
    GEOSPATIAL = "geospatial"
    ENVIRONMENTAL = "environmental"
    PROJECT_MANAGEMENT = "project_management"


@dataclass
class PromptClassification:
    """Complete prompt analysis result"""
    intent: PromptIntent
    complexity: ComplexityLevel
    domain: Domain
    confidence: float
    keywords: List[str]
    requires_kb_search: bool
    requires_cross_repo: bool
    requires_iterative: bool
    estimated_processing_time: int  # seconds
    suggested_workers: List[str]
    context_hints: Dict[str, Any]


class PromptClassifier:
    """Intelligent prompt classification and routing system"""
    
    def __init__(self, kb_brain=None):
        self.kb_brain = kb_brain
        self._load_classification_rules()
    
    def _load_classification_rules(self):
        """Load classification patterns and rules"""
        
        # Intent recognition patterns
        self.intent_patterns = {
            PromptIntent.SIMPLE_QUESTION: [
                r'\bwhat is\b', r'\bhow to\b', r'\bwhen did\b', r'\bwhere is\b',
                r'\bdefine\b', r'\bexplain\b'
            ],
            PromptIntent.CODE_REQUEST: [
                r'\bwrite code\b', r'\bimplement\b', r'\bcreate function\b',
                r'\bgenerate\b', r'\bbuild\b', r'\bcode for\b'
            ],
            PromptIntent.DEBUG_HELP: [
                r'\berror\b', r'\bbug\b', r'\bfailing\b', r'\bbroken\b',
                r'\bdebug\b', r'\btroubleshoot\b', r'\bfix\b'
            ],
            PromptIntent.PROJECT_STATUS: [
                r'\bproject status\b', r'\btodos\b', r'\bnext steps\b',
                r'\bwhat\'s next\b', r'\bcurrent work\b', r'\bprogress\b'
            ],
            PromptIntent.RESEARCH_ANALYSIS: [
                r'\banalyz\w+\b', r'\bresearch\b', r'\bstudy\b', r'\binvestigat\w+\b',
                r'\bcompare\b', r'\bevaluat\w+\b', r'\bassess\b'
            ],
            PromptIntent.SYSTEM_INTEGRATION: [
                r'\bintegrat\w+\b', r'\bconnect\b', r'\bsetup\b', r'\bconfigur\w+\b',
                r'\binstall\b', r'\bdeploy\b'
            ],
            PromptIntent.KNOWLEDGE_LOOKUP: [
                r'\bfind\b', r'\bsearch\b', r'\blookup\b', r'\blocate\b',
                r'\bknowledge\b', r'\bremember\b'
            ]
        }
        
        # Complexity indicators
        self.complexity_indicators = {
            ComplexityLevel.SIMPLE: [
                r'\bquick\b', r'\bsimple\b', r'\bbasic\b', r'\bjust\b'
            ],
            ComplexityLevel.MODERATE: [
                r'\banalyz\w+\b', r'\bcompare\b', r'\bfind\b', r'\bsearch\b'
            ],
            ComplexityLevel.COMPLEX: [
                r'\bcomprehensive\b', r'\bdetailed\b', r'\bcomplete\b',
                r'\bmulti\b', r'\bcross\b', r'\badvanced\b'
            ],
            ComplexityLevel.ITERATIVE: [
                r'\biterative\b', r'\bmultiple rounds\b', r'\bstep by step\b',
                r'\bprocess\b.*\banalyze\b', r'\blong.?running\b'
            ]
        }
        
        # Domain keywords
        self.domain_keywords = {
            Domain.PROGRAMMING: [
                'code', 'function', 'class', 'variable', 'import', 'library',
                'python', 'javascript', 'api', 'framework', 'debug'
            ],
            Domain.DATA_SCIENCE: [
                'data', 'analysis', 'dataset', 'model', 'machine learning',
                'pandas', 'numpy', 'scikit', 'jupyter', 'notebook'
            ],
            Domain.RESEARCH: [
                'research', 'study', 'analysis', 'paper', 'publication',
                'findings', 'results', 'methodology', 'validation'
            ],
            Domain.GEOSPATIAL: [
                'gis', 'spatial', 'landsat', 'satellite', 'boundary',
                'ndwi', 'remote sensing', 'geospatial', 'coordinates'
            ],
            Domain.ENVIRONMENTAL: [
                'environmental', 'monitoring', 'ecosystem', 'water',
                'gypsum', 'dunes', 'whsa', 'national park'
            ],
            Domain.SYSTEM_ADMIN: [
                'server', 'install', 'configuration', 'network', 'ssl',
                'certificate', 'permissions', 'deployment'
            ]
        }
        
        # Project context hints
        self.project_contexts = {
            'dunes': ['dunes', 'whsa', 'white sands', 'gypsum', 'boundary'],
            'kb_brain': ['kb brain', 'knowledge base', 'mcp', 'gpu'],
            'research': ['publication', 'paper', 'analysis', 'findings']
        }
    
    def classify_prompt(self, prompt: str, context: Optional[Dict] = None) -> PromptClassification:
        """
        Classify a prompt for intelligent processing routing
        
        Args:
            prompt: The input prompt to classify
            context: Optional context information (file path, project, etc.)
        
        Returns:
            PromptClassification with complete analysis
        """
        prompt_lower = prompt.lower()
        
        # Determine intent
        intent = self._classify_intent(prompt_lower)
        
        # Assess complexity
        complexity = self._assess_complexity(prompt_lower, intent)
        
        # Identify domain
        domain = self._identify_domain(prompt_lower, context)
        
        # Extract keywords
        keywords = self._extract_keywords(prompt_lower)
        
        # Determine processing requirements
        requires_kb_search = self._requires_kb_search(intent, keywords)
        requires_cross_repo = self._requires_cross_repo(prompt_lower, domain)
        requires_iterative = self._requires_iterative(complexity, prompt_lower)
        
        # Estimate processing time
        processing_time = self._estimate_processing_time(complexity, requires_iterative)
        
        # Suggest workers
        suggested_workers = self._suggest_workers(complexity, domain, requires_iterative)
        
        # Generate context hints
        context_hints = self._generate_context_hints(prompt_lower, domain, context)
        
        # Calculate confidence
        confidence = self._calculate_confidence(intent, complexity, domain)
        
        return PromptClassification(
            intent=intent,
            complexity=complexity,
            domain=domain,
            confidence=confidence,
            keywords=keywords,
            requires_kb_search=requires_kb_search,
            requires_cross_repo=requires_cross_repo,
            requires_iterative=requires_iterative,
            estimated_processing_time=processing_time,
            suggested_workers=suggested_workers,
            context_hints=context_hints
        )
    
    def _classify_intent(self, prompt: str) -> PromptIntent:
        """Classify the primary intent of the prompt"""
        scores = {}
        
        for intent, patterns in self.intent_patterns.items():
            score = sum(1 for pattern in patterns if re.search(pattern, prompt))
            if score > 0:
                scores[intent] = score
        
        if not scores:
            # Default classification based on prompt characteristics
            if any(word in prompt for word in ['analyze', 'research', 'study']):
                return PromptIntent.RESEARCH_ANALYSIS
            elif any(word in prompt for word in ['code', 'implement', 'build']):
                return PromptIntent.CODE_REQUEST
            else:
                return PromptIntent.SIMPLE_QUESTION
        
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _assess_complexity(self, prompt: str, intent: PromptIntent) -> ComplexityLevel:
        """Assess the processing complexity required"""
        
        # Check for explicit complexity indicators
        for level, indicators in self.complexity_indicators.items():
            if any(re.search(indicator, prompt) for indicator in indicators):
                return level
        
        # Intent-based complexity
        if intent in [PromptIntent.SIMPLE_QUESTION, PromptIntent.KNOWLEDGE_LOOKUP]:
            return ComplexityLevel.SIMPLE
        elif intent in [PromptIntent.DEBUG_HELP, PromptIntent.CODE_REQUEST]:
            return ComplexityLevel.MODERATE
        elif intent in [PromptIntent.RESEARCH_ANALYSIS, PromptIntent.PROJECT_STATUS]:
            return ComplexityLevel.COMPLEX
        else:
            return ComplexityLevel.MODERATE
    
    def _identify_domain(self, prompt: str, context: Optional[Dict]) -> Domain:
        """Identify the knowledge domain"""
        
        # Context-based domain hints
        if context:
            if 'file_path' in context:
                file_path = context['file_path'].lower()
                if 'dunes' in file_path:
                    return Domain.GEOSPATIAL
                elif 'kb' in file_path:
                    return Domain.PROGRAMMING
        
        # Keyword-based domain classification
        domain_scores = {}
        for domain, keywords in self.domain_keywords.items():
            score = sum(1 for keyword in keywords if keyword in prompt)
            if score > 0:
                domain_scores[domain] = score
        
        if domain_scores:
            return max(domain_scores.items(), key=lambda x: x[1])[0]
        
        return Domain.GENERAL
    
    def _extract_keywords(self, prompt: str) -> List[str]:
        """Extract key terms from the prompt"""
        
        # Technical terms and domain-specific keywords
        technical_terms = []
        all_keywords = []
        
        for domain_keywords in self.domain_keywords.values():
            all_keywords.extend(domain_keywords)
        
        for keyword in all_keywords:
            if keyword in prompt:
                technical_terms.append(keyword)
        
        # Add important non-technical words
        important_words = re.findall(r'\b[a-zA-Z]{4,}\b', prompt)
        technical_terms.extend([word for word in important_words[:10] if word not in technical_terms])
        
        return technical_terms[:15]  # Limit to most relevant terms
    
    def _requires_kb_search(self, intent: PromptIntent, keywords: List[str]) -> bool:
        """Determine if KB search is required"""
        
        kb_intents = [
            PromptIntent.KNOWLEDGE_LOOKUP,
            PromptIntent.PROJECT_STATUS,
            PromptIntent.DEBUG_HELP,
            PromptIntent.RESEARCH_ANALYSIS
        ]
        
        return intent in kb_intents or any(kw in ['project', 'status', 'problem', 'solution'] for kw in keywords)
    
    def _requires_cross_repo(self, prompt: str, domain: Domain) -> bool:
        """Determine if cross-repository analysis is needed"""
        
        cross_repo_indicators = [
            'cross', 'multiple projects', 'repositories', 'compare projects',
            'dunes', 'whsa', 'comprehensive analysis'
        ]
        
        return any(indicator in prompt for indicator in cross_repo_indicators) or \
               domain in [Domain.RESEARCH, Domain.GEOSPATIAL]
    
    def _requires_iterative(self, complexity: ComplexityLevel, prompt: str) -> bool:
        """Determine if iterative processing is needed"""
        
        iterative_indicators = [
            'step by step', 'iterative', 'multiple rounds', 'comprehensive',
            'detailed analysis', 'long running', 'background'
        ]
        
        return complexity == ComplexityLevel.ITERATIVE or \
               any(indicator in prompt for indicator in iterative_indicators)
    
    def _estimate_processing_time(self, complexity: ComplexityLevel, iterative: bool) -> int:
        """Estimate processing time in seconds"""
        
        base_times = {
            ComplexityLevel.SIMPLE: 5,
            ComplexityLevel.MODERATE: 30,
            ComplexityLevel.COMPLEX: 120,
            ComplexityLevel.ITERATIVE: 300
        }
        
        time = base_times[complexity]
        if iterative:
            time *= 3
        
        return time
    
    def _suggest_workers(self, complexity: ComplexityLevel, domain: Domain, iterative: bool) -> List[str]:
        """Suggest appropriate worker types"""
        
        workers = []
        
        if complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ITERATIVE]:
            workers.append('analysis_worker')
        
        if iterative:
            workers.extend(['progress_monitor', 'log_monitor'])
        
        if domain == Domain.GEOSPATIAL:
            workers.append('geospatial_worker')
        elif domain == Domain.DATA_SCIENCE:
            workers.append('data_analysis_worker')
        
        return workers or ['general_worker']
    
    def _generate_context_hints(self, prompt: str, domain: Domain, context: Optional[Dict]) -> Dict[str, Any]:
        """Generate processing context hints"""
        
        hints = {
            'domain': domain.value,
            'detected_projects': []
        }
        
        # Detect project contexts
        for project, indicators in self.project_contexts.items():
            if any(indicator in prompt for indicator in indicators):
                hints['detected_projects'].append(project)
        
        # Add file context if available
        if context and 'file_path' in context:
            hints['file_context'] = context['file_path']
        
        # Add urgency indicators
        urgency_words = ['urgent', 'asap', 'immediately', 'quickly', 'now']
        hints['urgency'] = any(word in prompt for word in urgency_words)
        
        return hints
    
    def _calculate_confidence(self, intent: PromptIntent, complexity: ComplexityLevel, domain: Domain) -> float:
        """Calculate classification confidence score"""
        
        # Base confidence is high for clear patterns
        base_confidence = 0.8
        
        # Adjust based on classification certainty
        if intent == PromptIntent.SIMPLE_QUESTION and complexity == ComplexityLevel.SIMPLE:
            return min(0.95, base_confidence + 0.15)
        elif domain != Domain.GENERAL:
            return min(0.9, base_confidence + 0.1)
        else:
            return base_confidence
    
    def get_processing_recommendations(self, classification: PromptClassification) -> Dict[str, Any]:
        """Get specific processing recommendations based on classification"""
        
        recommendations = {
            'route_to': 'direct_response',
            'kb_search_depth': 'shallow',
            'use_screen_workers': False,
            'expected_iterations': 1,
            'priority': 'normal'
        }
        
        # Routing decisions
        if classification.requires_iterative:
            recommendations['route_to'] = 'screen_processing'
            recommendations['use_screen_workers'] = True
            recommendations['expected_iterations'] = 3
        elif classification.requires_kb_search:
            recommendations['route_to'] = 'kb_enhanced_processing'
            
        # KB search depth
        if classification.complexity in [ComplexityLevel.COMPLEX, ComplexityLevel.ITERATIVE]:
            recommendations['kb_search_depth'] = 'deep'
        elif classification.requires_cross_repo:
            recommendations['kb_search_depth'] = 'cross_repo'
            
        # Priority
        if classification.context_hints.get('urgency', False):
            recommendations['priority'] = 'high'
        elif classification.complexity == ComplexityLevel.ITERATIVE:
            recommendations['priority'] = 'low'  # Can run in background
            
        return recommendations


def test_classifier():
    """Test the prompt classifier with sample prompts"""
    
    classifier = PromptClassifier()
    
    test_prompts = [
        "What is the status of the Dunes project?",
        "Help me debug this SSL certificate error",
        "Implement a GPU-accelerated similarity search function",
        "Analyze the boundary detection methods across all WHSA repositories",
        "Quick question: how do I import pandas?",
        "Run a comprehensive analysis of the research themes and create a publication timeline"
    ]
    
    for prompt in test_prompts:
        classification = classifier.classify_prompt(prompt)
        recommendations = classifier.get_processing_recommendations(classification)
        
        print(f"\nPrompt: {prompt}")
        print(f"Intent: {classification.intent.value}")
        print(f"Complexity: {classification.complexity.value}")
        print(f"Domain: {classification.domain.value}")
        print(f"KB Search: {classification.requires_kb_search}")
        print(f"Iterative: {classification.requires_iterative}")
        print(f"Route to: {recommendations['route_to']}")
        print(f"Confidence: {classification.confidence:.2f}")


if __name__ == "__main__":
    test_classifier()