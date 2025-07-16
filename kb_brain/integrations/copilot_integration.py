#!/usr/bin/env python3
"""
GitHub Copilot Enterprise Integration for KB Brain
Uses Windows authentication with requests_negotiate_sspi
"""

import json
import asyncio
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

try:
    import requests
    from requests_negotiate_sspi import HttpNegotiateAuth
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("⚠️  requests_negotiate_sspi not available. Install with: pip install requests-negotiate-sspi")

@dataclass
class CopilotUser:
    """Represents a Copilot user"""
    username: str
    display_name: str
    plan_type: str
    seat_breakdown: Dict[str, Any]
    last_activity_at: Optional[str] = None
    
@dataclass
class CopilotSuggestion:
    """Represents a Copilot suggestion enhanced with KB Brain"""
    suggestion_id: str
    language: str
    context: str
    original_suggestion: str
    enhanced_suggestion: str
    kb_context_used: bool
    confidence: float
    timestamp: str

class GitHubCopilotClient:
    """
    GitHub Copilot Enterprise client using Windows authentication
    
    This client respects corporate authentication and doesn't hammer the auth server.
    """
    
    def __init__(self, enterprise_org: str = "nationalparkservice", 
                 base_url: str = "https://api.github.com"):
        """
        Initialize Copilot client
        
        Args:
            enterprise_org: Your enterprise organization name
            base_url: GitHub API base URL
        """
        self.enterprise_org = enterprise_org
        self.base_url = base_url
        self.session = None
        self.auth_validated = False
        self.user_info = None
        
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests_negotiate_sspi required for Windows authentication")
    
    def _get_session(self) -> requests.Session:
        """Get authenticated session with Windows auth"""
        if self.session is None:
            self.session = requests.Session()
            self.session.auth = HttpNegotiateAuth()
            
            # Set headers
            self.session.headers.update({
                'Accept': 'application/vnd.github+json',
                'X-GitHub-Api-Version': '2022-11-28',
                'User-Agent': 'KB-Brain-Copilot-Integration/1.0'
            })
        
        return self.session
    
    def validate_access(self) -> Dict[str, Any]:
        """
        Validate access to GitHub Copilot without hammering auth server
        
        Returns user info and access status
        """
        if self.auth_validated and self.user_info:
            return self.user_info
        
        try:
            session = self._get_session()
            
            # First, check basic authentication with user endpoint
            response = session.get(f"{self.base_url}/user", timeout=10)
            
            if response.status_code == 401:
                return {
                    "authenticated": False,
                    "error": "Authentication failed - Windows auth not working",
                    "suggestion": "Check if you're logged into Windows with your GitHub enterprise account"
                }
            elif response.status_code != 200:
                return {
                    "authenticated": False,
                    "error": f"API request failed with status {response.status_code}",
                    "response": response.text[:200]
                }
            
            user_data = response.json()
            username = user_data.get('login', 'unknown')
            
            # Now check Copilot access (this is read-only, low impact)
            copilot_url = f"{self.base_url}/orgs/{self.enterprise_org}/copilot/billing"
            copilot_response = session.get(copilot_url, timeout=10)
            
            has_copilot_access = copilot_response.status_code == 200
            
            self.user_info = {
                "authenticated": True,
                "username": username,
                "display_name": user_data.get('name', username),
                "email": user_data.get('email'),
                "has_copilot_access": has_copilot_access,
                "organization": self.enterprise_org,
                "validated_at": datetime.now().isoformat()
            }
            
            if has_copilot_access:
                # Get Copilot subscription details
                copilot_data = copilot_response.json()
                self.user_info["copilot_plan"] = copilot_data.get('plan_type', 'unknown')
                self.user_info["copilot_seats"] = copilot_data.get('total_seats', 0)
            
            self.auth_validated = True
            return self.user_info
            
        except requests.exceptions.Timeout:
            return {
                "authenticated": False,
                "error": "Request timeout - network or auth server issues",
                "suggestion": "Try again later or check network connection"
            }
        except requests.exceptions.ConnectionError:
            return {
                "authenticated": False,
                "error": "Connection error - cannot reach GitHub API",
                "suggestion": "Check internet connection and firewall settings"
            }
        except Exception as e:
            return {
                "authenticated": False,
                "error": f"Unexpected error: {str(e)}",
                "suggestion": "Check error details and try again"
            }
    
    def get_user_copilot_status(self, username: str = None) -> Dict[str, Any]:
        """
        Get Copilot status for a user (rate-limited, use sparingly)
        
        Args:
            username: Username to check (defaults to authenticated user)
        """
        access_info = self.validate_access()
        if not access_info.get("authenticated"):
            return access_info
        
        if not access_info.get("has_copilot_access"):
            return {
                "error": "No Copilot access for this organization",
                "organization": self.enterprise_org
            }
        
        # Use current user if none specified
        if username is None:
            username = access_info.get("username")
        
        try:
            session = self._get_session()
            
            # Check user's Copilot seat
            url = f"{self.base_url}/enterprises/{self.enterprise_org}/members/{username}/copilot"
            response = session.get(url, timeout=10)
            
            if response.status_code == 404:
                return {
                    "user": username,
                    "has_copilot_seat": False,
                    "message": "User not found or no Copilot seat assigned"
                }
            elif response.status_code != 200:
                return {
                    "error": f"API request failed with status {response.status_code}",
                    "user": username
                }
            
            seat_data = response.json()
            
            return {
                "user": username,
                "has_copilot_seat": True,
                "plan_type": seat_data.get('plan_type'),
                "last_activity_at": seat_data.get('last_activity_at'),
                "created_at": seat_data.get('created_at'),
                "updated_at": seat_data.get('updated_at'),
                "seat_breakdown": seat_data.get('seat_breakdown', {})
            }
            
        except Exception as e:
            return {
                "error": f"Error checking Copilot status: {str(e)}",
                "user": username
            }

class KBBrainCopilotIntegration:
    """
    Integration between KB Brain and GitHub Copilot
    
    This provides enhanced code suggestions by combining Copilot with KB Brain's
    organizational knowledge base.
    """
    
    def __init__(self, enterprise_org: str = "nationalparkservice", kb_brain=None):
        """
        Initialize the integration
        
        Args:
            enterprise_org: GitHub enterprise organization
            kb_brain: KB Brain instance (optional)
        """
        self.copilot_client = GitHubCopilotClient(enterprise_org)
        self.kb_brain = kb_brain
        self.integration_cache = {}
        
        # Initialize KB Brain if not provided
        if self.kb_brain is None:
            self.kb_brain = self._init_kb_brain()
    
    def _init_kb_brain(self):
        """Initialize KB Brain if available"""
        try:
            from kb_brain.core.kb_brain_hybrid import HybridGPUKBBrain
            return HybridGPUKBBrain()
        except ImportError:
            print("⚠️  KB Brain not available. Install with: pip install kb-brain")
            return None
    
    def validate_integration(self) -> Dict[str, Any]:
        """Validate that integration is ready to use"""
        
        # Check Copilot access
        copilot_status = self.copilot_client.validate_access()
        
        # Check KB Brain status
        kb_brain_status = {"available": False}
        if self.kb_brain:
            try:
                kb_status = self.kb_brain.get_system_status()
                kb_brain_status = {
                    "available": True,
                    "gpu_available": kb_status.get('hybrid_gpu_status', {}).get('gpu_available', False),
                    "knowledge_embeddings": kb_status.get('hybrid_gpu_status', {}).get('knowledge_embeddings', 0),
                    "total_solutions": kb_status.get('total_solutions', 0)
                }
            except Exception as e:
                kb_brain_status = {
                    "available": False,
                    "error": str(e)
                }
        
        return {
            "copilot_access": copilot_status,
            "kb_brain_status": kb_brain_status,
            "integration_ready": (
                copilot_status.get("authenticated", False) and 
                copilot_status.get("has_copilot_access", False) and
                kb_brain_status.get("available", False)
            ),
            "validated_at": datetime.now().isoformat()
        }
    
    def enhance_code_suggestion(self, 
                              code_context: str, 
                              language: str = "python",
                              file_path: str = None) -> CopilotSuggestion:
        """
        Enhance code suggestions with KB Brain knowledge
        
        This simulates what the integration would do:
        1. Use KB Brain to find relevant organizational patterns
        2. Combine with Copilot's suggestions
        3. Return enhanced suggestion
        
        Args:
            code_context: The code context for suggestion
            language: Programming language
            file_path: Optional file path for context
        """
        
        suggestion_id = f"suggestion_{int(datetime.now().timestamp())}"
        
        # Search KB Brain for relevant patterns
        kb_context = ""
        kb_used = False
        
        if self.kb_brain:
            try:
                # Create search query from code context
                query = f"{language} {code_context}"
                if file_path:
                    query += f" {Path(file_path).stem}"
                
                solutions = self.kb_brain.find_best_solution_hybrid(
                    query, 
                    context={"language": language, "file_path": file_path},
                    top_k=3
                )
                
                if solutions:
                    kb_context = self._format_kb_context(solutions)
                    kb_used = True
                    
            except Exception as e:
                print(f"KB Brain search error: {e}")
        
        # Simulate Copilot suggestion (in real integration, this would call Copilot API)
        original_suggestion = f"# Generated suggestion for {language} code\n{code_context}"
        
        # Enhanced suggestion with KB Brain context
        enhanced_suggestion = original_suggestion
        if kb_used:
            enhanced_suggestion += f"\n\n# Organizational context:\n{kb_context}"
        
        return CopilotSuggestion(
            suggestion_id=suggestion_id,
            language=language,
            context=code_context,
            original_suggestion=original_suggestion,
            enhanced_suggestion=enhanced_suggestion,
            kb_context_used=kb_used,
            confidence=0.85 if kb_used else 0.70,
            timestamp=datetime.now().isoformat()
        )
    
    def _format_kb_context(self, solutions: List[Any]) -> str:
        """Format KB Brain solutions as context for code suggestions"""
        context_parts = []
        
        for solution in solutions[:2]:  # Use top 2 solutions
            context_parts.append(f"- {solution.solution_text[:100]}...")
        
        return "\n".join(context_parts)
    
    def get_integration_config(self) -> Dict[str, Any]:
        """Get configuration for IDE integration"""
        return {
            "enterprise_org": self.copilot_client.enterprise_org,
            "authentication": "Windows SSO",
            "features": {
                "copilot_access": True,
                "kb_brain_enhancement": bool(self.kb_brain),
                "organizational_context": True,
                "code_pattern_matching": True
            },
            "supported_languages": [
                "python", "javascript", "typescript", "java", 
                "go", "rust", "cpp", "c", "html", "css", "sql"
            ]
        }
    
    def create_vscode_config(self) -> Dict[str, Any]:
        """Create VSCode configuration for Copilot + KB Brain"""
        return {
            "github.copilot.enable": {
                "*": True,
                "yaml": True,
                "plaintext": False,
                "markdown": True
            },
            "github.copilot.advanced": {
                "length": 500,
                "temperature": 0.1,
                "top_p": 1,
                "stops": {
                    "*": ["\n\n\n"]
                }
            },
            "kb-brain.integration": {
                "enabled": True,
                "enhanceCopilotSuggestions": True,
                "useOrganizationalContext": True,
                "cacheResults": True
            }
        }

def main():
    """Example usage of GitHub Copilot integration"""
    
    print("🚀 GitHub Copilot + KB Brain Integration")
    print("=" * 50)
    
    # Create integration
    integration = KBBrainCopilotIntegration()
    
    # Validate integration
    print("🔍 Validating integration...")
    status = integration.validate_integration()
    
    print(f"✅ Copilot authenticated: {status['copilot_access'].get('authenticated', False)}")
    print(f"✅ Copilot access: {status['copilot_access'].get('has_copilot_access', False)}")
    print(f"✅ KB Brain available: {status['kb_brain_status'].get('available', False)}")
    print(f"✅ Integration ready: {status['integration_ready']}")
    
    if status['copilot_access'].get('authenticated'):
        user_info = status['copilot_access']
        print(f"\n👤 User: {user_info.get('username')} ({user_info.get('display_name')})")
        print(f"🏢 Organization: {user_info.get('organization')}")
        
        if user_info.get('has_copilot_access'):
            print(f"💻 Copilot plan: {user_info.get('copilot_plan', 'unknown')}")
    
    # Example enhanced suggestion
    if status['integration_ready']:
        print("\n🔄 Testing enhanced code suggestion...")
        suggestion = integration.enhance_code_suggestion(
            code_context="def authenticate_user():",
            language="python",
            file_path="/path/to/auth.py"
        )
        
        print(f"📝 Suggestion ID: {suggestion.suggestion_id}")
        print(f"🎯 KB Context used: {suggestion.kb_context_used}")
        print(f"📊 Confidence: {suggestion.confidence:.2f}")
    
    # Show configuration
    print("\n⚙️ Integration Configuration:")
    config = integration.get_integration_config()
    for key, value in config.items():
        print(f"  {key}: {value}")

if __name__ == "__main__":
    main()