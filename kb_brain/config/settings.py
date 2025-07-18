#!/usr/bin/env python3
"""
KB Brain Configuration Management
Centralized configuration with environment variable support
"""

import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class Settings:
    """Centralized configuration management with environment variable support"""
    
    # Base Configuration
    PROJECT_NAME = "kb-brain"
    VERSION = "1.1.0"
    
    # Environment Detection
    ENVIRONMENT = os.environ.get("KB_BRAIN_ENV", "development")
    
    # Base Paths - Use environment variables with secure defaults
    _home_path = Path.home()
    _project_root = Path(__file__).parent.parent.parent
    
    # Core KB Brain Paths
    KB_SYSTEM_PATH = Path(os.environ.get("KB_SYSTEM_PATH", _home_path / "kb_system"))
    KB_BRAIN_PATH = Path(os.environ.get("KB_BRAIN_PATH", _project_root))
    KB_DATA_PATH = Path(os.environ.get("KB_DATA_PATH", _project_root / "data"))
    
    # Temporary and Working Directories
    TEMP_DIR = Path(os.environ.get("KB_BRAIN_TEMP_DIR", "/tmp/kb_brain"))
    WORK_DIR = Path(os.environ.get("KB_BRAIN_WORK_DIR", _home_path / "kb_brain_work"))
    
    # SSL Configuration (secure defaults)
    SSL_CERT_PATH = Path(os.environ.get("SSL_CERT_PATH", "/etc/ssl/certs/ca-certificates.crt"))
    SSL_CA_BUNDLE = Path(os.environ.get("SSL_CA_BUNDLE", _home_path / "ca-certificates.crt"))
    
    # API Configuration
    API_KEY = os.environ.get("KB_BRAIN_API_KEY", "dev-key-change-in-production")
    API_BASE_URL = os.environ.get("KB_BRAIN_API_BASE", "http://localhost:8080")
    API_TIMEOUT = int(os.environ.get("KB_BRAIN_API_TIMEOUT", "30"))
    
    # Database Configuration
    DATABASE_URL = os.environ.get("KB_BRAIN_DATABASE_URL", "sqlite:///kb_brain.db")
    
    # Performance Configuration
    ENABLE_PERFORMANCE_OPTIMIZATIONS = os.environ.get("KB_BRAIN_PERFORMANCE", "false").lower() == "true"
    MAX_WORKERS = int(os.environ.get("KB_BRAIN_MAX_WORKERS", "4"))
    CACHE_SIZE = int(os.environ.get("KB_BRAIN_CACHE_SIZE", "1000"))
    
    # Search Configuration
    DEFAULT_SEARCH_RESULTS = int(os.environ.get("KB_BRAIN_DEFAULT_RESULTS", "10"))
    MAX_SEARCH_RESULTS = int(os.environ.get("KB_BRAIN_MAX_RESULTS", "100"))
    SIMILARITY_THRESHOLD = float(os.environ.get("KB_BRAIN_SIMILARITY_THRESHOLD", "0.3"))
    
    # Logging Configuration
    LOG_LEVEL = os.environ.get("KB_BRAIN_LOG_LEVEL", "INFO")
    LOG_FILE = os.environ.get("KB_BRAIN_LOG_FILE", "kb_brain.log")
    
    # Organization Configuration (configurable for different deployments)
    ORGANIZATION_NAME = os.environ.get("ORGANIZATION_NAME", "Research Organization")
    ORGANIZATION_EMAIL = os.environ.get("ORGANIZATION_EMAIL", "info@example.org")
    REPOSITORY_URL = os.environ.get("REPOSITORY_URL", "https://github.com/organization/kb-brain")
    
    # Continue Integration Configuration
    CONTINUE_PORT = int(os.environ.get("KB_BRAIN_CONTINUE_PORT", "8080"))
    CONTINUE_HOST = os.environ.get("KB_BRAIN_CONTINUE_HOST", "localhost")
    
    # MCP Configuration
    MCP_PYTHON_PATH = os.environ.get("MCP_PYTHON_PATH", "python3")
    MCP_PACKAGE_PATH = os.environ.get("MCP_PACKAGE_PATH", str(_project_root / "kb_brain" / "mcp" / "server.py"))
    
    @classmethod
    def ensure_directories(cls):
        """Ensure all required directories exist"""
        directories = [
            cls.KB_SYSTEM_PATH,
            cls.KB_DATA_PATH,
            cls.TEMP_DIR,
            cls.WORK_DIR,
        ]
        
        for directory in directories:
            try:
                directory.mkdir(parents=True, exist_ok=True)
                logger.debug(f"Ensured directory exists: {directory}")
            except Exception as e:
                logger.warning(f"Could not create directory {directory}: {e}")
    
    @classmethod
    def get_ssl_config(cls) -> Dict[str, Any]:
        """Get SSL configuration with fallback options"""
        ssl_config = {
            "verify": True,
            "cert_path": None,
            "ca_bundle": None
        }
        
        # Check for SSL certificate files
        if cls.SSL_CERT_PATH.exists():
            ssl_config["cert_path"] = str(cls.SSL_CERT_PATH)
        
        if cls.SSL_CA_BUNDLE.exists():
            ssl_config["ca_bundle"] = str(cls.SSL_CA_BUNDLE)
        
        return ssl_config
    
    @classmethod
    def get_api_config(cls) -> Dict[str, Any]:
        """Get API configuration"""
        return {
            "api_key": cls.API_KEY,
            "base_url": cls.API_BASE_URL,
            "timeout": cls.API_TIMEOUT,
            "verify_ssl": cls.get_ssl_config()
        }
    
    @classmethod
    def get_search_config(cls) -> Dict[str, Any]:
        """Get search configuration"""
        return {
            "default_results": cls.DEFAULT_SEARCH_RESULTS,
            "max_results": cls.MAX_SEARCH_RESULTS,
            "similarity_threshold": cls.SIMILARITY_THRESHOLD,
            "enable_performance": cls.ENABLE_PERFORMANCE_OPTIMIZATIONS
        }
    
    @classmethod
    def load_from_file(cls, config_file: Path) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        try:
            with open(config_file, 'r') as f:
                config_data = json.load(f)
                logger.info(f"Loaded configuration from {config_file}")
                return config_data
        except Exception as e:
            logger.warning(f"Could not load configuration from {config_file}: {e}")
            return {}
    
    @classmethod
    def save_to_file(cls, config_file: Path, config_data: Dict[str, Any]):
        """Save configuration to JSON file"""
        try:
            config_file.parent.mkdir(parents=True, exist_ok=True)
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
                logger.info(f"Saved configuration to {config_file}")
        except Exception as e:
            logger.error(f"Could not save configuration to {config_file}: {e}")
    
    @classmethod
    def get_environment_template(cls) -> str:
        """Get environment variable template for .env file"""
        return '''# KB Brain Configuration
# Copy this file to .env and customize for your environment

# Core Paths
KB_SYSTEM_PATH=${HOME}/kb_system
KB_BRAIN_PATH=${HOME}/kb-brain
KB_DATA_PATH=${HOME}/kb-brain/data

# Working Directories
KB_BRAIN_TEMP_DIR=/tmp/kb_brain
KB_BRAIN_WORK_DIR=${HOME}/kb_brain_work

# SSL Configuration (remove or customize for your network)
SSL_CERT_PATH=/etc/ssl/certs/ca-certificates.crt
SSL_CA_BUNDLE=${HOME}/ca-certificates.crt

# API Configuration
KB_BRAIN_API_KEY=your-secure-api-key-here
KB_BRAIN_API_BASE=http://localhost:8080
KB_BRAIN_API_TIMEOUT=30

# Performance Configuration
KB_BRAIN_PERFORMANCE=false
KB_BRAIN_MAX_WORKERS=4
KB_BRAIN_CACHE_SIZE=1000

# Search Configuration
KB_BRAIN_DEFAULT_RESULTS=10
KB_BRAIN_MAX_RESULTS=100
KB_BRAIN_SIMILARITY_THRESHOLD=0.3

# Logging Configuration
KB_BRAIN_LOG_LEVEL=INFO
KB_BRAIN_LOG_FILE=kb_brain.log

# Organization Configuration
ORGANIZATION_NAME=Your Organization Name
ORGANIZATION_EMAIL=info@yourorg.com
REPOSITORY_URL=https://github.com/yourorg/kb-brain

# Continue Integration
KB_BRAIN_CONTINUE_PORT=8080
KB_BRAIN_CONTINUE_HOST=localhost

# MCP Configuration
MCP_PYTHON_PATH=python3
MCP_PACKAGE_PATH=/path/to/kb_brain/mcp/server.py

# Environment
KB_BRAIN_ENV=development
'''
    
    @classmethod
    def validate_config(cls) -> Dict[str, Any]:
        """Validate current configuration and return status"""
        validation_result = {
            "valid": True,
            "warnings": [],
            "errors": [],
            "info": []
        }
        
        # Check required directories
        if not cls.KB_SYSTEM_PATH.exists():
            validation_result["warnings"].append(f"KB system path does not exist: {cls.KB_SYSTEM_PATH}")
        
        # Check SSL configuration
        ssl_config = cls.get_ssl_config()
        if not ssl_config["cert_path"] and not ssl_config["ca_bundle"]:
            validation_result["info"].append("No SSL certificates configured - using system defaults")
        
        # Check API key
        if cls.API_KEY == "dev-key-change-in-production":
            validation_result["warnings"].append("Using default API key - should be changed for production")
        
        # Check paths
        if not cls.KB_BRAIN_PATH.exists():
            validation_result["errors"].append(f"KB Brain path does not exist: {cls.KB_BRAIN_PATH}")
            validation_result["valid"] = False
        
        return validation_result

# Initialize directories on import
Settings.ensure_directories()

# Export commonly used settings
KB_SYSTEM_PATH = Settings.KB_SYSTEM_PATH
KB_BRAIN_PATH = Settings.KB_BRAIN_PATH
KB_DATA_PATH = Settings.KB_DATA_PATH
API_CONFIG = Settings.get_api_config()
SEARCH_CONFIG = Settings.get_search_config()
SSL_CONFIG = Settings.get_ssl_config()