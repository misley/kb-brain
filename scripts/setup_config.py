#!/usr/bin/env python3
"""
KB Brain Configuration Setup Script
Helps users create secure configuration files with proper defaults
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, Any, Optional

# Add kb_brain to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from kb_brain.config.settings import Settings

def create_environment_file():
    """Create .env file from template"""
    print("🔧 Creating KB Brain environment configuration")
    print("=" * 50)
    
    project_root = Path(__file__).parent.parent
    env_template = project_root / ".env.template"
    env_file = project_root / ".env"
    
    if env_file.exists():
        response = input(f"❓ .env file already exists. Overwrite? (y/N): ")
        if response.lower() != 'y':
            print("⏭️  Skipping .env file creation")
            return
    
    if not env_template.exists():
        print("❌ .env.template not found")
        return
    
    # Copy template to .env
    shutil.copy2(env_template, env_file)
    print(f"✅ Created .env file: {env_file}")
    
    # Customize with user input
    print("\n📝 Let's customize your configuration:")
    
    # Get user preferences
    config_updates = {}
    
    # Organization info
    org_name = input("Organization name [Research Organization]: ").strip()
    if org_name:
        config_updates["ORGANIZATION_NAME"] = org_name
    
    org_email = input("Organization email [info@example.org]: ").strip()
    if org_email:
        config_updates["ORGANIZATION_EMAIL"] = org_email
    
    # Performance settings
    enable_perf = input("Enable performance optimizations? (y/N): ").strip().lower()
    if enable_perf == 'y':
        config_updates["KB_BRAIN_PERFORMANCE"] = "true"
    
    # API settings
    api_key = input("API key (leave empty for default): ").strip()
    if api_key:
        config_updates["KB_BRAIN_API_KEY"] = api_key
    
    # SSL settings
    ssl_cert = input("SSL certificate path (leave empty for system default): ").strip()
    if ssl_cert:
        config_updates["SSL_CERT_PATH"] = ssl_cert
    
    # Update .env file
    if config_updates:
        update_env_file(env_file, config_updates)
        print("✅ Updated .env file with your preferences")
    
    print(f"\n📁 Configuration file created: {env_file}")
    print("🔒 This file contains sensitive information - it's already in .gitignore")

def update_env_file(env_file: Path, updates: Dict[str, str]):
    """Update .env file with new values"""
    with open(env_file, 'r') as f:
        lines = f.readlines()
    
    updated_lines = []
    for line in lines:
        line = line.strip()
        
        # Skip empty lines and comments
        if not line or line.startswith('#'):
            updated_lines.append(line + '\n')
            continue
        
        # Check if this is a variable we want to update
        if '=' in line:
            key = line.split('=')[0]
            if key in updates:
                updated_lines.append(f"{key}={updates[key]}\n")
                continue
        
        updated_lines.append(line + '\n')
    
    with open(env_file, 'w') as f:
        f.writelines(updated_lines)

def generate_config_files():
    """Generate configuration files with environment variable substitution"""
    print("\n🔧 Generating configuration files")
    print("=" * 40)
    
    project_root = Path(__file__).parent.parent
    
    # Generate Continue config
    continue_config = generate_continue_config()
    continue_config_file = project_root / "continue_integration" / "continue_config.json"
    
    with open(continue_config_file, 'w') as f:
        json.dump(continue_config, f, indent=2)
    print(f"✅ Generated Continue config: {continue_config_file}")
    
    # Generate MCP config
    mcp_config = generate_mcp_config()
    mcp_config_file = project_root / "config" / "mcp_config.json"
    
    with open(mcp_config_file, 'w') as f:
        json.dump(mcp_config, f, indent=2)
    print(f"✅ Generated MCP config: {mcp_config_file}")

def generate_continue_config() -> Dict[str, Any]:
    """Generate Continue configuration with environment variables"""
    return {
        "models": [
            {
                "title": "KB Brain",
                "provider": "custom",
                "model": "kb-brain-hybrid",
                "apiBase": "${KB_BRAIN_API_BASE}/kb-brain",
                "apiKey": "${KB_BRAIN_API_KEY}",
                "systemMessage": "You are an AI assistant with access to a knowledge base. Use the KB Brain system to find relevant solutions and context for coding problems."
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
                "description": "Find debugging solutions",
                "prompt": "Find debugging solutions for: {input}"
            }
        ],
        "contextProviders": [
            {
                "name": "kb-brain",
                "description": "KB Brain Knowledge Base",
                "type": "custom",
                "config": {
                    "serverUrl": "${KB_BRAIN_API_BASE}/kb-brain",
                    "apiKey": "${KB_BRAIN_API_KEY}",
                    "timeout": 5000,
                    "maxSuggestions": 5
                }
            }
        ],
        "allowAnonymousTelemetry": False,
        "enableExperimentalFeatures": True
    }

def generate_mcp_config() -> Dict[str, Any]:
    """Generate MCP configuration with environment variables"""
    return {
        "mcpServers": {
            "kb-brain": {
                "command": "${MCP_PYTHON_PATH}",
                "args": ["${MCP_PACKAGE_PATH}"],
                "env": {
                    "KB_DATA_PATH": "${KB_DATA_PATH}"
                }
            }
        }
    }

def validate_configuration():
    """Validate the current configuration"""
    print("\n🔍 Validating configuration")
    print("=" * 30)
    
    # Load settings
    validation_result = Settings.validate_config()
    
    if validation_result["valid"]:
        print("✅ Configuration is valid")
    else:
        print("❌ Configuration has errors")
        for error in validation_result["errors"]:
            print(f"   Error: {error}")
    
    if validation_result["warnings"]:
        print("\n⚠️  Warnings:")
        for warning in validation_result["warnings"]:
            print(f"   {warning}")
    
    if validation_result["info"]:
        print("\nℹ️  Information:")
        for info in validation_result["info"]:
            print(f"   {info}")
    
    # Check SSL configuration
    print("\n🔒 SSL Configuration:")
    ssl_config = Settings.get_ssl_config()
    
    if ssl_config["cert_path"]:
        cert_path = Path(ssl_config["cert_path"])
        if cert_path.exists():
            print(f"   ✅ SSL certificate found: {cert_path}")
        else:
            print(f"   ❌ SSL certificate not found: {cert_path}")
    else:
        print("   ℹ️  Using system SSL defaults")
    
    # Check API configuration
    print("\n🔌 API Configuration:")
    api_config = Settings.get_api_config()
    print(f"   Base URL: {api_config['base_url']}")
    print(f"   API Key: {'*' * len(api_config['api_key'][:8]) + '...' if len(api_config['api_key']) > 8 else 'Not set'}")
    
    return validation_result["valid"]

def setup_directories():
    """Set up required directories"""
    print("\n📁 Setting up directories")
    print("=" * 30)
    
    try:
        Settings.ensure_directories()
        print("✅ All directories created successfully")
        
        # List created directories
        directories = [
            Settings.KB_SYSTEM_PATH,
            Settings.KB_DATA_PATH,
            Settings.TEMP_DIR,
            Settings.WORK_DIR,
        ]
        
        for directory in directories:
            if directory.exists():
                print(f"   📁 {directory}")
            else:
                print(f"   ❌ Failed to create: {directory}")
                
    except Exception as e:
        print(f"❌ Error setting up directories: {e}")

def main():
    """Main setup function"""
    print("🚀 KB Brain Configuration Setup")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("kb_brain").exists():
        print("❌ Please run this script from the KB Brain project root")
        sys.exit(1)
    
    # Step 1: Create environment file
    create_environment_file()
    
    # Step 2: Generate configuration files
    generate_config_files()
    
    # Step 3: Set up directories
    setup_directories()
    
    # Step 4: Validate configuration
    is_valid = validate_configuration()
    
    print("\n✅ Setup completed!")
    print("\n📋 Next steps:")
    print("1. Review and customize your .env file")
    print("2. Test the configuration with: python3 scripts/validate_integration_simple.py")
    print("3. Run SSL setup if needed: python3 scripts/configure_ssl.py")
    
    if not is_valid:
        print("\n⚠️  Configuration validation failed - please review the errors above")
        sys.exit(1)

if __name__ == "__main__":
    main()