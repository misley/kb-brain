# Configuration Guide

<div class="alert alert-info">
<strong>Configuration Overview:</strong> KB Brain uses a centralized configuration system with environment variables, providing secure and flexible deployment options.
</div>

## Configuration Architecture

KB Brain uses a three-tier configuration system:

1. **Environment Variables** - Primary configuration method
2. **Configuration Files** - Structured settings for complex configurations
3. **Runtime Parameters** - Dynamic configuration for specific operations

## Environment Variables

### Core Configuration

#### Path Configuration
```bash
# Knowledge base paths
KB_SYSTEM_PATH=/path/to/kb_system           # Main knowledge base directory
KB_BRAIN_PATH=/path/to/kb-brain             # KB Brain application directory
KB_DATA_PATH=/path/to/data                  # Data storage directory

# Working directories
KB_BRAIN_TEMP_DIR=/tmp/kb_brain             # Temporary files
KB_BRAIN_WORK_DIR=/home/user/kb_brain_work  # Working directory
```

#### API Configuration
```bash
# API server settings
KB_BRAIN_API_KEY=your-secure-api-key-here  # API authentication key
KB_BRAIN_API_BASE=http://localhost:8080    # Base API URL
KB_BRAIN_API_TIMEOUT=30                    # Request timeout (seconds)

# Continue integration
KB_BRAIN_CONTINUE_PORT=8080                # Continue server port
KB_BRAIN_CONTINUE_HOST=localhost           # Continue server host
```

#### Performance Configuration
```bash
# Performance optimizations
KB_BRAIN_PERFORMANCE=true                  # Enable performance optimizations
KB_BRAIN_MAX_WORKERS=4                     # Maximum worker threads
KB_BRAIN_CACHE_SIZE=1000                   # Cache size limit

# Search configuration
KB_BRAIN_DEFAULT_RESULTS=10                # Default search results count
KB_BRAIN_MAX_RESULTS=100                   # Maximum search results
KB_BRAIN_SIMILARITY_THRESHOLD=0.3          # Minimum similarity threshold
```

### Security Configuration

#### SSL Configuration
```bash
# SSL certificate paths
SSL_CERT_PATH=/etc/ssl/certs/ca-certificates.crt  # System certificates
SSL_CA_BUNDLE=/path/to/ca-bundle.crt              # Custom CA bundle

# Certificate validation
SSL_VERIFY_CERTS=true                      # Enable certificate verification
SSL_STRICT_MODE=false                      # Strict SSL mode
```

#### Organization Configuration
```bash
# Organization settings (customize for your deployment)
ORGANIZATION_NAME=Your Organization Name   # Organization name
ORGANIZATION_EMAIL=info@yourorg.com       # Contact email
REPOSITORY_URL=https://github.com/yourorg/kb-brain  # Repository URL
```

### Advanced Configuration

#### Logging Configuration
```bash
# Logging settings
KB_BRAIN_LOG_LEVEL=INFO                    # Log level (DEBUG, INFO, WARNING, ERROR)
KB_BRAIN_LOG_FILE=kb_brain.log            # Log file path
KB_BRAIN_LOG_FORMAT=json                  # Log format (json, text)
KB_BRAIN_LOG_MAX_SIZE=10MB                # Maximum log file size
```

#### Database Configuration
```bash
# Database settings
KB_BRAIN_DATABASE_URL=sqlite:///kb_brain.db  # Database connection string
KB_BRAIN_DB_POOL_SIZE=5                      # Connection pool size
KB_BRAIN_DB_TIMEOUT=30                       # Database timeout
```

#### Development Configuration
```bash
# Development settings
KB_BRAIN_ENV=development                   # Environment (development, production)
KB_BRAIN_DEBUG=false                      # Debug mode
KB_BRAIN_RELOAD=false                     # Auto-reload on changes
```

## Configuration Files

### Environment File (.env)

Create your environment file from the template:

```bash
# Copy template
cp .env.template .env

# Edit with your settings
nano .env
```

Example `.env` file:
```bash
# KB Brain Production Configuration
KB_BRAIN_ENV=production
KB_SYSTEM_PATH=/opt/kb_brain/kb_system
KB_BRAIN_PATH=/opt/kb_brain
KB_DATA_PATH=/opt/kb_brain/data

# Security
KB_BRAIN_API_KEY=prod-api-key-here
SSL_CERT_PATH=/etc/ssl/certs/ca-certificates.crt

# Performance
KB_BRAIN_PERFORMANCE=true
KB_BRAIN_MAX_WORKERS=8
KB_BRAIN_CACHE_SIZE=2000

# Organization
ORGANIZATION_NAME=Acme Corporation
ORGANIZATION_EMAIL=admin@acme.com
REPOSITORY_URL=https://github.com/acme/kb-brain
```

### Settings Module

The centralized settings module (`kb_brain/config/settings.py`) provides:

```python
from kb_brain.config.settings import Settings

# Access configuration
print(Settings.KB_SYSTEM_PATH)
print(Settings.API_KEY)

# Get configuration groups
api_config = Settings.get_api_config()
ssl_config = Settings.get_ssl_config()
search_config = Settings.get_search_config()

# Validate configuration
validation = Settings.validate_config()
if validation["valid"]:
    print("Configuration is valid")
```

### Continue Configuration

Configure the Continue extension (`continue_integration/continue_config.json`):

```json
{
  "models": [
    {
      "title": "KB Brain",
      "provider": "custom",
      "model": "kb-brain-hybrid",
      "apiBase": "${KB_BRAIN_API_BASE}/kb-brain",
      "apiKey": "${KB_BRAIN_API_KEY}",
      "systemMessage": "You are an AI assistant with access to a knowledge base."
    }
  ],
  "customCommands": [
    {
      "name": "kb-search",
      "description": "Search KB Brain knowledge base",
      "prompt": "Search the knowledge base for: {input}"
    }
  ]
}
```

### MCP Configuration

Configure the MCP server (`config/mcp_config.json`):

```json
{
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
```

## Configuration Management

### Setup Script

Use the configuration setup script for guided setup:

```bash
python3 scripts/setup_config.py
```

This script will:
1. Create `.env` file from template
2. Prompt for custom settings
3. Generate configuration files
4. Validate the configuration
5. Set up required directories

### Configuration Validation

Validate your configuration:

```bash
# Python validation
python3 -c "from kb_brain.config.settings import Settings; print(Settings.validate_config())"

# Script validation
python3 scripts/validate_integration_simple.py
```

### SSL Configuration

Configure SSL certificates:

```bash
# Run SSL configuration script
python3 scripts/configure_ssl.py

# Manual SSL setup
export SSL_CERT_PATH=/path/to/your/certificate.crt
export SSL_CA_BUNDLE=/path/to/ca-bundle.crt
```

## Deployment Configurations

### Development Environment

```bash
# Development .env
KB_BRAIN_ENV=development
KB_BRAIN_DEBUG=true
KB_BRAIN_LOG_LEVEL=DEBUG
KB_BRAIN_PERFORMANCE=false
KB_BRAIN_API_KEY=dev-key-not-secure
```

### Production Environment

```bash
# Production .env
KB_BRAIN_ENV=production
KB_BRAIN_DEBUG=false
KB_BRAIN_LOG_LEVEL=INFO
KB_BRAIN_PERFORMANCE=true
KB_BRAIN_API_KEY=secure-production-key
SSL_CERT_PATH=/etc/ssl/certs/ca-certificates.crt
```

### Docker Configuration

```dockerfile
# Docker environment
FROM python:3.12-slim

# Set environment variables
ENV KB_BRAIN_ENV=production
ENV KB_BRAIN_LOG_LEVEL=INFO
ENV KB_BRAIN_PERFORMANCE=true

# Copy configuration
COPY .env /app/.env
COPY config/ /app/config/

# Install and run
WORKDIR /app
RUN pip install kb-brain
CMD ["kb-brain", "interactive"]
```

### Kubernetes Configuration

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: kb-brain-config
data:
  KB_BRAIN_ENV: "production"
  KB_BRAIN_LOG_LEVEL: "INFO"
  KB_BRAIN_PERFORMANCE: "true"
  KB_BRAIN_API_BASE: "https://api.kbbrain.com"
  
---
apiVersion: v1
kind: Secret
metadata:
  name: kb-brain-secrets
type: Opaque
data:
  KB_BRAIN_API_KEY: <base64-encoded-api-key>
  SSL_CERT: <base64-encoded-certificate>
```

## Configuration Best Practices

### Security Best Practices

1. **Never commit secrets** to version control
2. **Use environment variables** for sensitive data
3. **Rotate API keys** regularly
4. **Validate certificates** in production
5. **Set appropriate file permissions**:
   ```bash
   chmod 600 .env
   chmod 644 config/*.json
   ```

### Performance Best Practices

1. **Enable performance optimizations** in production
2. **Configure appropriate worker counts**:
   ```bash
   KB_BRAIN_MAX_WORKERS=$(nproc)  # Use all CPU cores
   ```
3. **Set cache sizes** based on available memory
4. **Use GPU acceleration** when available

### Deployment Best Practices

1. **Environment-specific configurations**:
   - Development: Debug enabled, performance disabled
   - Staging: Performance enabled, debug logging
   - Production: Optimized settings, minimal logging

2. **Configuration validation**:
   ```bash
   # Pre-deployment validation
   python3 scripts/validate_integration_simple.py
   python3 scripts/configure_ssl.py
   ```

3. **Monitoring and logging**:
   ```bash
   KB_BRAIN_LOG_LEVEL=INFO
   KB_BRAIN_LOG_FORMAT=json
   ```

## Troubleshooting Configuration

### Common Issues

#### 1. Environment Variables Not Loading
```bash
# Check if .env file exists
ls -la .env

# Verify environment variables
echo $KB_BRAIN_API_KEY
echo $KB_SYSTEM_PATH

# Load manually
source .env
```

#### 2. SSL Certificate Issues
```bash
# Test SSL configuration
python3 scripts/configure_ssl.py

# Check certificate paths
ls -la $SSL_CERT_PATH
ls -la $SSL_CA_BUNDLE
```

#### 3. Performance Optimizations Not Working
```bash
# Check dependencies
pip install intel-extension-for-scikit-learn numba

# Verify configuration
python3 -c "from kb_brain.config.settings import Settings; print(Settings.ENABLE_PERFORMANCE_OPTIMIZATIONS)"
```

#### 4. Path Issues
```bash
# Check directory permissions
ls -la $KB_SYSTEM_PATH
ls -la $KB_DATA_PATH

# Create missing directories
python3 -c "from kb_brain.config.settings import Settings; Settings.ensure_directories()"
```

### Debug Configuration

Enable debug mode to troubleshoot:

```bash
# Enable debug logging
export KB_BRAIN_LOG_LEVEL=DEBUG
export KB_BRAIN_DEBUG=true

# Run with debug output
kb-brain status
```

### Configuration Testing

Test your configuration:

```bash
# Test basic functionality
kb-brain status

# Test search
kb-brain search "test query"

# Test performance
kb-brain benchmark

# Validate all settings
python3 scripts/validate_integration_simple.py
```

## Advanced Configuration

### Custom Configuration Providers

Create custom configuration providers:

```python
from kb_brain.config.settings import Settings

class CustomConfigProvider:
    def __init__(self, config_source):
        self.config_source = config_source
    
    def get_config(self, key):
        # Custom configuration logic
        return self.config_source.get(key)

# Register custom provider
Settings.register_provider(CustomConfigProvider(my_config))
```

### Dynamic Configuration

Update configuration at runtime:

```python
from kb_brain.config.settings import Settings

# Update configuration
Settings.update_setting('KB_BRAIN_LOG_LEVEL', 'DEBUG')

# Reload configuration
Settings.reload_config()
```

### Configuration Monitoring

Monitor configuration changes:

```python
from kb_brain.config.settings import Settings

def config_changed_handler(key, old_value, new_value):
    print(f"Configuration changed: {key} = {new_value}")

# Register change handler
Settings.register_change_handler(config_changed_handler)
```

## Migration Guide

### Upgrading from Previous Versions

1. **Backup existing configuration**:
   ```bash
   cp .env .env.backup
   cp -r config config.backup
   ```

2. **Update configuration format**:
   ```bash
   # Run migration script
   python3 scripts/migrate_config.py
   ```

3. **Validate new configuration**:
   ```bash
   python3 scripts/validate_integration_simple.py
   ```

### Configuration Schema Changes

Check for schema changes in new versions:

```python
from kb_brain.config.settings import Settings

# Check configuration version
print(Settings.get_config_version())

# Validate against schema
validation = Settings.validate_schema()
if not validation["valid"]:
    print("Configuration schema validation failed")
```

---

<div class="footer">
<p>KB Brain Configuration Guide • Version 1.1.0 • Last Updated: January 15, 2024</p>
</div>