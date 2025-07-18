# KB Brain Security Configuration

## Overview

This document outlines the security configuration for KB Brain, including environment variable management, SSL certificate handling, and secure deployment practices.

## Security Principles

### 1. No Hardcoded Secrets
- All sensitive information is configured via environment variables
- Default values are non-sensitive placeholders
- Production values must be explicitly set

### 2. Secure Defaults
- SSL verification enabled by default
- API keys require explicit configuration
- Personal/organizational information externalized

### 3. Environment Isolation
- Development and production configurations separated
- Local configuration files excluded from version control
- Secure certificate management

## Configuration Management

### Environment Variables

KB Brain uses environment variables for all sensitive configuration:

```bash
# Core Security Settings
KB_BRAIN_API_KEY=your-secure-api-key-here
SSL_CERT_PATH=/path/to/your/certificates.crt
SSL_CA_BUNDLE=/path/to/ca-bundle.crt

# Organization Settings
ORGANIZATION_NAME=Your Organization
ORGANIZATION_EMAIL=security@yourorg.com
REPOSITORY_URL=https://github.com/yourorg/kb-brain

# Network Configuration
KB_BRAIN_API_BASE=https://your-api-server.com
KB_BRAIN_CONTINUE_HOST=localhost
KB_BRAIN_CONTINUE_PORT=8080
```

### Configuration Files

#### .env File Structure
```bash
# Copy .env.template to .env and customize
cp .env.template .env
```

#### Secure Configuration Setup
```bash
# Run the configuration setup script
python3 scripts/setup_config.py

# Configure SSL certificates
python3 scripts/configure_ssl.py
```

## SSL Certificate Management

### Corporate Network SSL

For corporate networks with custom SSL certificates:

1. **Obtain Certificate Files**
   - Get your organization's root CA certificate
   - Usually named `ca-certificate.crt` or similar
   - Contact your IT security team if needed

2. **Configure Certificate Path**
   ```bash
   export SSL_CERT_PATH=/path/to/your/ca-certificate.crt
   export SSL_CA_BUNDLE=/path/to/ca-bundle.crt
   ```

3. **Test SSL Configuration**
   ```bash
   python3 scripts/configure_ssl.py
   ```

### Certificate Validation

KB Brain validates SSL certificates by:
- Checking certificate file existence
- Validating certificate format
- Testing HTTPS connectivity
- Providing fallback to system certificates

## API Security

### API Key Management

1. **Generate Secure API Keys**
   ```bash
   # Generate a secure API key
   python3 -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. **Set API Key**
   ```bash
   export KB_BRAIN_API_KEY=your-generated-key-here
   ```

3. **API Key Rotation**
   - Rotate keys regularly (recommend monthly)
   - Update all client configurations
   - Test connectivity after rotation

### API Endpoint Security

- All API endpoints require authentication
- Rate limiting implemented
- Request validation and sanitization
- HTTPS required for production

## File Security

### Sensitive File Exclusion

The following files are excluded from version control:
```gitignore
# Environment files
.env
.env.*

# Certificate files
*.pem
*.crt
*.key

# API keys and credentials
api_keys.json
credentials.json
secrets.json
```

### File Permissions

Set appropriate permissions for sensitive files:
```bash
# Environment files
chmod 600 .env

# Certificate files
chmod 644 *.crt
chmod 600 *.key

# Configuration files
chmod 600 config/local_config.json
```

## Deployment Security

### Development Environment
- Use `.env.template` as starting point
- Default values are safe for development
- No production secrets in development

### Production Environment
- All secrets set via environment variables
- SSL certificates properly configured
- API keys securely generated and stored
- Network access restricted to necessary ports

### Container Security
```dockerfile
# Example secure Dockerfile configuration
FROM python:3.12-slim

# Create non-root user
RUN useradd -m -u 1000 kbrain

# Set secure environment
ENV KB_BRAIN_ENV=production
ENV KB_BRAIN_LOG_LEVEL=INFO

# Copy application
COPY --chown=kbrain:kbrain . /app
WORKDIR /app

# Switch to non-root user
USER kbrain

# Run application
CMD ["python3", "-m", "kb_brain.cli.main"]
```

## Network Security

### SSL/TLS Configuration
- TLS 1.2+ required
- Certificate validation enabled
- Secure cipher suites only
- Certificate pinning for production

### Network Access
- API server bound to specific interfaces
- Firewall rules restrict access
- VPN access for remote management
- Network segmentation for sensitive components

## Monitoring and Auditing

### Security Logging
```python
# Configure security logging
import logging

security_logger = logging.getLogger('kb_brain.security')
security_logger.setLevel(logging.INFO)

# Log security events
security_logger.info("API key validation successful")
security_logger.warning("Invalid API key attempt")
security_logger.error("SSL certificate validation failed")
```

### Audit Trail
- API access logging
- Configuration changes tracked
- Certificate usage monitored
- Failed authentication attempts logged

## Incident Response

### Security Incident Steps
1. **Immediate Response**
   - Rotate compromised API keys
   - Block suspicious IP addresses
   - Review access logs

2. **Investigation**
   - Analyze log files
   - Check for unauthorized access
   - Assess data exposure

3. **Recovery**
   - Update security configurations
   - Patch vulnerabilities
   - Test system integrity

### Emergency Contacts
- Security team: security@yourorg.com
- IT support: support@yourorg.com
- Management: management@yourorg.com

## Security Testing

### Regular Security Checks
```bash
# Validate current configuration
python3 scripts/validate_integration_simple.py

# Test SSL configuration
python3 scripts/configure_ssl.py

# Security scan
python3 scripts/security_scan.py
```

### Security Checklist
- [ ] API keys are secure and rotated regularly
- [ ] SSL certificates are valid and up-to-date
- [ ] Environment variables are properly set
- [ ] Sensitive files are not in version control
- [ ] Network access is restricted
- [ ] Logging is configured and monitored
- [ ] Backup and recovery procedures tested

## Best Practices

### Development
- Never commit secrets to version control
- Use `.env.template` for documentation
- Test with secure configurations
- Regular security code reviews

### Production
- Environment variable injection only
- Secure certificate management
- Regular security updates
- Monitoring and alerting
- Access control and authorization

### Maintenance
- Regular security audits
- Vulnerability scanning
- Dependency updates
- Certificate renewal
- Key rotation procedures

## Support and Resources

### Configuration Help
- Run `python3 scripts/setup_config.py` for guided setup
- Check `docs/troubleshooting.md` for common issues
- Review `.env.template` for all available settings

### Security Resources
- OWASP Security Guidelines
- Corporate security policies
- Industry best practices
- Security training materials

### Contact Information
- Security questions: security@yourorg.com
- Technical support: support@yourorg.com
- Documentation: docs@yourorg.com