#!/bin/bash

# Fix DOI Certificate Trust in WSL for nvm/node connectivity
# Uses existing local DOI certificate files

echo "=== Fixing DOI SSL Certificates in WSL ==="
echo "This will enable nvm/node and other tools to work on corporate network"
echo ""

# Certificate files
DOI_CERT="/mnt/c/Users/misley/Documents/Projects/doi_cacert.pem"
BUNDLE_CERT="/mnt/c/Users/misley/Documents/Projects/cacert.pem"

# Verify certificates exist
if [ ! -f "$DOI_CERT" ]; then
    echo "Error: DOI certificate not found at $DOI_CERT"
    exit 1
fi

echo "Found DOI certificate: $DOI_CERT"

# 1. Install DOI certificate in system certificate store
echo ""
echo "1. Installing DOI certificate in system store..."
sudo cp "$DOI_CERT" /usr/local/share/ca-certificates/doi-root-ca.crt
sudo update-ca-certificates

# 2. Configure environment variables for SSL
echo ""
echo "2. Setting up SSL environment variables..."

# Create a shell config that works for both bash and zsh
cat > ~/.ssl_env << 'EOF'
# DOI SSL Certificate Configuration
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_DIR=/etc/ssl/certs
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export CURL_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export NODE_EXTRA_CA_CERTS=/usr/local/share/ca-certificates/doi-root-ca.crt

# For npm/node specifically
export NODE_TLS_REJECT_UNAUTHORIZED=1
export npm_config_cafile=/etc/ssl/certs/ca-certificates.crt
EOF

# Add to .bashrc if not already there
if ! grep -q "source ~/.ssl_env" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# Load SSL environment settings" >> ~/.bashrc
    echo "[ -f ~/.ssl_env ] && source ~/.ssl_env" >> ~/.bashrc
fi

# Also add to .profile for non-interactive shells
if ! grep -q "source ~/.ssl_env" ~/.profile 2>/dev/null; then
    echo "" >> ~/.profile
    echo "# Load SSL environment settings" >> ~/.profile
    echo "[ -f ~/.ssl_env ] && source ~/.ssl_env" >> ~/.profile
fi

# Source it now
source ~/.ssl_env

# 3. Configure git to use the certificates
echo ""
echo "3. Configuring git SSL settings..."
git config --global http.sslCAInfo /etc/ssl/certs/ca-certificates.crt

# 4. Configure npm/nvm specifically
echo ""
echo "4. Configuring npm/nvm..."

# Set npm to use system certificates
npm config set cafile /etc/ssl/certs/ca-certificates.crt 2>/dev/null || true
npm config set strict-ssl true 2>/dev/null || true

# If nvm is installed, ensure it uses the certificates
if [ -d "$HOME/.nvm" ]; then
    echo "nvm detected, configuring..."
    # Create nvm default packages with ca settings
    mkdir -p ~/.nvm
    echo "ca=/etc/ssl/certs/ca-certificates.crt" > ~/.nvm/default-packages
fi

# 5. Test connectivity
echo ""
echo "5. Testing connectivity..."

# Test basic HTTPS
echo -n "Testing curl HTTPS: "
if curl -s -I https://github.com > /dev/null 2>&1; then
    echo "SUCCESS"
else
    echo "FAILED"
fi

# Test npm registry
echo -n "Testing npm registry: "
if curl -s -I https://registry.npmjs.org > /dev/null 2>&1; then
    echo "SUCCESS"
else
    echo "FAILED"
fi

# Test nodejs.org (for nvm)
echo -n "Testing nodejs.org: "
if curl -s -I https://nodejs.org > /dev/null 2>&1; then
    echo "SUCCESS"
else
    echo "FAILED"
fi

echo ""
echo "=== Setup Complete ==="
echo ""
echo "To apply changes to your current shell:"
echo "  source ~/.bashrc"
echo ""
echo "For nvm/node to work properly:"
echo "1. Close and reopen your terminal"
echo "2. Or run: source ~/.bashrc && source ~/.nvm/nvm.sh"
echo ""
echo "If you still have issues, you can temporarily use:"
echo "  export NODE_TLS_REJECT_UNAUTHORIZED=0  # Not recommended for production"
echo ""
echo "Certificate locations:"
echo "  System: /usr/local/share/ca-certificates/doi-root-ca.crt"
echo "  Bundle: /etc/ssl/certs/ca-certificates.crt"