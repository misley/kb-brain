#!/bin/bash

# KB Brain Continue Extension Installation Script
echo "🔌 Installing KB Brain Continue Extension Integration"
echo "=================================================="

# Check if Continue extension is installed
if ! code --list-extensions | grep -q "continue.continue"; then
    echo "📦 Installing Continue extension..."
    code --install-extension continue.continue
    
    if [ $? -eq 0 ]; then
        echo "✅ Continue extension installed successfully"
    else
        echo "❌ Failed to install Continue extension"
        echo "Please install manually from: https://marketplace.visualstudio.com/items?itemName=continue.continue"
        exit 1
    fi
else
    echo "✅ Continue extension already installed"
fi

# Create Continue configuration directory
CONTINUE_DIR="$HOME/.continue"
mkdir -p "$CONTINUE_DIR"

# Check if KB Brain is installed
if ! python3 -c "import kb_brain" 2>/dev/null; then
    echo "❌ KB Brain not found. Please install KB Brain first:"
    echo "   pip install kb-brain"
    exit 1
fi

echo "✅ KB Brain found"

# Generate Continue configuration
echo "📝 Generating Continue configuration..."

# Get current directory
CURRENT_DIR=$(pwd)
CONFIG_FILE="$CONTINUE_DIR/config.json"

# Copy configuration
cp "$CURRENT_DIR/continue_config.json" "$CONFIG_FILE"

echo "✅ Continue configuration installed at: $CONFIG_FILE"

# Start KB Brain Continue server
echo "🚀 Starting KB Brain Continue server..."

# Check if server is already running
if curl -s http://localhost:8080/kb-brain/status > /dev/null 2>&1; then
    echo "✅ KB Brain Continue server already running"
else
    echo "🔄 Starting KB Brain Continue server..."
    
    # Start server in background
    nohup python3 -m kb_brain.integrations.continue_adapter --port 8080 > ~/.kb-brain/continue-server.log 2>&1 &
    
    # Wait for server to start
    sleep 3
    
    # Check if server started
    if curl -s http://localhost:8080/kb-brain/status > /dev/null 2>&1; then
        echo "✅ KB Brain Continue server started successfully"
        echo "📋 Server running at: http://localhost:8080/kb-brain"
        echo "📁 Logs available at: ~/.kb-brain/continue-server.log"
    else
        echo "❌ Failed to start KB Brain Continue server"
        echo "📋 Check logs at: ~/.kb-brain/continue-server.log"
        exit 1
    fi
fi

# Test integration
echo "🧪 Testing integration..."

# Test server status
SERVER_STATUS=$(curl -s http://localhost:8080/kb-brain/status)
if [ $? -eq 0 ]; then
    echo "✅ Server communication successful"
    echo "📊 Server status: $SERVER_STATUS"
else
    echo "❌ Server communication failed"
fi

# Instructions
echo ""
echo "🎉 KB Brain Continue Integration Setup Complete!"
echo "============================================="
echo ""
echo "📋 Next steps:"
echo "  1. Restart VSCode to load the Continue extension"
echo "  2. Open a code file and try the following commands:"
echo "     - /kb-search <query>     - Search knowledge base"
echo "     - /kb-debug <error>      - Debug with KB Brain"
echo "     - /kb-explain <code>     - Explain code"
echo "     - /kb-refactor <code>    - Get refactoring suggestions"
echo ""
echo "🔧 Configuration:"
echo "  - Continue config: $CONFIG_FILE"
echo "  - Server URL: http://localhost:8080/kb-brain"
echo "  - Server logs: ~/.kb-brain/continue-server.log"
echo ""
echo "🎯 Usage examples:"
echo "  - Select code and use Ctrl+I to get KB Brain suggestions"
echo "  - Type '/kb ' to see available KB Brain commands"
echo "  - Use the Continue chat panel for knowledge base queries"
echo ""
echo "🔍 Troubleshooting:"
echo "  - Check server status: curl http://localhost:8080/kb-brain/status"
echo "  - Restart server: pkill -f continue_adapter && python3 -m kb_brain.integrations.continue_adapter"
echo "  - Check logs: tail -f ~/.kb-brain/continue-server.log"
echo ""
echo "📚 Documentation: https://github.com/misley/kb-brain/blob/main/continue_integration/README.md"

# Create desktop shortcut for server management
DESKTOP_DIR="$HOME/Desktop"
if [ -d "$DESKTOP_DIR" ]; then
    cat > "$DESKTOP_DIR/KB Brain Continue Server.desktop" << 'EOF'
[Desktop Entry]
Version=1.0
Type=Application
Name=KB Brain Continue Server
Comment=Start/Stop KB Brain Continue Integration Server
Exec=bash -c "if curl -s http://localhost:8080/kb-brain/status > /dev/null 2>&1; then echo 'Server running'; else python3 -m kb_brain.integrations.continue_adapter --port 8080; fi"
Icon=applications-development
Terminal=true
Categories=Development;
EOF
    
    chmod +x "$DESKTOP_DIR/KB Brain Continue Server.desktop"
    echo "🖥️  Desktop shortcut created: $DESKTOP_DIR/KB Brain Continue Server.desktop"
fi

echo ""
echo "🚀 Ready to enhance your coding experience with KB Brain + Continue!"