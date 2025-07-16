#!/bin/bash

# KB Brain MCP Setup Script
echo "🔌 Setting up KB Brain MCP integration..."

# Create MCP configuration
MCP_CONFIG="$HOME/.config/claude/mcp_servers.json"
mkdir -p "$(dirname "$MCP_CONFIG")"

echo "📝 Creating MCP configuration..."
cat > "$MCP_CONFIG" << 'MCPEOF'
{
  "mcpServers": {
    "kb-brain": {
      "command": "kb-brain-mcp",
      "args": []
    }
  }
}
MCPEOF

echo "✅ MCP configuration created at $MCP_CONFIG"

# Test MCP server
echo "🧪 Testing MCP server..."
timeout 5 kb-brain-mcp & 
sleep 2
if pgrep -f "kb-brain-mcp" > /dev/null; then
    echo "✅ MCP server started successfully"
    pkill -f "kb-brain-mcp"
else
    echo "❌ MCP server failed to start"
    exit 1
fi

echo "🎉 KB Brain MCP setup complete!"
echo "📋 Next steps:"
echo "  1. Restart Claude to load new MCP configuration"
echo "  2. Use KB Brain tools in Claude conversations"
