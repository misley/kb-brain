# VS Code Integration Guide

<div class="alert alert-info">
<strong>Complete VS Code Integration:</strong> This guide covers integrating KB Brain with VS Code through Continue extension, GitHub Copilot, and direct API usage.
</div>

## Overview

KB Brain provides multiple integration paths for VS Code:

1. **Continue Extension** - Native integration with context-aware suggestions
2. **GitHub Copilot** - Enhanced prompts with KB Brain context
3. **Direct API** - Custom extensions and workflows
4. **Command Palette** - Direct KB Brain commands

## Continue Extension Integration

### 1. Installation

#### Install Continue Extension
```bash
# In VS Code, install Continue extension from the marketplace
# Or via command line:
code --install-extension continue.continue
```

#### Start KB Brain Continue Server
```bash
# Start the integration server
kb-brain-continue start --port 8080 --optimize-performance

# Or with custom configuration
kb-brain-continue start --port 8080 --kb-root /path/to/kb --optimize-performance
```

### 2. Configuration

#### Configure Continue Extension
Add KB Brain to your Continue configuration (`~/.continue/config.json`):

```json
{
  "models": [
    {
      "title": "KB Brain",
      "provider": "custom",
      "model": "kb-brain-hybrid",
      "apiBase": "http://localhost:8080/kb-brain",
      "apiKey": "your-api-key-here",
      "systemMessage": "You are an AI assistant with access to a comprehensive knowledge base. Use KB Brain to find relevant solutions, code patterns, and context for development tasks."
    }
  ],
  "customCommands": [
    {
      "name": "kb-search",
      "description": "Search KB Brain knowledge base",
      "prompt": "Search the knowledge base for: {input}\n\nProvide relevant solutions and context."
    },
    {
      "name": "kb-debug",
      "description": "Find debugging solutions",
      "prompt": "Find debugging solutions for this error: {input}\n\nSearch for similar errors and proven solutions."
    },
    {
      "name": "kb-explain",
      "description": "Explain code with KB context",
      "prompt": "Explain this code using knowledge base context: {input}\n\nProvide explanations and related patterns."
    },
    {
      "name": "kb-refactor",
      "description": "Get refactoring suggestions",
      "prompt": "Suggest refactoring improvements for: {input}\n\nSearch for better patterns and practices."
    }
  ],
  "contextProviders": [
    {
      "name": "kb-brain",
      "description": "KB Brain Knowledge Base",
      "type": "custom",
      "config": {
        "serverUrl": "http://localhost:8080/kb-brain",
        "apiKey": "your-api-key-here",
        "timeout": 5000,
        "maxSuggestions": 5
      }
    }
  ],
  "slashCommands": [
    {
      "name": "kb",
      "description": "Quick KB Brain search",
      "run": "kb-search {input}"
    },
    {
      "name": "debug",
      "description": "Debug with KB Brain",
      "run": "kb-debug {input}"
    },
    {
      "name": "explain",
      "description": "Explain with KB Brain",
      "run": "kb-explain {input}"
    }
  ]
}
```

### 3. Usage

#### Slash Commands
```javascript
// In VS Code, use Continue slash commands:
/kb SSL certificate configuration
/debug GPU memory error
/explain TfidfVectorizer implementation
/refactor similarity search function
```

#### Context-Aware Suggestions
```python
# Type code, Continue suggests based on KB Brain
import cupy as cp
# Continue suggests: pool = cp.get_default_memory_pool()
# Based on KB Brain GPU optimization knowledge

def optimize_ssl_certificates():
    # Continue suggests certificate validation patterns
    # from KB Brain SSL configuration knowledge
```

#### Code Completion
```python
# Continue provides context-aware completions
import ssl
context = ssl.create_default_context()
# Continue suggests: context.check_hostname = False
# Based on KB Brain corporate network patterns
```

## GitHub Copilot Integration

### 1. Enhanced Prompts with KB Brain Context

#### Create KB Brain Context Provider
```python
# Save as: ~/.vscode/kb-brain-copilot.py
import requests
import json

class KBBrainCopilotProvider:
    def __init__(self, api_base="http://localhost:8080/kb-brain", api_key="your-api-key"):
        self.api_base = api_base
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def get_context(self, query, max_results=3):
        """Get KB Brain context for Copilot prompts"""
        response = requests.post(
            f"{self.api_base}/search",
            json={
                "query": query,
                "max_results": max_results,
                "include_metadata": True
            },
            headers=self.headers
        )
        
        if response.status_code == 200:
            results = response.json()["results"]
            context = "\n".join([
                f"KB Solution: {result['title']}\n{result['content'][:200]}..."
                for result in results
            ])
            return context
        return ""
    
    def enhance_prompt(self, original_prompt, context_query):
        """Enhance Copilot prompt with KB Brain context"""
        context = self.get_context(context_query)
        if context:
            return f"{original_prompt}\n\nRelevant KB Brain context:\n{context}"
        return original_prompt
```

### 2. VS Code Extension for Copilot Integration

#### Create Custom Extension
```json
// package.json for KB Brain Copilot extension
{
  "name": "kb-brain-copilot",
  "displayName": "KB Brain Copilot Integration",
  "description": "Enhance GitHub Copilot with KB Brain context",
  "version": "1.0.0",
  "engines": {
    "vscode": "^1.60.0"
  },
  "categories": ["Other"],
  "activationEvents": [
    "onCommand:kb-brain-copilot.enhancePrompt"
  ],
  "main": "./out/extension.js",
  "contributes": {
    "commands": [
      {
        "command": "kb-brain-copilot.enhancePrompt",
        "title": "Enhance with KB Brain",
        "category": "KB Brain"
      },
      {
        "command": "kb-brain-copilot.searchKB",
        "title": "Search KB Brain",
        "category": "KB Brain"
      }
    ],
    "keybindings": [
      {
        "command": "kb-brain-copilot.enhancePrompt",
        "key": "ctrl+shift+k",
        "when": "editorTextFocus"
      }
    ]
  }
}
```

#### Extension Implementation
```typescript
// extension.ts
import * as vscode from 'vscode';
import axios from 'axios';

export function activate(context: vscode.ExtensionContext) {
    let enhancePromptCommand = vscode.commands.registerCommand('kb-brain-copilot.enhancePrompt', async () => {
        const editor = vscode.window.activeTextEditor;
        if (!editor) {
            return;
        }

        const selection = editor.selection;
        const selectedText = editor.document.getText(selection);
        
        if (!selectedText) {
            vscode.window.showWarningMessage('Please select code to enhance with KB Brain context');
            return;
        }

        // Get KB Brain context
        const context = await getKBBrainContext(selectedText);
        
        if (context) {
            // Insert context as comment above selection
            const contextComment = `// KB Brain Context:\n// ${context.replace(/\n/g, '\n// ')}\n\n`;
            editor.edit(editBuilder => {
                editBuilder.insert(selection.start, contextComment);
            });
        }
    });

    let searchKBCommand = vscode.commands.registerCommand('kb-brain-copilot.searchKB', async () => {
        const query = await vscode.window.showInputBox({
            placeHolder: 'Enter search query for KB Brain...',
            prompt: 'Search KB Brain knowledge base'
        });

        if (query) {
            const results = await searchKBBrain(query);
            if (results.length > 0) {
                const panel = vscode.window.createWebviewPanel(
                    'kbBrainResults',
                    'KB Brain Search Results',
                    vscode.ViewColumn.Two,
                    {}
                );
                
                panel.webview.html = generateResultsHTML(results);
            }
        }
    });

    context.subscriptions.push(enhancePromptCommand, searchKBCommand);
}

async function getKBBrainContext(query: string): Promise<string | null> {
    try {
        const response = await axios.post('http://localhost:8080/kb-brain/search', {
            query: query,
            max_results: 3
        }, {
            headers: {
                'Authorization': 'Bearer your-api-key',
                'Content-Type': 'application/json'
            }
        });

        if (response.data.results.length > 0) {
            return response.data.results[0].content.substring(0, 200);
        }
    } catch (error) {
        console.error('KB Brain API error:', error);
    }
    return null;
}

async function searchKBBrain(query: string): Promise<any[]> {
    try {
        const response = await axios.post('http://localhost:8080/kb-brain/search', {
            query: query,
            max_results: 10
        }, {
            headers: {
                'Authorization': 'Bearer your-api-key',
                'Content-Type': 'application/json'
            }
        });

        return response.data.results;
    } catch (error) {
        console.error('KB Brain search error:', error);
        return [];
    }
}

function generateResultsHTML(results: any[]): string {
    return `
        <html>
        <head>
            <style>
                body { font-family: var(--vscode-font-family); }
                .result { margin: 10px 0; padding: 10px; border: 1px solid #ccc; }
                .title { font-weight: bold; color: var(--vscode-textLink-foreground); }
                .content { margin-top: 5px; }
                .metadata { font-size: 0.9em; color: var(--vscode-descriptionForeground); }
            </style>
        </head>
        <body>
            <h2>KB Brain Search Results</h2>
            ${results.map(result => `
                <div class="result">
                    <div class="title">${result.title || 'Solution'}</div>
                    <div class="content">${result.content.substring(0, 300)}...</div>
                    <div class="metadata">
                        Confidence: ${result.confidence.toFixed(2)} | 
                        Similarity: ${result.similarity_score.toFixed(2)}
                    </div>
                </div>
            `).join('')}
        </body>
        </html>
    `;
}
```

### 3. Copilot Chat Integration

#### Enhanced Chat Commands
```typescript
// Add to VS Code settings.json
{
  "github.copilot.chat.welcomeMessage": "Hi! I'm enhanced with KB Brain context. Ask me about your codebase patterns, SSL configurations, or performance optimizations.",
  "github.copilot.chat.models": [
    {
      "name": "KB Brain Enhanced",
      "provider": "custom",
      "endpoint": "http://localhost:8080/kb-brain/chat"
    }
  ]
}
```

#### Chat Integration Script
```python
# kb-brain-chat-enhancer.py
import requests
import json
import sys

def enhance_copilot_chat(message):
    """Enhance Copilot chat with KB Brain context"""
    
    # Search KB Brain for relevant context
    kb_response = requests.post(
        "http://localhost:8080/kb-brain/search",
        json={
            "query": message,
            "max_results": 3,
            "include_metadata": True
        },
        headers={
            "Authorization": "Bearer your-api-key",
            "Content-Type": "application/json"
        }
    )
    
    if kb_response.status_code == 200:
        results = kb_response.json()["results"]
        if results:
            context = "\n".join([
                f"• {result['title']}: {result['content'][:100]}..."
                for result in results
            ])
            
            enhanced_message = f"""
{message}

Relevant KB Brain context:
{context}

Please provide a response that considers this context from our knowledge base.
"""
            return enhanced_message
    
    return message

if __name__ == "__main__":
    if len(sys.argv) > 1:
        original_message = " ".join(sys.argv[1:])
        enhanced = enhance_copilot_chat(original_message)
        print(enhanced)
    else:
        print("Usage: python kb-brain-chat-enhancer.py 'your message here'")
```

## Direct API Integration

### 1. VS Code Tasks

#### Configure KB Brain Tasks
```json
// .vscode/tasks.json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "KB Brain Search",
            "type": "shell",
            "command": "kb-brain",
            "args": ["search", "${input:searchQuery}"],
            "group": "test",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "shared"
            }
        },
        {
            "label": "KB Brain Status",
            "type": "shell",
            "command": "kb-brain",
            "args": ["status", "--optimize-performance"],
            "group": "test"
        },
        {
            "label": "KB Brain Benchmark",
            "type": "shell",
            "command": "kb-brain",
            "args": ["benchmark"],
            "group": "test"
        }
    ],
    "inputs": [
        {
            "id": "searchQuery",
            "description": "Enter search query for KB Brain",
            "default": "SSL certificate issues",
            "type": "promptString"
        }
    ]
}
```

### 2. Snippets Integration

#### Create KB Brain Snippets
```json
// .vscode/snippets/kb-brain.json
{
    "KB Brain SSL Configuration": {
        "scope": "python,bash",
        "prefix": "kb-ssl",
        "body": [
            "# KB Brain SSL Configuration Pattern",
            "# Based on corporate network SSL handling",
            "import ssl",
            "import os",
            "",
            "# Configure SSL context",
            "context = ssl.create_default_context()",
            "cert_path = os.environ.get('SSL_CERT_PATH', '/etc/ssl/certs/ca-certificates.crt')",
            "if os.path.exists(cert_path):",
            "    context.load_verify_locations(cert_path)",
            "",
            "# ${1:Your SSL implementation here}",
            "$0"
        ],
        "description": "KB Brain SSL configuration pattern"
    },
    "KB Brain GPU Optimization": {
        "scope": "python",
        "prefix": "kb-gpu",
        "body": [
            "# KB Brain GPU Memory Optimization Pattern",
            "import cupy as cp",
            "",
            "# Configure GPU memory pool",
            "pool = cp.get_default_memory_pool()",
            "pool.set_limit(size=2**30)  # 1GB limit",
            "",
            "# ${1:Your GPU implementation here}",
            "",
            "# Cleanup",
            "pool.free_all_blocks()",
            "$0"
        ],
        "description": "KB Brain GPU optimization pattern"
    }
}
```

## Advanced Integration

### 1. Workspace Configuration

#### KB Brain Workspace Settings
```json
// .vscode/settings.json
{
    "kb-brain.apiBase": "http://localhost:8080/kb-brain",
    "kb-brain.apiKey": "${env:KB_BRAIN_API_KEY}",
    "kb-brain.enablePerformanceOptimizations": true,
    "kb-brain.maxResults": 10,
    "kb-brain.similarityThreshold": 0.3,
    "kb-brain.enableAutoComplete": true,
    "kb-brain.enableContextProvider": true,
    
    "continue.serverUrl": "http://localhost:8080/kb-brain",
    "continue.apiKey": "${env:KB_BRAIN_API_KEY}",
    "continue.enableKBBrain": true,
    
    "github.copilot.enable": {
        "*": true,
        "kb-brain": true
    }
}
```

### 2. Launch Configuration

#### Debug with KB Brain Context
```json
// .vscode/launch.json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "KB Brain Server",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/kb_brain/cli/main.py",
            "args": ["interactive", "--optimize-performance"],
            "console": "integratedTerminal",
            "env": {
                "KB_BRAIN_API_KEY": "${env:KB_BRAIN_API_KEY}",
                "KB_BRAIN_PERFORMANCE": "true"
            }
        },
        {
            "name": "Continue Server",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/kb_brain/cli/continue_cli.py",
            "args": ["start", "--port", "8080", "--optimize-performance"],
            "console": "integratedTerminal"
        }
    ]
}
```

## Troubleshooting

### Common Issues

#### 1. Continue Extension Not Connecting
```bash
# Check if KB Brain server is running
curl -X GET http://localhost:8080/kb-brain/status

# Check Continue configuration
cat ~/.continue/config.json | jq '.models[] | select(.title == "KB Brain")'

# Restart Continue server
kb-brain-continue stop
kb-brain-continue start --port 8080 --optimize-performance
```

#### 2. GitHub Copilot Not Using Context
```typescript
// Verify KB Brain API is accessible
const testKBBrain = async () => {
    try {
        const response = await fetch('http://localhost:8080/kb-brain/status');
        const data = await response.json();
        console.log('KB Brain Status:', data);
    } catch (error) {
        console.error('KB Brain not accessible:', error);
    }
};
```

#### 3. Performance Issues
```bash
# Enable performance optimizations
export KB_BRAIN_PERFORMANCE=true

# Check system resources
kb-brain status --optimize-performance

# Monitor Continue server
kb-brain-continue monitor --port 8080
```

### Debug Mode

#### Enable Debug Logging
```json
// VS Code settings.json
{
    "kb-brain.debug": true,
    "kb-brain.logLevel": "debug",
    "continue.debug": true
}
```

## Best Practices

### 1. Security
- Store API keys in environment variables
- Use HTTPS in production
- Rotate API keys regularly
- Validate SSL certificates

### 2. Performance
- Enable performance optimizations
- Use appropriate similarity thresholds
- Limit result counts for better performance
- Monitor resource usage

### 3. Workflow
- Use specific search terms for better results
- Leverage slash commands for quick access
- Combine Continue and Copilot for maximum benefit
- Regular KB Brain knowledge updates

---

<div class="footer">
<p>VS Code Integration Guide • KB Brain Version 1.1.0 • Last Updated: January 15, 2024</p>
</div>