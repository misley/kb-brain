#!/usr/bin/env python3
"""
KB Brain Continue CLI - Command-line interface for Continue integration
"""

import click
import json
import asyncio
import signal
import sys
from pathlib import Path
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.spinner import Spinner
import requests
import time

console = Console()

@click.group()
def continue_cli():
    """KB Brain Continue Extension Integration CLI"""
    pass

@continue_cli.command()
@click.option('--port', default=8080, help='Server port')
@click.option('--kb-root', help='KB Brain data directory')
@click.option('--debug', is_flag=True, help='Enable debug mode')
@click.option('--optimize-performance', is_flag=True, default=True, help='Enable performance optimizations (default: enabled)')
def start(port, kb_root, debug, optimize_performance):
    """Start KB Brain Continue integration server"""
    
    console.print(f"🚀 Starting KB Brain Continue server on port {port}")
    
    if optimize_performance:
        console.print("🚀 Performance optimizations enabled")
    
    try:
        from kb_brain.integrations.continue_adapter import ContinueAdapter
        
        # Create adapter with performance optimization settings
        adapter = ContinueAdapter(enable_performance_optimizations=optimize_performance)
        
        # Start server
        asyncio.run(adapter.start_continue_server(port))
        
    except KeyboardInterrupt:
        console.print("\n🛑 Server stopped by user")
    except Exception as e:
        console.print(f"❌ Error starting server: {e}")
        sys.exit(1)

@continue_cli.command()
@click.option('--port', default=8080, help='Server port')
def status(port):
    """Check KB Brain Continue server status"""
    
    try:
        response = requests.get(f"http://localhost:{port}/kb-brain/status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            # Create status table
            table = Table(title="KB Brain Continue Server Status")
            table.add_column("Component", style="cyan")
            table.add_column("Status", style="green")
            table.add_column("Details", style="yellow")
            
            table.add_row("Server", "✅ Online", f"Port {port}")
            table.add_row("Continue Adapter", data.get('continue_adapter', 'Unknown'), "Ready")
            
            kb_status = data.get('kb_brain_status', {})
            gpu_status = kb_status.get('hybrid_gpu_status', {})
            
            table.add_row("KB Brain", "✅ Ready", f"GPU: {'✅' if gpu_status.get('gpu_available') else '❌'}")
            table.add_row("Knowledge Base", "✅ Loaded", f"Embeddings: {gpu_status.get('knowledge_embeddings', 0)}")
            table.add_row("Solutions", "✅ Available", f"Count: {gpu_status.get('solution_texts', 0)}")
            
            console.print(table)
            
        else:
            console.print(f"❌ Server returned status {response.status_code}")
            
    except requests.ConnectionError:
        console.print(f"❌ Cannot connect to server on port {port}")
        console.print(f"💡 Start server with: kb-brain-continue start --port {port}")
    except Exception as e:
        console.print(f"❌ Error checking status: {e}")

@continue_cli.command()
@click.option('--port', default=8080, help='Server port')
def stop(port):
    """Stop KB Brain Continue server"""
    
    try:
        # Try to connect first
        response = requests.get(f"http://localhost:{port}/kb-brain/status", timeout=2)
        
        if response.status_code == 200:
            console.print(f"🛑 Stopping KB Brain Continue server on port {port}")
            
            # Send shutdown signal (this is a simplified approach)
            # In a real implementation, you'd have a proper shutdown endpoint
            import subprocess
            subprocess.run(['pkill', '-f', 'continue_adapter'], check=False)
            
            time.sleep(2)
            
            # Check if stopped
            try:
                requests.get(f"http://localhost:{port}/kb-brain/status", timeout=2)
                console.print("⚠️  Server may still be running")
            except requests.ConnectionError:
                console.print("✅ Server stopped successfully")
        else:
            console.print(f"❌ Server not responding on port {port}")
            
    except requests.ConnectionError:
        console.print(f"❌ Server not running on port {port}")
    except Exception as e:
        console.print(f"❌ Error stopping server: {e}")

@continue_cli.command()
@click.option('--output', help='Output file path')
def config(output):
    """Generate Continue extension configuration"""
    
    try:
        from kb_brain.integrations.continue_adapter import ContinueAdapter
        
        adapter = ContinueAdapter()
        config_data = adapter.get_continue_config()
        
        if output:
            with open(output, 'w') as f:
                json.dump(config_data, f, indent=2)
            console.print(f"✅ Configuration saved to: {output}")
        else:
            console.print(Panel(
                json.dumps(config_data, indent=2),
                title="Continue Extension Configuration",
                border_style="blue"
            ))
            
    except Exception as e:
        console.print(f"❌ Error generating config: {e}")

@continue_cli.command()
@click.argument('query')
@click.option('--port', default=8080, help='Server port')
def search(query, port):
    """Search KB Brain knowledge base"""
    
    try:
        console.print(f"🔍 Searching KB Brain for: {query}")
        
        # Prepare request
        request_data = {
            'codeSnippet': query,
            'language': 'python',
            'intent': 'completion',
            'filePath': '',
            'cursorPosition': 0,
            'selectedText': query,
            'projectRoot': ''
        }
        
        # Send request
        response = requests.post(
            f"http://localhost:{port}/kb-brain/search",
            json=request_data,
            timeout=10
        )
        
        if response.status_code == 200:
            data = response.json()
            
            console.print(f"📊 Found {data.get('kb_matches', 0)} matches")
            console.print(f"🎯 Confidence: {data.get('confidence', 0):.2f}")
            
            suggestions = data.get('suggestions', [])
            if suggestions:
                console.print("\n💡 Suggestions:")
                for i, suggestion in enumerate(suggestions, 1):
                    console.print(Panel(
                        suggestion,
                        title=f"Suggestion {i}",
                        border_style="green"
                    ))
            
            explanations = data.get('explanations', [])
            if explanations:
                console.print("\n📚 Explanations:")
                for i, explanation in enumerate(explanations, 1):
                    console.print(Panel(
                        explanation,
                        title=f"Explanation {i}",
                        border_style="blue"
                    ))
                    
        else:
            console.print(f"❌ Search failed with status {response.status_code}")
            
    except requests.ConnectionError:
        console.print(f"❌ Cannot connect to server on port {port}")
    except Exception as e:
        console.print(f"❌ Error searching: {e}")

@continue_cli.command()
@click.option('--port', default=8080, help='Server port')
@click.option('--interval', default=5, help='Update interval in seconds')
def monitor(port, interval):
    """Monitor KB Brain Continue server"""
    
    def get_status():
        try:
            response = requests.get(f"http://localhost:{port}/kb-brain/status", timeout=2)
            if response.status_code == 200:
                return response.json()
            return None
        except:
            return None
    
    def create_status_display(status_data):
        if not status_data:
            return Panel("❌ Server offline", title="KB Brain Continue Monitor", border_style="red")
        
        kb_status = status_data.get('kb_brain_status', {})
        gpu_status = kb_status.get('hybrid_gpu_status', {})
        
        content = f"""
🟢 **Server Status:** Online
🔌 **Port:** {port}
🧠 **KB Brain:** Ready
🎮 **GPU:** {'✅ Available' if gpu_status.get('gpu_available') else '❌ Not Available'}
📚 **Knowledge Embeddings:** {gpu_status.get('knowledge_embeddings', 0)}
💡 **Solution Texts:** {gpu_status.get('solution_texts', 0)}
🔧 **Brain Type:** {gpu_status.get('brain_type', 'Unknown')}
📊 **Total Solutions:** {kb_status.get('total_solutions', 0)}
📁 **KB Files:** {kb_status.get('kb_files_count', 0)}
🕐 **Last Updated:** {kb_status.get('last_updated', 'Unknown')}
        """
        
        return Panel(content, title="KB Brain Continue Monitor", border_style="green")
    
    console.print("🔍 Starting KB Brain Continue monitor...")
    console.print(f"📊 Port: {port}, Update interval: {interval}s")
    console.print("Press Ctrl+C to stop")
    
    try:
        with Live(create_status_display(None), refresh_per_second=1) as live:
            while True:
                status_data = get_status()
                live.update(create_status_display(status_data))
                time.sleep(interval)
                
    except KeyboardInterrupt:
        console.print("\n🛑 Monitor stopped")

@continue_cli.command()
def install():
    """Install Continue extension and configure KB Brain integration"""
    
    console.print("🔌 Installing KB Brain Continue Integration")
    
    try:
        # Run installation script
        import subprocess
        import os
        
        script_path = Path(__file__).parent.parent.parent / "continue_integration" / "install_continue.sh"
        
        if script_path.exists():
            result = subprocess.run([str(script_path)], cwd=str(script_path.parent))
            if result.returncode == 0:
                console.print("✅ Installation completed successfully")
            else:
                console.print("❌ Installation failed")
        else:
            console.print(f"❌ Installation script not found: {script_path}")
            
    except Exception as e:
        console.print(f"❌ Error during installation: {e}")

if __name__ == "__main__":
    continue_cli()