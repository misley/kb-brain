#!/usr/bin/env python3
"""
KB Brain Hybrid GPU MCP Server
Uses CuPy for GPU acceleration with scikit-learn fallback
"""

import sys
import os
import json
import asyncio
from pathlib import Path

# Set up package path
PACKAGE_DIR = Path(__file__).parent
DATA_DIR = PACKAGE_DIR / "data"
sys.path.insert(0, str(PACKAGE_DIR))

# Set KB data path
os.environ["KB_DATA_PATH"] = str(DATA_DIR)

# Import hybrid GPU brain and screen manager
from kb_brain_hybrid_gpu import HybridGPUKBBrain
from screen_manager import ScreenManager

class HybridKBBrainMCP:
    def __init__(self):
        self.brain = HybridGPUKBBrain(kb_root=str(DATA_DIR))
        self.screen_manager = ScreenManager(kb_root="/mnt/c/Users/misley/Documents/Projects/kb_system")
        self.tools = self._define_tools()
        print(f"✅ Hybrid KB Brain MCP server initialized", file=sys.stderr)
        
        # Print GPU status
        status = self.brain.get_hybrid_status()
        print(f"🎯 GPU Available: {status['gpu_available']}", file=sys.stderr)
        print(f"🎯 GPU Enabled: {status['gpu_enabled']}", file=sys.stderr)
        print(f"🎯 Mode: {status['brain_type']}", file=sys.stderr)
    
    def _define_tools(self):
        return [
            {
                "name": "find_solution",
                "description": "Find best solution using hybrid GPU/CPU processing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "problem_description": {"type": "string"},
                        "top_k": {"type": "integer", "default": 5}
                    },
                    "required": ["problem_description"]
                }
            },
            {
                "name": "record_solution_feedback",
                "description": "Record feedback on solution effectiveness",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "solution_id": {"type": "string"},
                        "success": {"type": "boolean"},
                        "notes": {"type": "string"}
                    },
                    "required": ["solution_id", "success"]
                }
            },
            {
                "name": "get_kb_status",
                "description": "Get hybrid GPU/CPU system status",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_gpu_info": {"type": "boolean", "default": True}
                    }
                }
            },
            {
                "name": "rebuild_knowledge_index",
                "description": "Rebuild knowledge index with hybrid processing",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "force": {"type": "boolean", "default": False}
                    }
                }
            },
            {
                "name": "create_screen_monitor",
                "description": "Create screen-based monitoring session with multiple workers",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "monitor_type": {"type": "string", "enum": ["cuml", "task"], "default": "task"},
                        "command": {"type": "string"},
                        "log_file": {"type": "string"}
                    },
                    "required": ["task_name"]
                }
            },
            {
                "name": "list_screen_sessions",
                "description": "List all active screen sessions",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "include_details": {"type": "boolean", "default": True}
                    }
                }
            },
            {
                "name": "get_screen_session",
                "description": "Get detailed information about a specific screen session",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_name": {"type": "string"}
                    },
                    "required": ["session_name"]
                }
            },
            {
                "name": "kill_screen_session",
                "description": "Kill a screen session and clean up resources",
                "inputSchema": {
                    "type": "object",
                    "properties": {
                        "session_name": {"type": "string"}
                    },
                    "required": ["session_name"]
                }
            }
        ]
    
    async def handle_request(self, request):
        method = request.get("method")
        params = request.get("params", {})
        
        if method == "tools/list":
            return {"tools": self.tools}
        elif method == "tools/call":
            return await self._handle_tool_call(params)
        else:
            return {"error": f"Unknown method: {method}"}
    
    async def _handle_tool_call(self, params):
        tool_name = params.get("name")
        args = params.get("arguments", {})
        
        if tool_name == "find_solution":
            return await self._find_solution(args)
        elif tool_name == "record_solution_feedback":
            return await self._record_feedback(args)
        elif tool_name == "get_kb_status":
            return await self._get_status(args)
        elif tool_name == "rebuild_knowledge_index":
            return await self._rebuild_index(args)
        elif tool_name == "create_screen_monitor":
            return await self._create_screen_monitor(args)
        elif tool_name == "list_screen_sessions":
            return await self._list_screen_sessions(args)
        elif tool_name == "get_screen_session":
            return await self._get_screen_session(args)
        elif tool_name == "kill_screen_session":
            return await self._kill_screen_session(args)
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    async def _find_solution(self, args):
        problem = args["problem_description"]
        top_k = args.get("top_k", 5)
        
        try:
            # Use hybrid GPU/CPU search
            solutions = self.brain.find_best_solution_hybrid(problem, top_k=top_k)
            
            if not solutions:
                return {"content": [{"type": "text", "text": "No solutions found."}]}
            
            # Format response with GPU indicators
            response_parts = [
                f"🧠 Found {len(solutions)} solutions using hybrid GPU/CPU processing",
                f"🔍 Query: {problem}\\n"
            ]
            
            for i, solution in enumerate(solutions, 1):
                response_parts.append(f"## Solution {i}")
                response_parts.append(f"**🎯 Similarity Score:** {solution.similarity_score:.3f}")
                response_parts.append(f"**🔧 Confidence:** {solution.confidence:.3f}")
                response_parts.append(f"**📊 Success Rate:** {solution.success_rate:.3f}")
                response_parts.append(f"**📚 Source:** {solution.source_kb}")
                response_parts.append(f"**🔍 Context:** {solution.problem_context[:200]}...")
                response_parts.append(f"**💡 Solution:** {solution.solution_text[:400]}...")
                
                if solution.tags:
                    response_parts.append(f"**🏷️ Tags:** {', '.join(solution.tags)}")
                
                response_parts.append(f"**🆔 ID:** `{solution.solution_id}`\\n")
            
            # Add GPU status footer
            gpu_status = self.brain.get_hybrid_status()
            response_parts.append("---")
            response_parts.append(f"**🎮 GPU Acceleration:** {'✅ Active' if gpu_status['gpu_enabled'] else '❌ Disabled'}")
            response_parts.append(f"**🖥️ Processing Mode:** {gpu_status['brain_type']}")
            
            return {"content": [{"type": "text", "text": "\\n".join(response_parts)}]}
            
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error: {str(e)}"}]}
    
    async def _record_feedback(self, args):
        solution_id = args["solution_id"]
        success = args["success"]
        notes = args.get("notes", "")
        
        try:
            self.brain.record_solution_feedback(solution_id, success)
            status = "✅ successful" if success else "❌ unsuccessful"
            response = f"📝 Feedback recorded for `{solution_id}`: {status}"
            if notes:
                response += f"\\n📋 Notes: {notes}"
            return {"content": [{"type": "text", "text": response}]}
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error: {str(e)}"}]}
    
    async def _get_status(self, args):
        include_gpu = args.get("include_gpu_info", True)
        
        try:
            status = self.brain.get_system_status()
            hybrid_status = status.get('hybrid_gpu_status', {})
            
            response_parts = [
                "# 🧠 KB Brain Hybrid System Status",
                f"**🎮 GPU Available:** {'✅ Yes' if hybrid_status.get('gpu_available') else '❌ No'}",
                f"**🎯 GPU Enabled:** {'✅ Yes' if hybrid_status.get('gpu_enabled') else '❌ No'}",
                f"**🖥️ Processing Mode:** {hybrid_status.get('brain_type', 'Unknown')}",
                f"**📚 Knowledge Embeddings:** {hybrid_status.get('knowledge_embeddings', 0)}",
                f"**📖 Solution Texts:** {hybrid_status.get('solution_texts', 0)}",
                f"**🎯 Total Solutions:** {status.get('total_solutions', 0)}",
                f"**📈 Average Success Rate:** {status.get('avg_success_rate', 0):.2%}",
                f"**📁 KB Files:** {status.get('kb_files_count', 0)}",
                f"**🕐 Last Updated:** {status.get('last_updated', 'Unknown')}"
            ]
            
            if include_gpu and hybrid_status.get('gpu_available'):
                response_parts.append("\\n**🎮 GPU Capabilities:**")
                response_parts.append("- ✅ CuPy vector operations")
                response_parts.append("- ✅ GPU-accelerated similarity search")
                response_parts.append("- ✅ Automatic CPU fallback")
                response_parts.append("- ⚠️ CuML not available (using scikit-learn)")
            
            return {"content": [{"type": "text", "text": "\\n".join(response_parts)}]}
            
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error: {str(e)}"}]}
    
    async def _rebuild_index(self, args):
        force = args.get("force", False)
        
        try:
            self.brain.rebuild_knowledge_index()
            gpu_status = self.brain.get_hybrid_status()
            method = "🎮 Hybrid GPU/CPU" if gpu_status['gpu_enabled'] else "🖥️ CPU"
            
            response_parts = [
                f"✅ Knowledge index rebuilt successfully",
                f"🔧 Method: {method}",
                f"📊 Embeddings: {gpu_status.get('knowledge_embeddings', 0)}",
                f"📚 Texts: {gpu_status.get('solution_texts', 0)}"
            ]
            
            return {"content": [{"type": "text", "text": "\\n".join(response_parts)}]}
            
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error: {str(e)}"}]}
    
    async def _create_screen_monitor(self, args):
        """Create a screen monitoring session"""
        task_name = args["task_name"]
        monitor_type = args.get("monitor_type", "task")
        command = args.get("command", "")
        log_file = args.get("log_file", "")
        
        try:
            if monitor_type == "cuml":
                result = self.screen_manager.create_cuml_monitor()
            else:
                result = self.screen_manager.create_task_monitor(
                    task_name=task_name,
                    command=command,
                    log_file=log_file if log_file else None
                )
            
            if result.get("success"):
                response_parts = [
                    f"✅ Screen monitoring session created successfully",
                    f"📛 **Session Name:** `{result['session_name']}`",
                    f"🎯 **Purpose:** {monitor_type.title()} Monitoring",
                    f"👥 **Workers:** {', '.join(result.get('workers', []))}"
                ]
                
                if 'attach_command' in result:
                    response_parts.append(f"🖥️  **Attach Command:** `{result['attach_command']}`")
                
                response_parts.extend([
                    "",
                    "**🎮 Screen Navigation:**",
                    "- `Ctrl+A then N` - Next window",
                    "- `Ctrl+A then P` - Previous window", 
                    "- `Ctrl+A then \"` - List windows",
                    "- `Ctrl+A then D` - Detach (keeps running)"
                ])
                
                return {"content": [{"type": "text", "text": "\\n".join(response_parts)}]}
            else:
                return {"content": [{"type": "text", "text": f"❌ Failed to create screen session: {result.get('error', 'Unknown error')}"}]}
                
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error creating screen session: {str(e)}"}]}
    
    async def _list_screen_sessions(self, args):
        """List all active screen sessions"""
        include_details = args.get("include_details", True)
        
        try:
            sessions = self.screen_manager.list_active_sessions()
            
            if not sessions:
                return {"content": [{"type": "text", "text": "📭 No active screen sessions found"}]}
            
            response_parts = [
                f"🖥️  **Active Screen Sessions ({len(sessions)})**",
                "=" * 40
            ]
            
            for session in sessions:
                response_parts.append(f"**📛 {session.name}**")
                response_parts.append(f"- **PID:** {session.pid}")
                response_parts.append(f"- **Status:** {session.status}")
                response_parts.append(f"- **Purpose:** {session.purpose}")
                response_parts.append(f"- **Created:** {session.created}")
                
                if include_details and session.windows:
                    response_parts.append(f"- **Windows:** {', '.join(session.windows)}")
                
                response_parts.append(f"- **Attach:** `screen -r {session.name}`")
                response_parts.append("")
            
            return {"content": [{"type": "text", "text": "\\n".join(response_parts)}]}
            
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error listing sessions: {str(e)}"}]}
    
    async def _get_screen_session(self, args):
        """Get detailed information about a specific screen session"""
        session_name = args["session_name"]
        
        try:
            status = self.screen_manager.get_session_status(session_name)
            
            if not status.get("exists"):
                return {"content": [{"type": "text", "text": f"❌ Session '{session_name}' not found"}]}
            
            session = status["session"]
            
            response_parts = [
                f"🖥️  **Screen Session: {session_name}**",
                "=" * 40,
                f"**📛 Name:** {session['name']}",
                f"**🆔 PID:** {session['pid']}",
                f"**📊 Status:** {session['status']}",
                f"**🎯 Purpose:** {session['purpose']}",
                f"**📅 Created:** {session['created']}",
                ""
            ]
            
            if session.get('windows'):
                response_parts.append("**🪟 Windows:**")
                for window in session['windows']:
                    response_parts.append(f"- {window}")
                response_parts.append("")
            
            if session.get('workers'):
                response_parts.append("**👥 Workers:**")
                for worker in session['workers']:
                    response_parts.append(f"- **{worker['name']}** (created: {worker['created']})")
                response_parts.append("")
            
            response_parts.extend([
                "**🎮 Commands:**",
                f"- **Attach:** `{status['attach_command']}`",
                f"- **Detach:** {status['detach_command']}",
                "",
                "**🎮 Navigation:**",
                f"- **Next Window:** {status['navigation']['next_window']}",
                f"- **Previous Window:** {status['navigation']['prev_window']}",
                f"- **List Windows:** {status['navigation']['list_windows']}"
            ])
            
            return {"content": [{"type": "text", "text": "\\n".join(response_parts)}]}
            
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error getting session info: {str(e)}"}]}
    
    async def _kill_screen_session(self, args):
        """Kill a screen session"""
        session_name = args["session_name"]
        
        try:
            result = self.screen_manager.kill_session(session_name)
            
            if result.get("success"):
                return {"content": [{"type": "text", "text": f"✅ Screen session '{session_name}' killed successfully"}]}
            else:
                return {"content": [{"type": "text", "text": f"❌ Failed to kill session: {result.get('error', 'Unknown error')}"}]}
                
        except Exception as e:
            return {"content": [{"type": "text", "text": f"❌ Error killing session: {str(e)}"}]}
    
    async def run(self):
        try:
            while True:
                line = await asyncio.get_event_loop().run_in_executor(None, sys.stdin.readline)
                if not line:
                    break
                
                try:
                    request = json.loads(line.strip())
                    response = await self.handle_request(request)
                    print(json.dumps(response), flush=True)
                except json.JSONDecodeError:
                    print(json.dumps({"error": "Invalid JSON"}), flush=True)
        except KeyboardInterrupt:
            print("🛑 Hybrid KB Brain server stopped", file=sys.stderr)
        except Exception as e:
            print(f"❌ Server error: {e}", file=sys.stderr)

def main():
    server = HybridKBBrainMCP()
    asyncio.run(server.run())

if __name__ == "__main__":
    main()