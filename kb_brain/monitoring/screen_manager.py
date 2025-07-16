#!/usr/bin/env python3
"""
Screen Manager for KB Brain MCP Server
Manages screen sessions for long-running tasks and monitoring
"""

import subprocess
import json
import time
import os
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

@dataclass
class ScreenSession:
    """Represents a screen session"""
    name: str
    pid: int
    status: str
    created: str
    windows: List[str]
    purpose: str

class ScreenManager:
    """Manages screen sessions for KB Brain tasks"""
    
    def __init__(self, kb_root: str = "/mnt/c/Users/misley/Documents/Projects/kb_system"):
        self.kb_root = Path(kb_root)
        self.sessions_file = self.kb_root / "screen_sessions.json"
        self.load_sessions()
    
    def load_sessions(self):
        """Load session metadata"""
        if self.sessions_file.exists():
            try:
                with open(self.sessions_file, 'r') as f:
                    self.session_metadata = json.load(f)
            except Exception:
                self.session_metadata = {}
        else:
            self.session_metadata = {}
    
    def save_sessions(self):
        """Save session metadata"""
        with open(self.sessions_file, 'w') as f:
            json.dump(self.session_metadata, f, indent=2)
    
    def list_active_sessions(self) -> List[ScreenSession]:
        """List all active screen sessions"""
        sessions = []
        
        try:
            # Get screen list
            result = subprocess.run(
                ["screen", "-list"], 
                capture_output=True, 
                text=True
            )
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                for line in lines:
                    if '\t' in line and '(' in line:
                        # Parse screen session line
                        parts = line.strip().split('\t')
                        if len(parts) >= 2:
                            session_info = parts[0]
                            if '.' in session_info:
                                pid_str, name = session_info.split('.', 1)
                                
                                # Get metadata
                                metadata = self.session_metadata.get(name, {})
                                
                                # Get windows
                                windows = self._get_session_windows(name)
                                
                                session = ScreenSession(
                                    name=name,
                                    pid=int(pid_str) if pid_str.isdigit() else 0,
                                    status=parts[1] if len(parts) > 1 else "Unknown",
                                    created=metadata.get('created', 'Unknown'),
                                    windows=windows,
                                    purpose=metadata.get('purpose', 'Unknown')
                                )
                                sessions.append(session)
        
        except Exception as e:
            print(f"Error listing sessions: {e}")
        
        return sessions
    
    def _get_session_windows(self, session_name: str) -> List[str]:
        """Get windows for a session"""
        try:
            result = subprocess.run([
                "screen", "-S", session_name, "-X", "windows"
            ], capture_output=True, text=True)
            
            if result.returncode == 0:
                return result.stdout.strip().split('\n')
        except Exception:
            pass
        
        return []
    
    def create_monitoring_session(self, task_name: str, purpose: str = "monitoring") -> Dict[str, Any]:
        """Create a monitoring session with multiple workers"""
        
        session_name = f"kb_brain_{task_name}"
        
        # Kill existing session if it exists
        self.kill_session(session_name)
        
        # Create main session
        try:
            # Create detached session
            subprocess.run([
                "screen", "-dmS", session_name, "bash", "-c", f"""
echo '🖥️  KB Brain {task_name.title()} Monitor'
echo '=================================='
echo 'Session: {session_name}'
echo 'Purpose: {purpose}'
echo 'Created: {time.strftime("%Y-%m-%d %H:%M:%S")}'
echo ''
echo 'Available workers will be created automatically...'
echo ''
echo 'To reattach: screen -r {session_name}'
exec bash
"""
            ], check=True)
            
            # Save metadata
            self.session_metadata[session_name] = {
                'created': time.strftime("%Y-%m-%d %H:%M:%S"),
                'purpose': purpose,
                'task_name': task_name,
                'workers': []
            }
            self.save_sessions()
            
            return {
                'success': True,
                'session_name': session_name,
                'message': f'Monitoring session created: {session_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def add_worker(self, session_name: str, worker_name: str, command: str) -> Dict[str, Any]:
        """Add a worker window to a session"""
        
        try:
            # Create new window in session
            subprocess.run([
                "screen", "-S", session_name, "-X", "screen", "-t", worker_name,
                "bash", "-c", command
            ], check=True)
            
            # Update metadata
            if session_name in self.session_metadata:
                if 'workers' not in self.session_metadata[session_name]:
                    self.session_metadata[session_name]['workers'] = []
                
                self.session_metadata[session_name]['workers'].append({
                    'name': worker_name,
                    'command': command,
                    'created': time.strftime("%Y-%m-%d %H:%M:%S")
                })
                self.save_sessions()
            
            return {
                'success': True,
                'message': f'Worker {worker_name} added to {session_name}'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def create_cuml_monitor(self) -> Dict[str, Any]:
        """Create CuML installation monitoring session"""
        
        # Create main session
        result = self.create_monitoring_session("cuml_install", "CuML Installation Monitoring")
        
        if not result['success']:
            return result
        
        session_name = result['session_name']
        
        # Add progress worker
        self.add_worker(session_name, "progress", f"""
echo '📊 CuML Installation Progress Monitor'
echo '===================================='
while true; do
    if [ -f /tmp/cuml_install_progress.txt ]; then
        echo "🔄 $(date '+%H:%M:%S') - Latest Progress:"
        tail -1 /tmp/cuml_install_progress.txt
        echo ""
        
        if grep -q 'completed successfully' /tmp/cuml_install_progress.txt; then
            echo '✅ Installation completed successfully!'
            break
        elif grep -q 'installation failed' /tmp/cuml_install_progress.txt; then
            echo '❌ Installation failed!'
            break
        fi
    else
        echo '⏳ Waiting for installation to start...'
    fi
    sleep 5
done
echo 'Progress monitoring finished.'
exec bash
""")
        
        # Add log worker
        self.add_worker(session_name, "logs", f"""
echo '📋 CuML Installation Log Monitor'
echo '================================'
if [ -f /tmp/cuml_install.log ]; then
    tail -f /tmp/cuml_install.log
else
    echo '📁 Waiting for log file...'
    while [ ! -f /tmp/cuml_install.log ]; do
        sleep 1
    done
    tail -f /tmp/cuml_install.log
fi
""")
        
        # Add status worker
        self.add_worker(session_name, "status", f"""
echo '🔍 CuML Installation Status Monitor'
echo '==================================='
while true; do
    echo "🎯 Status Check - $(date)"
    echo "=========================="
    
    if /tmp/kb_brain_venv/bin/python -c 'import cuml; print("CuML version:", cuml.__version__)' 2>/dev/null; then
        echo '✅ CuML is installed and working!'
        echo ""
        echo "Testing GPU ML capabilities..."
        /tmp/kb_brain_venv/bin/python -c 'from cuml.cluster import KMeans; print("✅ CuML GPU ML operations available")'
        echo ""
        echo '🎉 Installation completed successfully!'
        break
    else
        echo '⏳ CuML not yet installed'
    fi
    
    echo ""
    echo "--- Next check in 30 seconds ---"
    sleep 30
done
echo 'Status monitoring finished.'
exec bash
""")
        
        return {
            'success': True,
            'session_name': session_name,
            'message': f'CuML monitoring session created with 3 workers',
            'attach_command': f'screen -r {session_name}',
            'workers': ['progress', 'logs', 'status']
        }
    
    def create_task_monitor(self, task_name: str, command: str, 
                          log_file: Optional[str] = None) -> Dict[str, Any]:
        """Create a monitoring session for a long-running task"""
        
        session_name = f"kb_brain_{task_name}"
        
        # Create main session
        result = self.create_monitoring_session(task_name, f"Task Monitoring: {task_name}")
        
        if not result['success']:
            return result
        
        # Add task execution worker
        self.add_worker(session_name, "task", f"""
echo '🚀 Task Execution: {task_name}'
echo '============================='
echo 'Command: {command}'
echo 'Started: {time.strftime("%Y-%m-%d %H:%M:%S")}'
echo ''
{command}
echo ''
echo 'Task completed at: {time.strftime("%Y-%m-%d %H:%M:%S")}'
exec bash
""")
        
        # Add log monitor if log file specified
        if log_file:
            self.add_worker(session_name, "logs", f"""
echo '📋 Log Monitor: {task_name}'
echo '========================='
if [ -f {log_file} ]; then
    tail -f {log_file}
else
    echo '📁 Waiting for log file: {log_file}'
    while [ ! -f {log_file} ]; do
        sleep 1
    done
    tail -f {log_file}
fi
""")
        
        # Add system monitor
        self.add_worker(session_name, "system", f"""
echo '🖥️  System Monitor: {task_name}'
echo '=============================='
while true; do
    echo "--- System Status $(date '+%H:%M:%S') ---"
    echo "CPU: $(top -bn1 | grep 'Cpu(s)' | head -1)"
    echo "Memory: $(free -h | grep 'Mem:' | awk '{{print $3 "/" $2}}')"
    echo "Disk /tmp: $(df -h /tmp | tail -1 | awk '{{print $3 "/" $2 " (" $5 ")"}}')"
    echo ""
    sleep 10
done
""")
        
        return {
            'success': True,
            'session_name': session_name,
            'message': f'Task monitoring session created for {task_name}',
            'attach_command': f'screen -r {session_name}',
            'workers': ['task', 'logs', 'system'] if log_file else ['task', 'system']
        }
    
    def kill_session(self, session_name: str) -> Dict[str, Any]:
        """Kill a screen session"""
        
        try:
            subprocess.run([
                "screen", "-S", session_name, "-X", "quit"
            ], check=True)
            
            # Remove from metadata
            if session_name in self.session_metadata:
                del self.session_metadata[session_name]
                self.save_sessions()
            
            return {
                'success': True,
                'message': f'Session {session_name} killed'
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_session_status(self, session_name: str) -> Dict[str, Any]:
        """Get detailed status of a session"""
        
        sessions = self.list_active_sessions()
        session = next((s for s in sessions if s.name == session_name), None)
        
        if not session:
            return {
                'exists': False,
                'message': f'Session {session_name} not found'
            }
        
        metadata = self.session_metadata.get(session_name, {})
        
        return {
            'exists': True,
            'session': {
                'name': session.name,
                'pid': session.pid,
                'status': session.status,
                'created': session.created,
                'windows': session.windows,
                'purpose': session.purpose,
                'workers': metadata.get('workers', [])
            },
            'attach_command': f'screen -r {session_name}',
            'detach_command': 'Ctrl+A then D',
            'navigation': {
                'next_window': 'Ctrl+A then N',
                'prev_window': 'Ctrl+A then P',
                'list_windows': 'Ctrl+A then "'
            }
        }

def main():
    """Test the screen manager"""
    manager = ScreenManager()
    
    print("🖥️  Screen Manager Test")
    print("=" * 30)
    
    # List active sessions
    sessions = manager.list_active_sessions()
    print(f"Active sessions: {len(sessions)}")
    
    for session in sessions:
        print(f"  - {session.name} (PID: {session.pid})")
    
    # Create test monitoring session
    result = manager.create_cuml_monitor()
    print(f"CuML monitor: {result}")

if __name__ == "__main__":
    main()