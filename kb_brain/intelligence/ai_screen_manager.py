"""
AI Screen Manager for Background Processing
Manages screen sessions for long-running AI processing tasks with intelligent monitoring
"""

import json
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
import logging
import tempfile
import os
from pathlib import Path

logger = logging.getLogger(__name__)


class TaskType(Enum):
    """Types of AI processing tasks"""
    PROMPT_PROCESSING = "prompt_processing"
    RESEARCH_ANALYSIS = "research_analysis"
    CROSS_REPO_ANALYSIS = "cross_repo_analysis"
    ITERATIVE_ANALYSIS = "iterative_analysis"
    KB_CONSOLIDATION = "kb_consolidation"
    CUSTOM = "custom"


class TaskStatus(Enum):
    """Task processing status"""
    CREATED = "created"
    STARTING = "starting"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"
    CANCELLED = "cancelled"


@dataclass
class AITask:
    """AI processing task definition"""
    id: str
    type: TaskType
    prompt: str
    context: Dict[str, Any]
    priority: str = "normal"
    max_runtime: int = 1800  # 30 minutes default
    retry_count: int = 0
    max_retries: int = 2
    created_at: datetime = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


@dataclass
class AISessionInfo:
    """AI processing session information"""
    session_name: str
    task: AITask
    status: TaskStatus
    output_file: str
    log_file: str
    progress_file: str
    screen_pid: Optional[int] = None
    progress: Dict[str, Any] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if self.progress is None:
            self.progress = {'stage': 'initializing', 'percent': 0}


class AIScreenManager:
    """Manages AI processing sessions in screen environments"""
    
    def __init__(self, 
                 work_dir: str = "/tmp/kb_brain_ai_sessions",
                 ai_processor: Optional[Callable] = None,
                 max_concurrent_sessions: int = 5):
        """
        Initialize AI Screen Manager
        
        Args:
            work_dir: Directory for session files
            ai_processor: AI processing function
            max_concurrent_sessions: Maximum concurrent sessions
        """
        self.work_dir = Path(work_dir)
        self.work_dir.mkdir(exist_ok=True)
        
        self.ai_processor = ai_processor
        self.max_concurrent_sessions = max_concurrent_sessions
        
        # Session tracking
        self.active_sessions: Dict[str, AISessionInfo] = {}
        self.session_history: List[AISessionInfo] = []
        
        # Task queue
        self.pending_tasks: List[AITask] = []
        
        # Cleanup old sessions
        self._cleanup_old_sessions()
        
        logger.info(f"AI Screen Manager initialized with work_dir: {self.work_dir}")
    
    def create_processing_session(self, 
                                session_name: str,
                                task_type: str,
                                prompt: str,
                                context: Dict[str, Any],
                                priority: str = "normal",
                                max_runtime: int = 1800) -> Dict[str, Any]:
        """
        Create a new AI processing session
        
        Args:
            session_name: Unique session name
            task_type: Type of AI processing task
            prompt: The prompt to process
            context: Processing context
            priority: Task priority
            max_runtime: Maximum runtime in seconds
        
        Returns:
            Session information dictionary
        """
        
        # Check concurrent session limit
        if len(self.active_sessions) >= self.max_concurrent_sessions:
            raise Exception(f"Maximum concurrent sessions ({self.max_concurrent_sessions}) reached")
        
        # Create task
        task = AITask(
            id=str(uuid.uuid4()),
            type=TaskType(task_type),
            prompt=prompt,
            context=context,
            priority=priority,
            max_runtime=max_runtime
        )
        
        # Create session files
        session_dir = self.work_dir / session_name
        session_dir.mkdir(exist_ok=True)
        
        output_file = str(session_dir / "output.json")
        log_file = str(session_dir / "process.log")
        progress_file = str(session_dir / "progress.json")
        
        # Create session info
        session_info = AISessionInfo(
            session_name=session_name,
            task=task,
            status=TaskStatus.CREATED,
            output_file=output_file,
            log_file=log_file,
            progress_file=progress_file
        )
        
        # Create processing script
        script_file = self._create_processing_script(session_info)
        
        try:
            # Start screen session
            screen_cmd = [
                'screen', '-dmS', session_name,
                'bash', str(script_file)
            ]
            
            result = subprocess.run(screen_cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                raise Exception(f"Failed to start screen session: {result.stderr}")
            
            # Get screen PID
            session_info.screen_pid = self._get_screen_pid(session_name)
            session_info.status = TaskStatus.STARTING
            
            # Track session
            self.active_sessions[session_name] = session_info
            
            logger.info(f"Created AI processing session: {session_name}")
            
            return {
                'session_name': session_name,
                'task_id': task.id,
                'status': session_info.status.value,
                'created_at': task.created_at.isoformat(),
                'max_runtime': max_runtime
            }
            
        except Exception as e:
            logger.error(f"Failed to create processing session {session_name}: {e}")
            raise
    
    def _create_processing_script(self, session_info: AISessionInfo) -> Path:
        """Create the processing script for the session"""
        
        session_dir = Path(session_info.output_file).parent
        script_file = session_dir / "process.sh"
        
        # Python script content for AI processing
        python_script = f'''
import json
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

# Add KB Brain to path
sys.path.insert(0, "/tmp/kb_brain_venv/lib/python3.12/site-packages")

def update_progress(stage, percent, message=""):
    progress = {{
        "stage": stage,
        "percent": percent,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }}
    with open("{session_info.progress_file}", "w") as f:
        json.dump(progress, f)

def log_message(message):
    timestamp = datetime.now().isoformat()
    with open("{session_info.log_file}", "a") as f:
        f.write(f"[{{timestamp}}] {{message}}\\n")

def main():
    try:
        # Load task information
        task_data = {json.dumps(asdict(session_info.task))}
        
        log_message("Starting AI processing task")
        update_progress("initializing", 10, "Loading task data")
        
        # Initialize processing
        update_progress("processing", 30, "Running AI analysis")
        
        # Simulate AI processing stages
        stages = [
            ("analyzing_prompt", 40, "Analyzing prompt"),
            ("kb_search", 60, "Searching knowledge base"),
            ("ai_processing", 80, "Running AI processing"),
            ("synthesizing", 95, "Synthesizing results")
        ]
        
        results = {{
            "task_id": task_data["id"],
            "prompt": task_data["prompt"],
            "response": "",
            "insights": {{}},
            "confidence": 0.8,
            "processing_stages": []
        }}
        
        for stage, percent, message in stages:
            update_progress(stage, percent, message)
            log_message(f"Stage: {{stage}} - {{message}}")
            time.sleep(2)  # Simulate processing time
            
            results["processing_stages"].append({{
                "stage": stage,
                "completed_at": datetime.now().isoformat(),
                "status": "completed"
            }})
        
        # Generate final response based on task type
        task_type = task_data["type"]
        if task_type == "prompt_processing":
            results["response"] = f"Processed prompt: {{task_data['prompt']}}"
        elif task_type == "research_analysis":
            results["response"] = "Comprehensive research analysis completed"
            results["insights"]["research_themes"] = ["theme1", "theme2", "theme3"]
        elif task_type == "cross_repo_analysis":
            results["response"] = "Cross-repository analysis completed"
            results["insights"]["repositories_analyzed"] = ["repo1", "repo2"]
        else:
            results["response"] = f"Custom processing completed for type: {{task_type}}"
        
        # Save results
        with open("{session_info.output_file}", "w") as f:
            json.dump(results, f, indent=2)
        
        update_progress("completed", 100, "Processing completed successfully")
        log_message("AI processing completed successfully")
        
    except Exception as e:
        error_msg = f"Processing failed: {{str(e)}}"
        log_message(error_msg)
        log_message(traceback.format_exc())
        
        error_result = {{
            "error": str(e),
            "traceback": traceback.format_exc(),
            "status": "failed"
        }}
        
        with open("{session_info.output_file}", "w") as f:
            json.dump(error_result, f, indent=2)
        
        update_progress("failed", 0, error_msg)

if __name__ == "__main__":
    main()
'''
        
        # Bash script wrapper
        bash_script = f'''#!/bin/bash

# Set up environment
cd {session_dir}
echo "Starting AI processing session: {session_info.session_name}" >> {session_info.log_file}

# Activate virtual environment if available
if [ -d "/tmp/kb_brain_venv" ]; then
    source /tmp/kb_brain_venv/bin/activate
fi

# Run Python processing script
python3 -c '{python_script}'

# Mark completion
echo "Session completed at $(date)" >> {session_info.log_file}
'''
        
        # Write script file
        with open(script_file, 'w') as f:
            f.write(bash_script)
        
        # Make executable
        script_file.chmod(0o755)
        
        return script_file
    
    def _get_screen_pid(self, session_name: str) -> Optional[int]:
        """Get PID of screen session"""
        
        try:
            result = subprocess.run(
                ['screen', '-list'], 
                capture_output=True, 
                text=True
            )
            
            for line in result.stdout.split('\n'):
                if session_name in line:
                    # Extract PID from screen list output
                    parts = line.strip().split()
                    if parts:
                        pid_part = parts[0]
                        if '.' in pid_part:
                            return int(pid_part.split('.')[0])
            
        except Exception as e:
            logger.error(f"Error getting screen PID: {e}")
        
        return None
    
    def get_session_status(self, session_name: str) -> Dict[str, Any]:
        """Get current status of a processing session"""
        
        if session_name not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.active_sessions[session_name]
        
        # Update session status
        self._update_session_status(session_info)
        
        # Load progress information
        progress = self._load_progress(session_info.progress_file)
        
        return {
            'session_name': session_name,
            'task_id': session_info.task.id,
            'status': session_info.status.value,
            'progress': progress,
            'runtime': self._calculate_runtime(session_info.task),
            'max_runtime': session_info.task.max_runtime,
            'completed': session_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED]
        }
    
    def get_session_output(self, session_name: str) -> Dict[str, Any]:
        """Get output from a completed session"""
        
        if session_name not in self.active_sessions:
            return {'error': 'Session not found'}
        
        session_info = self.active_sessions[session_name]
        
        # Load output file
        try:
            if os.path.exists(session_info.output_file):
                with open(session_info.output_file, 'r') as f:
                    output = json.load(f)
                return output
            else:
                return {'error': 'Output file not found'}
        except Exception as e:
            return {'error': f'Failed to load output: {str(e)}'}
    
    def _update_session_status(self, session_info: AISessionInfo):
        """Update session status based on current state"""
        
        # Check if screen session is still running
        if session_info.screen_pid:
            try:
                # Check if process exists
                subprocess.run(
                    ['kill', '-0', str(session_info.screen_pid)], 
                    check=True, 
                    capture_output=True
                )
                
                # Process exists, check progress
                progress = self._load_progress(session_info.progress_file)
                if progress.get('stage') == 'completed':
                    session_info.status = TaskStatus.COMPLETED
                    session_info.task.completed_at = datetime.now()
                elif progress.get('stage') == 'failed':
                    session_info.status = TaskStatus.FAILED
                else:
                    session_info.status = TaskStatus.RUNNING
                    
            except subprocess.CalledProcessError:
                # Process doesn't exist
                if session_info.status == TaskStatus.RUNNING:
                    # Check if completed normally
                    progress = self._load_progress(session_info.progress_file)
                    if progress.get('stage') == 'completed':
                        session_info.status = TaskStatus.COMPLETED
                    else:
                        session_info.status = TaskStatus.FAILED
                        
        # Check for timeout
        if session_info.status == TaskStatus.RUNNING:
            runtime = self._calculate_runtime(session_info.task)
            if runtime > session_info.task.max_runtime:
                session_info.status = TaskStatus.TIMEOUT
                self._kill_session(session_info.session_name)
    
    def _load_progress(self, progress_file: str) -> Dict[str, Any]:
        """Load progress information from file"""
        
        try:
            if os.path.exists(progress_file):
                with open(progress_file, 'r') as f:
                    return json.load(f)
        except Exception as e:
            logger.error(f"Error loading progress file: {e}")
        
        return {'stage': 'unknown', 'percent': 0}
    
    def _calculate_runtime(self, task: AITask) -> int:
        """Calculate current runtime in seconds"""
        
        if task.started_at:
            end_time = task.completed_at or datetime.now()
            return int((end_time - task.started_at).total_seconds())
        
        return 0
    
    def list_sessions(self) -> List[Dict[str, Any]]:
        """List all active sessions"""
        
        sessions = []
        
        for session_name, session_info in self.active_sessions.items():
            self._update_session_status(session_info)
            
            sessions.append({
                'session_name': session_name,
                'task_id': session_info.task.id,
                'task_type': session_info.task.type.value,
                'status': session_info.status.value,
                'created_at': session_info.task.created_at.isoformat(),
                'runtime': self._calculate_runtime(session_info.task),
                'priority': session_info.task.priority
            })
        
        return sorted(sessions, key=lambda x: x['created_at'], reverse=True)
    
    def kill_session(self, session_name: str) -> bool:
        """Kill a running session"""
        
        if session_name not in self.active_sessions:
            return False
        
        session_info = self.active_sessions[session_name]
        success = self._kill_session(session_name)
        
        if success:
            session_info.status = TaskStatus.CANCELLED
            logger.info(f"Killed session: {session_name}")
        
        return success
    
    def _kill_session(self, session_name: str) -> bool:
        """Internal method to kill a screen session"""
        
        try:
            # Kill screen session
            result = subprocess.run(
                ['screen', '-S', session_name, '-X', 'quit'],
                capture_output=True,
                text=True
            )
            
            return result.returncode == 0
            
        except Exception as e:
            logger.error(f"Error killing session {session_name}: {e}")
            return False
    
    def cleanup_completed_sessions(self):
        """Clean up completed sessions"""
        
        completed_sessions = []
        
        for session_name, session_info in list(self.active_sessions.items()):
            self._update_session_status(session_info)
            
            if session_info.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.TIMEOUT]:
                completed_sessions.append(session_name)
                self.session_history.append(session_info)
                del self.active_sessions[session_name]
        
        logger.info(f"Cleaned up {len(completed_sessions)} completed sessions")
        
        # Keep history manageable
        if len(self.session_history) > 100:
            self.session_history = self.session_history[-50:]
        
        return completed_sessions
    
    def _cleanup_old_sessions(self):
        """Clean up old session files and directories"""
        
        try:
            # Remove session directories older than 7 days
            cutoff_time = datetime.now() - timedelta(days=7)
            
            for session_dir in self.work_dir.iterdir():
                if session_dir.is_dir():
                    dir_mtime = datetime.fromtimestamp(session_dir.stat().st_mtime)
                    if dir_mtime < cutoff_time:
                        logger.info(f"Removing old session directory: {session_dir}")
                        subprocess.run(['rm', '-rf', str(session_dir)], capture_output=True)
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get manager status"""
        
        # Update all session statuses
        for session_info in self.active_sessions.values():
            self._update_session_status(session_info)
        
        status_counts = {}
        for status in TaskStatus:
            status_counts[status.value] = len([
                s for s in self.active_sessions.values() 
                if s.status == status
            ])
        
        return {
            'active_sessions': len(self.active_sessions),
            'max_concurrent': self.max_concurrent_sessions,
            'session_history_count': len(self.session_history),
            'status_breakdown': status_counts,
            'work_directory': str(self.work_dir),
            'ai_processor_available': self.ai_processor is not None
        }


def test_ai_screen_manager():
    """Test the AI screen manager"""
    
    manager = AIScreenManager()
    
    # Test session creation
    session_info = manager.create_processing_session(
        session_name="test_session",
        task_type="prompt_processing",
        prompt="Test prompt for AI processing",
        context={'test': True},
        max_runtime=60
    )
    
    print(f"Created session: {session_info}")
    
    # Monitor progress
    for i in range(10):
        time.sleep(3)
        status = manager.get_session_status("test_session")
        print(f"Status check {i+1}: {status}")
        
        if status.get('completed', False):
            output = manager.get_session_output("test_session")
            print(f"Final output: {output}")
            break
    
    # List sessions
    sessions = manager.list_sessions()
    print(f"Active sessions: {sessions}")
    
    # Cleanup
    manager.cleanup_completed_sessions()
    
    # Manager status
    status = manager.get_status()
    print(f"Manager status: {status}")


if __name__ == "__main__":
    test_ai_screen_manager()