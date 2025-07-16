#!/usr/bin/env python3
"""
KB Brain Startup Optimizer
Manages persistent venv and fast startup without rebuilding
"""

import os
import sys
import json
import time
import subprocess
from pathlib import Path
from typing import Dict, Optional

class KBBrainStartupOptimizer:
    """Optimizes KB Brain startup performance"""
    
    def __init__(self):
        self.kb_root = Path("/mnt/c/Users/misley/Documents/Projects/kb_system")
        self.venv_path = Path("/tmp/kb_brain_venv")
        self.persistent_state_file = self.kb_root / "kb_brain_persistent_state.json"
        self.startup_cache_file = self.kb_root / "startup_cache.json"
        
    def check_venv_health(self) -> Dict[str, bool]:
        """Check if virtual environment is healthy and ready"""
        
        health_status = {
            "venv_exists": self.venv_path.exists(),
            "python_works": False,
            "kb_brain_installed": False,
            "dependencies_ok": False,
            "gpu_ready": False
        }
        
        if not health_status["venv_exists"]:
            return health_status
            
        try:
            # Test Python execution
            result = subprocess.run([
                str(self.venv_path / "bin" / "python"), 
                "-c", "import sys; print('Python OK')"
            ], capture_output=True, text=True, timeout=5)
            
            health_status["python_works"] = (result.returncode == 0)
            
            if health_status["python_works"]:
                # Test KB Brain import
                result = subprocess.run([
                    str(self.venv_path / "bin" / "python"), 
                    "-c", "import kb_brain; print('KB Brain OK')"
                ], capture_output=True, text=True, timeout=5)
                
                health_status["kb_brain_installed"] = (result.returncode == 0)
                
                # Test dependencies
                result = subprocess.run([
                    str(self.venv_path / "bin" / "python"), 
                    "-c", "import sklearn, numpy; print('Dependencies OK')"
                ], capture_output=True, text=True, timeout=5)
                
                health_status["dependencies_ok"] = (result.returncode == 0)
                
                # Test GPU readiness
                result = subprocess.run([
                    str(self.venv_path / "bin" / "python"), 
                    "-c", "import cupy; print('GPU Ready')"
                ], capture_output=True, text=True, timeout=5)
                
                health_status["gpu_ready"] = (result.returncode == 0)
                
        except Exception as e:
            print(f"⚠️  Health check error: {e}")
            
        return health_status
    
    def save_persistent_state(self, state: Dict):
        """Save persistent state to avoid rebuilding"""
        state["last_updated"] = time.time()
        state["venv_path"] = str(self.venv_path)
        
        with open(self.persistent_state_file, 'w') as f:
            json.dump(state, f, indent=2)
            
        print(f"💾 Persistent state saved to {self.persistent_state_file}")
    
    def load_persistent_state(self) -> Optional[Dict]:
        """Load persistent state if available"""
        if self.persistent_state_file.exists():
            try:
                with open(self.persistent_state_file, 'r') as f:
                    state = json.load(f)
                    print(f"📂 Loaded persistent state from {self.persistent_state_file}")
                    return state
            except Exception as e:
                print(f"⚠️  Error loading persistent state: {e}")
        return None
    
    def optimize_startup(self) -> Dict:
        """Optimize startup performance"""
        
        optimization_report = {
            "startup_time": time.time(),
            "venv_health": {},
            "optimization_actions": [],
            "ready_for_use": False
        }
        
        print("🚀 KB Brain Startup Optimizer")
        print("=" * 40)
        
        # Check existing state
        persistent_state = self.load_persistent_state()
        
        # Check venv health
        health = self.check_venv_health()
        optimization_report["venv_health"] = health
        
        print("🔍 Virtual Environment Health Check:")
        for check, status in health.items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check.replace('_', ' ').title()}")
        
        # Determine optimization actions
        if not health["venv_exists"]:
            optimization_report["optimization_actions"].append("Create virtual environment")
            print("🔧 Action needed: Create virtual environment")
        
        if not health["python_works"]:
            optimization_report["optimization_actions"].append("Fix Python installation")
            print("🔧 Action needed: Fix Python installation")
        
        if not health["kb_brain_installed"]:
            optimization_report["optimization_actions"].append("Install KB Brain package")
            print("🔧 Action needed: Install KB Brain package")
        
        if not health["dependencies_ok"]:
            optimization_report["optimization_actions"].append("Install dependencies")
            print("🔧 Action needed: Install dependencies")
        
        # Check if ready for immediate use
        if all([health["venv_exists"], health["python_works"], 
                health["kb_brain_installed"], health["dependencies_ok"]]):
            optimization_report["ready_for_use"] = True
            print("✅ KB Brain ready for immediate use!")
            
            # Update persistent state
            self.save_persistent_state({
                "healthy": True,
                "last_health_check": time.time(),
                "gpu_available": health["gpu_ready"]
            })
        else:
            print("⚠️  KB Brain needs setup before use")
        
        return optimization_report
    
    def get_fast_startup_command(self) -> str:
        """Get command for fast startup"""
        
        # Check if everything is ready
        health = self.check_venv_health()
        
        if all([health["venv_exists"], health["python_works"], 
                health["kb_brain_installed"], health["dependencies_ok"]]):
            
            # Return direct command without rebuilding
            return f"{self.venv_path}/bin/python -m kb_brain"
        else:
            return "# Environment not ready - run optimization first"
    
    def create_startup_script(self):
        """Create optimized startup script"""
        
        startup_script = self.kb_root / "start_kb_brain_optimized.sh"
        
        script_content = f"""#!/bin/bash
# KB Brain Optimized Startup Script
# Avoids rebuilding venv each time

VENV_PATH="{self.venv_path}"
KB_ROOT="{self.kb_root}"

echo "🚀 KB Brain Fast Startup"

# Quick health check
if [ -f "$VENV_PATH/bin/python" ]; then
    echo "✅ Virtual environment ready"
    
    # Test KB Brain import
    if $VENV_PATH/bin/python -c "import kb_brain" 2>/dev/null; then
        echo "✅ KB Brain package ready"
        
        # Start KB Brain
        echo "🧠 Starting KB Brain..."
        $VENV_PATH/bin/python -c "
import sys
sys.path.insert(0, '/tmp/kb_brain_venv/lib/python3.12/site-packages/kb_brain')
from kb_brain_hybrid_gpu import HybridGPUKBBrain
brain = HybridGPUKBBrain()
print('🎯 KB Brain ready for use!')
print('📊 Status:', brain.get_hybrid_status())
"
    else
        echo "❌ KB Brain not installed - run setup first"
        exit 1
    fi
else
    echo "❌ Virtual environment not found - run setup first"
    exit 1
fi
"""
        
        with open(startup_script, 'w') as f:
            f.write(script_content)
            
        os.chmod(startup_script, 0o755)
        print(f"📝 Optimized startup script created: {startup_script}")
        
        return startup_script
    
    def test_performance(self):
        """Test startup performance"""
        
        print("⏱️  Testing startup performance...")
        
        start_time = time.time()
        
        # Test KB Brain initialization
        try:
            result = subprocess.run([
                str(self.venv_path / "bin" / "python"), 
                "-c", """
import time
import sys
sys.path.insert(0, '/tmp/kb_brain_venv/lib/python3.12/site-packages/kb_brain')
start = time.time()
from kb_brain_hybrid_gpu import HybridGPUKBBrain
brain = HybridGPUKBBrain()
end = time.time()
print(f'Initialization time: {end - start:.2f} seconds')
print(f'Status: {brain.get_hybrid_status()}')
"""
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                print("✅ Performance test successful:")
                print(result.stdout)
            else:
                print("❌ Performance test failed:")
                print(result.stderr)
                
        except Exception as e:
            print(f"❌ Performance test error: {e}")
        
        total_time = time.time() - start_time
        print(f"📊 Total test time: {total_time:.2f} seconds")

def main():
    """Main function"""
    optimizer = KBBrainStartupOptimizer()
    
    print("🎯 KB Brain Startup Optimization")
    print("=" * 50)
    
    # Run optimization
    report = optimizer.optimize_startup()
    
    print("\n📈 Optimization Report:")
    print("-" * 30)
    print(f"Ready for use: {'✅ Yes' if report['ready_for_use'] else '❌ No'}")
    print(f"Actions needed: {len(report['optimization_actions'])}")
    
    if report["ready_for_use"]:
        # Create startup script
        startup_script = optimizer.create_startup_script()
        
        # Test performance
        optimizer.test_performance()
        
        print(f"\n🚀 Fast startup command:")
        print(f"   {optimizer.get_fast_startup_command()}")
        print(f"\n📝 Or use optimized script:")
        print(f"   {startup_script}")
    else:
        print("\n🔧 Setup required before optimization")

if __name__ == "__main__":
    main()