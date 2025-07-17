"""
CPU Performance Optimizer with Intel Extensions
Automatically enables Intel optimizations when available
"""

import os
import sys
import logging
from typing import Dict, Any, Optional, Tuple
import warnings

logger = logging.getLogger(__name__)


class CPUOptimizer:
    """CPU optimization manager with Intel extensions support"""
    
    def __init__(self, auto_optimize: bool = True):
        """
        Initialize CPU optimizer
        
        Args:
            auto_optimize: Whether to automatically apply optimizations
        """
        self.intel_available = False
        self.mkl_available = False
        self.optimizations_applied = []
        self.performance_baseline = None
        
        if auto_optimize:
            self.apply_optimizations()
    
    def apply_optimizations(self) -> Dict[str, Any]:
        """
        Apply all available CPU optimizations
        
        Returns:
            Dictionary of applied optimizations and their status
        """
        logger.info("Applying CPU optimizations")
        
        results = {}
        
        # Intel scikit-learn extension
        intel_result = self._enable_intel_sklearn()
        results["intel_sklearn"] = intel_result
        
        # MKL optimizations
        mkl_result = self._enable_mkl_optimizations()
        results["mkl_optimizations"] = mkl_result
        
        # Thread optimizations
        thread_result = self._optimize_threading()
        results["threading"] = thread_result
        
        # NumPy optimizations
        numpy_result = self._optimize_numpy()
        results["numpy"] = numpy_result
        
        # Environment optimizations
        env_result = self._optimize_environment()
        results["environment"] = env_result
        
        logger.info(f"Applied {len([r for r in results.values() if r['enabled']])} optimizations")
        
        return results
    
    def _enable_intel_sklearn(self) -> Dict[str, Any]:
        """Enable Intel extension for scikit-learn"""
        
        try:
            # Try to patch scikit-learn with Intel optimizations
            from sklearnex import patch_sklearn
            patch_sklearn()
            
            # Verify the patch worked
            import sklearn
            intel_info = getattr(sklearn, '__intel_optimizations__', None)
            
            if intel_info:
                self.intel_available = True
                self.optimizations_applied.append("intel_sklearn")
                
                logger.info("Intel scikit-learn extensions enabled")
                return {
                    "enabled": True,
                    "version": intel_info.get("version", "unknown"),
                    "features": intel_info.get("features", []),
                    "performance_gain": "2-10x speedup expected"
                }
            else:
                # Manual import verification
                try:
                    import sklearnex
                    self.intel_available = True
                    self.optimizations_applied.append("intel_sklearn")
                    
                    logger.info("Intel scikit-learn extensions enabled (manual verification)")
                    return {
                        "enabled": True,
                        "version": sklearnex.__version__,
                        "features": ["daal4py_optimizations"],
                        "performance_gain": "2-10x speedup expected"
                    }
                except Exception:
                    pass
        
        except ImportError:
            logger.info("Intel scikit-learn extension not available - using standard scikit-learn")
        except Exception as e:
            logger.warning(f"Failed to enable Intel scikit-learn: {e}")
        
        return {"enabled": False, "reason": "not_available"}
    
    def _enable_mkl_optimizations(self) -> Dict[str, Any]:
        """Enable Intel MKL optimizations"""
        
        try:
            import numpy as np
            
            # Check if MKL is available
            blas_info = np.__config__.get_info('blas_info')
            lapack_info = np.__config__.get_info('lapack_info')
            
            mkl_detected = (
                'mkl' in str(blas_info).lower() or
                'mkl' in str(lapack_info).lower()
            )
            
            if mkl_detected:
                self.mkl_available = True
                self.optimizations_applied.append("mkl")
                
                # Set MKL threading
                try:
                    import mkl
                    mkl.set_num_threads(os.cpu_count())
                    
                    logger.info(f"MKL optimizations enabled with {os.cpu_count()} threads")
                    return {
                        "enabled": True,
                        "threads": os.cpu_count(),
                        "version": getattr(mkl, '__version__', 'unknown'),
                        "performance_gain": "2-5x speedup for linear algebra"
                    }
                    
                except ImportError:
                    # MKL available but no direct control
                    logger.info("MKL detected but no direct control interface")
                    return {
                        "enabled": True,
                        "threads": "auto",
                        "version": "system",
                        "performance_gain": "2-5x speedup for linear algebra"
                    }
            
        except Exception as e:
            logger.warning(f"Failed to detect/enable MKL: {e}")
        
        return {"enabled": False, "reason": "not_available"}
    
    def _optimize_threading(self) -> Dict[str, Any]:
        """Optimize threading for CPU performance"""
        
        try:
            cpu_count = os.cpu_count()
            
            # Set environment variables for optimal threading
            threading_vars = {
                'OMP_NUM_THREADS': str(cpu_count),
                'MKL_NUM_THREADS': str(cpu_count),
                'NUMEXPR_NUM_THREADS': str(cpu_count),
                'OPENBLAS_NUM_THREADS': str(cpu_count),
                'VECLIB_MAXIMUM_THREADS': str(cpu_count)
            }
            
            applied_vars = {}
            for var, value in threading_vars.items():
                if var not in os.environ:
                    os.environ[var] = value
                    applied_vars[var] = value
            
            self.optimizations_applied.append("threading")
            
            logger.info(f"Threading optimized for {cpu_count} cores")
            return {
                "enabled": True,
                "cpu_count": cpu_count,
                "applied_vars": applied_vars,
                "performance_gain": "Better CPU utilization"
            }
            
        except Exception as e:
            logger.warning(f"Threading optimization failed: {e}")
            return {"enabled": False, "error": str(e)}
    
    def _optimize_numpy(self) -> Dict[str, Any]:
        """Optimize NumPy for performance"""
        
        try:
            import numpy as np
            
            # Get NumPy configuration
            config = np.__config__.show()
            
            # Set NumPy error handling for performance
            old_settings = np.seterr(all='ignore')  # Suppress warnings for speed
            
            # Check for optimized BLAS
            blas_info = np.__config__.get_info('blas_info')
            optimized_blas = any(
                lib in str(blas_info).lower() 
                for lib in ['mkl', 'openblas', 'atlas', 'accelerate']
            )
            
            self.optimizations_applied.append("numpy")
            
            logger.info("NumPy optimizations applied")
            return {
                "enabled": True,
                "version": np.__version__,
                "optimized_blas": optimized_blas,
                "blas_library": str(blas_info.get('libraries', 'unknown')),
                "performance_gain": "Optimized linear algebra operations"
            }
            
        except Exception as e:
            logger.warning(f"NumPy optimization failed: {e}")
            return {"enabled": False, "error": str(e)}
    
    def _optimize_environment(self) -> Dict[str, Any]:
        """Set environment variables for optimal performance"""
        
        try:
            # Performance-oriented environment variables
            perf_vars = {
                'PYTHONHASHSEED': '0',  # Reproducible hashing
                'MALLOC_ARENA_MAX': '4',  # Limit memory arenas
                'PYTHONUNBUFFERED': '1',  # Unbuffered output
            }
            
            # Memory optimization
            if sys.platform == 'linux':
                perf_vars.update({
                    'MALLOC_MMAP_THRESHOLD_': '131072',
                    'MALLOC_TRIM_THRESHOLD_': '131072',
                })
            
            applied_vars = {}
            for var, value in perf_vars.items():
                if var not in os.environ:
                    os.environ[var] = value
                    applied_vars[var] = value
            
            self.optimizations_applied.append("environment")
            
            logger.info("Environment variables optimized")
            return {
                "enabled": True,
                "applied_vars": applied_vars,
                "performance_gain": "Memory and runtime optimizations"
            }
            
        except Exception as e:
            logger.warning(f"Environment optimization failed: {e}")
            return {"enabled": False, "error": str(e)}
    
    def get_optimization_status(self) -> Dict[str, Any]:
        """Get current optimization status"""
        
        return {
            "intel_sklearn_available": self.intel_available,
            "mkl_available": self.mkl_available,
            "optimizations_applied": self.optimizations_applied,
            "cpu_count": os.cpu_count(),
            "python_version": sys.version,
            "platform": sys.platform
        }
    
    def benchmark_performance(self, test_size: int = 10000) -> Dict[str, float]:
        """
        Benchmark performance improvements
        
        Args:
            test_size: Size of test data for benchmarking
        
        Returns:
            Performance metrics
        """
        
        try:
            import numpy as np
            import time
            from sklearn.metrics.pairwise import cosine_similarity
            
            # Generate test data
            np.random.seed(42)
            X = np.random.rand(test_size, 100)
            y = np.random.rand(100, 100)
            
            # Benchmark matrix operations
            start_time = time.time()
            
            # Similarity computation
            similarities = cosine_similarity(X, y)
            
            # Matrix multiplication
            result = np.dot(X, y.T)
            
            # Linear algebra operations
            eigenvals = np.linalg.eigvals(y)
            
            total_time = time.time() - start_time
            
            return {
                "total_time_seconds": total_time,
                "operations_per_second": (test_size * 3) / total_time,
                "matrix_size": test_size,
                "optimizations_active": len(self.optimizations_applied)
            }
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            return {"error": str(e)}
    
    def disable_optimizations(self):
        """Disable applied optimizations (limited reversibility)"""
        
        logger.info("Disabling optimizations (limited reversibility)")
        
        # Reset NumPy error handling
        try:
            import numpy as np
            np.seterr(all='warn')
        except:
            pass
        
        # Note: Some optimizations like Intel sklearn patching cannot be reversed
        # without restarting the Python process
        
        logger.warning("Some optimizations require process restart to fully disable")


def enable_intel_optimizations() -> bool:
    """
    Convenience function to enable Intel optimizations
    
    Returns:
        True if Intel optimizations were successfully enabled
    """
    
    optimizer = CPUOptimizer(auto_optimize=False)
    intel_result = optimizer._enable_intel_sklearn()
    
    return intel_result.get("enabled", False)


def get_cpu_info() -> Dict[str, Any]:
    """Get detailed CPU information"""
    
    info = {
        "cpu_count": os.cpu_count(),
        "platform": sys.platform,
        "python_version": sys.version
    }
    
    # Try to get more detailed CPU info
    try:
        import platform
        info.update({
            "processor": platform.processor(),
            "machine": platform.machine(),
            "architecture": platform.architecture()
        })
    except:
        pass
    
    # Check for Intel CPU
    try:
        if sys.platform == 'linux':
            with open('/proc/cpuinfo', 'r') as f:
                cpuinfo = f.read()
                if 'Intel' in cpuinfo:
                    info["cpu_vendor"] = "Intel"
                elif 'AMD' in cpuinfo:
                    info["cpu_vendor"] = "AMD"
        elif sys.platform == 'darwin':
            import subprocess
            result = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], 
                                 capture_output=True, text=True)
            if result.returncode == 0:
                info["cpu_brand"] = result.stdout.strip()
    except:
        pass
    
    return info


def test_cpu_optimizer():
    """Test the CPU optimizer"""
    
    print("=== CPU Optimizer Test ===")
    
    # Get CPU info
    cpu_info = get_cpu_info()
    print(f"CPU Info: {cpu_info}")
    
    # Create optimizer
    optimizer = CPUOptimizer(auto_optimize=True)
    
    # Get status
    status = optimizer.get_optimization_status()
    print(f"Optimization Status: {status}")
    
    # Benchmark performance
    benchmark = optimizer.benchmark_performance(test_size=5000)
    print(f"Performance Benchmark: {benchmark}")
    
    return optimizer


if __name__ == "__main__":
    test_cpu_optimizer()