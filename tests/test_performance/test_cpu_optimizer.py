"""
Tests for CPU Optimizer
"""

import pytest
import os
import numpy as np
from unittest.mock import Mock, patch

from kb_brain.performance.cpu_optimizer import (
    CPUOptimizer, enable_intel_optimizations, get_cpu_info
)


class TestCPUOptimizer:
    """Test cases for CPU Optimizer"""
    
    @pytest.fixture
    def optimizer(self):
        """Create CPU optimizer instance"""
        return CPUOptimizer(auto_optimize=False)
    
    def test_optimizer_initialization(self, optimizer):
        """Test optimizer initialization"""
        assert optimizer.intel_available in [True, False]
        assert optimizer.mkl_available in [True, False]
        assert isinstance(optimizer.optimizations_applied, list)
        assert optimizer.performance_baseline is None
    
    def test_apply_optimizations(self, optimizer):
        """Test applying CPU optimizations"""
        results = optimizer.apply_optimizations()
        
        assert isinstance(results, dict)
        assert "intel_sklearn" in results
        assert "mkl_optimizations" in results
        assert "threading" in results
        assert "numpy" in results
        assert "environment" in results
        
        # Each result should have enabled status
        for opt_name, result in results.items():
            assert "enabled" in result
            assert isinstance(result["enabled"], bool)
    
    def test_intel_sklearn_optimization(self, optimizer):
        """Test Intel scikit-learn optimization"""
        result = optimizer._enable_intel_sklearn()
        
        assert isinstance(result, dict)
        assert "enabled" in result
        
        if result["enabled"]:
            assert "version" in result
            assert "performance_gain" in result
        else:
            assert "reason" in result
    
    def test_mkl_optimization(self, optimizer):
        """Test MKL optimization"""
        result = optimizer._enable_mkl_optimizations()
        
        assert isinstance(result, dict)
        assert "enabled" in result
        
        if result["enabled"]:
            assert "performance_gain" in result
    
    def test_threading_optimization(self, optimizer):
        """Test threading optimization"""
        result = optimizer._optimize_threading()
        
        assert isinstance(result, dict)
        assert "enabled" in result
        
        if result["enabled"]:
            assert "cpu_count" in result
            assert result["cpu_count"] == os.cpu_count()
    
    def test_numpy_optimization(self, optimizer):
        """Test NumPy optimization"""
        result = optimizer._optimize_numpy()
        
        assert isinstance(result, dict)
        assert "enabled" in result
        
        if result["enabled"]:
            assert "version" in result
            assert "optimized_blas" in result
    
    def test_environment_optimization(self, optimizer):
        """Test environment variable optimization"""
        result = optimizer._optimize_environment()
        
        assert isinstance(result, dict)
        assert "enabled" in result
        
        if result["enabled"]:
            assert "applied_vars" in result
    
    def test_optimization_status(self, optimizer):
        """Test getting optimization status"""
        # Apply some optimizations first
        optimizer.apply_optimizations()
        
        status = optimizer.get_optimization_status()
        
        assert isinstance(status, dict)
        assert "intel_sklearn_available" in status
        assert "mkl_available" in status
        assert "optimizations_applied" in status
        assert "cpu_count" in status
        assert "python_version" in status
        assert "platform" in status
    
    def test_performance_benchmark(self, optimizer):
        """Test performance benchmarking"""
        benchmark = optimizer.benchmark_performance(test_size=1000)
        
        if "error" not in benchmark:
            assert "total_time_seconds" in benchmark
            assert "operations_per_second" in benchmark
            assert "matrix_size" in benchmark
            assert benchmark["matrix_size"] == 1000
            assert benchmark["total_time_seconds"] > 0
            assert benchmark["operations_per_second"] > 0
    
    def test_enable_intel_optimizations_function(self):
        """Test standalone Intel optimization function"""
        result = enable_intel_optimizations()
        assert isinstance(result, bool)
    
    def test_get_cpu_info_function(self):
        """Test CPU information function"""
        info = get_cpu_info()
        
        assert isinstance(info, dict)
        assert "cpu_count" in info
        assert "platform" in info
        assert "python_version" in info
        assert info["cpu_count"] == os.cpu_count()
    
    def test_auto_optimize_flag(self):
        """Test auto-optimize flag in constructor"""
        # Test with auto_optimize=True
        auto_optimizer = CPUOptimizer(auto_optimize=True)
        assert len(auto_optimizer.optimizations_applied) >= 0
        
        # Test with auto_optimize=False
        manual_optimizer = CPUOptimizer(auto_optimize=False)
        assert len(manual_optimizer.optimizations_applied) == 0
    
    def test_optimization_tracking(self, optimizer):
        """Test optimization tracking"""
        initial_count = len(optimizer.optimizations_applied)
        
        # Apply optimizations
        optimizer.apply_optimizations()
        
        # Should have tracked applied optimizations
        final_count = len(optimizer.optimizations_applied)
        assert final_count >= initial_count
    
    def test_disable_optimizations(self, optimizer):
        """Test disabling optimizations"""
        # Apply optimizations first
        optimizer.apply_optimizations()
        
        # Disable (note: limited reversibility)
        optimizer.disable_optimizations()
        
        # This mainly tests that the method doesn't crash
        # Full reversal requires process restart for some optimizations
    
    @patch('os.cpu_count')
    def test_threading_with_different_cpu_counts(self, mock_cpu_count, optimizer):
        """Test threading optimization with different CPU counts"""
        # Test with different CPU counts
        for cpu_count in [1, 4, 8, 16]:
            mock_cpu_count.return_value = cpu_count
            
            result = optimizer._optimize_threading()
            
            if result["enabled"]:
                assert result["cpu_count"] == cpu_count
    
    def test_benchmark_error_handling(self, optimizer):
        """Test benchmark error handling"""
        # Test with invalid test size
        benchmark = optimizer.benchmark_performance(test_size=0)
        
        # Should either work or return error
        assert isinstance(benchmark, dict)
        if "error" in benchmark:
            assert isinstance(benchmark["error"], str)
    
    def test_mkl_detection_robustness(self, optimizer):
        """Test MKL detection robustness"""
        # This tests that MKL detection doesn't crash
        result = optimizer._enable_mkl_optimizations()
        
        assert isinstance(result, dict)
        assert "enabled" in result
        
        # Should not crash regardless of MKL availability
    
    def test_environment_variable_setting(self, optimizer):
        """Test environment variable setting"""
        original_env = os.environ.copy()
        
        try:
            # Remove some variables if they exist
            test_vars = ['OMP_NUM_THREADS', 'MKL_NUM_THREADS']
            for var in test_vars:
                if var in os.environ:
                    del os.environ[var]
            
            result = optimizer._optimize_environment()
            
            if result["enabled"]:
                # Should have set some variables
                assert "applied_vars" in result
                
        finally:
            # Restore original environment
            os.environ.clear()
            os.environ.update(original_env)


if __name__ == "__main__":
    pytest.main([__file__])