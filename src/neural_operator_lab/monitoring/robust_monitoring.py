"""Robust monitoring and error handling system."""

import logging
import time
import threading
import queue
import json
import os
import traceback
import psutil
import gc
from typing import Dict, Any, Optional, List, Callable, Union
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager
import warnings
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False


@dataclass
class PerformanceMetrics:
    """Performance metrics container."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    gpu_memory_mb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_usage_mb: Optional[float] = None
    network_io_mb: Optional[float] = None


@dataclass 
class ErrorReport:
    """Structured error report."""
    timestamp: float
    error_type: str
    error_message: str
    traceback_str: str
    context: Dict[str, Any]
    severity: str = "ERROR"
    component: str = "unknown"
    

class RobustLogger:
    """Robust logging system with error recovery."""
    
    def __init__(self, name: str, log_dir: str = "logs", max_log_size_mb: int = 100):
        self.name = name
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.max_log_size = max_log_size_mb * 1024 * 1024
        
        # Setup file handlers with rotation
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # File handler with rotation
        log_file = self.log_dir / f"{name}.log"
        self.file_handler = logging.FileHandler(log_file)
        self.file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        self.console_handler = logging.StreamHandler()
        self.console_handler.setLevel(logging.INFO)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.file_handler.setFormatter(formatter)
        self.console_handler.setFormatter(formatter)
        
        self.logger.addHandler(self.file_handler)
        self.logger.addHandler(self.console_handler)
        
        # Error queue for async processing
        self.error_queue = queue.Queue()
        self.error_processor = threading.Thread(target=self._process_errors, daemon=True)
        self.error_processor.start()
        
    def _process_errors(self):
        """Process errors asynchronously."""
        while True:
            try:
                error_report = self.error_queue.get(timeout=1)
                if error_report is None:  # Shutdown signal
                    break
                
                # Log error with context
                self.logger.error(
                    f"[{error_report.component}] {error_report.error_type}: "
                    f"{error_report.error_message}\n"
                    f"Context: {json.dumps(error_report.context, indent=2)}\n"
                    f"Traceback: {error_report.traceback_str}"
                )
                
                # Rotate log if too large
                self._rotate_log_if_needed()
                
            except queue.Empty:
                continue
            except Exception as e:
                # Fallback logging to console
                print(f"Error in error processor: {e}")
    
    def _rotate_log_if_needed(self):
        """Rotate log file if it exceeds size limit."""
        try:
            log_file = self.log_dir / f"{self.name}.log"
            if log_file.exists() and log_file.stat().st_size > self.max_log_size:
                # Rotate log
                backup_file = self.log_dir / f"{self.name}_backup.log"
                if backup_file.exists():
                    backup_file.unlink()
                log_file.rename(backup_file)
                
                # Create new handler
                self.logger.removeHandler(self.file_handler)
                self.file_handler = logging.FileHandler(log_file)
                formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                self.file_handler.setFormatter(formatter)
                self.logger.addHandler(self.file_handler)
        
        except Exception as e:
            print(f"Log rotation error: {e}")
    
    def log_error(self, error: Exception, context: Dict[str, Any] = None, component: str = "unknown"):
        """Log error with full context."""
        error_report = ErrorReport(
            timestamp=time.time(),
            error_type=type(error).__name__,
            error_message=str(error),
            traceback_str=traceback.format_exc(),
            context=context or {},
            component=component
        )
        
        try:
            self.error_queue.put(error_report, timeout=1)
        except queue.Full:
            # Fallback to direct logging
            self.logger.error(f"Error queue full, direct log: {error}")
    
    def shutdown(self):
        """Shutdown logger gracefully."""
        self.error_queue.put(None)  # Shutdown signal
        self.error_processor.join(timeout=5)


class SystemMonitor:
    """System resource monitoring."""
    
    def __init__(self, monitor_interval: float = 1.0):
        self.monitor_interval = monitor_interval
        self.metrics_history: List[PerformanceMetrics] = []
        self.max_history = 1000
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start system monitoring."""
        if self.monitoring:
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
    
    def stop_monitoring(self):
        """Stop system monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5)
    
    def _monitor_loop(self):
        """Monitoring loop."""
        while self.monitoring:
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                
                # Trim history
                if len(self.metrics_history) > self.max_history:
                    self.metrics_history.pop(0)
                
                time.sleep(self.monitor_interval)
                
            except Exception as e:
                print(f"Monitoring error: {e}")
                time.sleep(self.monitor_interval)
    
    def _collect_metrics(self) -> PerformanceMetrics:
        """Collect current system metrics."""
        # CPU and Memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        memory_mb = memory.used / (1024 * 1024)
        
        # GPU metrics if available
        gpu_memory_mb = None
        gpu_utilization = None
        
        if HAS_TORCH and torch.cuda.is_available():
            try:
                gpu_memory_mb = torch.cuda.memory_allocated() / (1024 * 1024)
                # GPU utilization requires nvidia-ml-py, skip for now
            except Exception:
                pass
        
        # Disk usage
        disk_usage = psutil.disk_usage('/')
        disk_usage_mb = disk_usage.used / (1024 * 1024)
        
        return PerformanceMetrics(
            timestamp=time.time(),
            cpu_percent=cpu_percent,
            memory_mb=memory_mb,
            gpu_memory_mb=gpu_memory_mb,
            gpu_utilization=gpu_utilization,
            disk_usage_mb=disk_usage_mb
        )
    
    def get_current_metrics(self) -> Optional[PerformanceMetrics]:
        """Get current system metrics."""
        try:
            return self._collect_metrics()
        except Exception:
            return None
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        if not self.metrics_history:
            return {}
        
        cpu_values = [m.cpu_percent for m in self.metrics_history]
        memory_values = [m.memory_mb for m in self.metrics_history]
        
        summary = {
            'cpu_avg': sum(cpu_values) / len(cpu_values),
            'cpu_max': max(cpu_values),
            'memory_avg_mb': sum(memory_values) / len(memory_values),
            'memory_max_mb': max(memory_values),
            'sample_count': len(self.metrics_history)
        }
        
        # GPU metrics if available
        gpu_memory_values = [m.gpu_memory_mb for m in self.metrics_history if m.gpu_memory_mb is not None]
        if gpu_memory_values:
            summary.update({
                'gpu_memory_avg_mb': sum(gpu_memory_values) / len(gpu_memory_values),
                'gpu_memory_max_mb': max(gpu_memory_values)
            })
        
        return summary


class HealthChecker:
    """System health monitoring and alerting."""
    
    def __init__(self, logger: RobustLogger, monitor: SystemMonitor):
        self.logger = logger
        self.monitor = monitor
        self.health_checks: List[Callable] = []
        self.thresholds = {
            'cpu_percent': 90.0,
            'memory_percent': 90.0,
            'gpu_memory_mb': 8000,  # 8GB
            'disk_usage_percent': 90.0
        }
        
    def add_health_check(self, check_func: Callable[[PerformanceMetrics], Optional[str]]):
        """Add custom health check function."""
        self.health_checks.append(check_func)
    
    def check_system_health(self) -> Dict[str, Any]:
        """Comprehensive system health check."""
        health_report = {
            'timestamp': time.time(),
            'status': 'healthy',
            'warnings': [],
            'errors': [],
            'metrics': {}
        }
        
        try:
            # Get current metrics
            metrics = self.monitor.get_current_metrics()
            if metrics:
                health_report['metrics'] = asdict(metrics)
                
                # CPU check
                if metrics.cpu_percent > self.thresholds['cpu_percent']:
                    health_report['warnings'].append(f"High CPU usage: {metrics.cpu_percent:.1f}%")
                    health_report['status'] = 'warning'
                
                # Memory check
                memory = psutil.virtual_memory()
                if memory.percent > self.thresholds['memory_percent']:
                    health_report['warnings'].append(f"High memory usage: {memory.percent:.1f}%")
                    health_report['status'] = 'warning'
                
                # GPU memory check
                if metrics.gpu_memory_mb and metrics.gpu_memory_mb > self.thresholds['gpu_memory_mb']:
                    health_report['warnings'].append(f"High GPU memory: {metrics.gpu_memory_mb:.0f}MB")
                    health_report['status'] = 'warning'
                
                # Disk usage check
                disk = psutil.disk_usage('/')
                disk_percent = disk.used / disk.total * 100
                if disk_percent > self.thresholds['disk_usage_percent']:
                    health_report['warnings'].append(f"High disk usage: {disk_percent:.1f}%")
                    health_report['status'] = 'warning'
                
                # Custom health checks
                for check_func in self.health_checks:
                    try:
                        result = check_func(metrics)
                        if result:
                            health_report['warnings'].append(result)
                            health_report['status'] = 'warning'
                    except Exception as e:
                        health_report['errors'].append(f"Health check error: {e}")
                        health_report['status'] = 'error'
        
        except Exception as e:
            health_report['errors'].append(f"Health check failed: {e}")
            health_report['status'] = 'error'
            self.logger.log_error(e, {'component': 'health_checker'})
        
        return health_report
    
    def alert_if_unhealthy(self):
        """Check health and log alerts if unhealthy."""
        health = self.check_system_health()
        
        if health['status'] != 'healthy':
            self.logger.logger.warning(f"System health: {health['status']}")
            
            for warning in health['warnings']:
                self.logger.logger.warning(f"Health warning: {warning}")
            
            for error in health['errors']:
                self.logger.logger.error(f"Health error: {error}")


class RobustTrainingMonitor:
    """Comprehensive training monitoring with error recovery."""
    
    def __init__(self, name: str = "training", log_dir: str = "logs"):
        self.logger = RobustLogger(name, log_dir)
        self.monitor = SystemMonitor()
        self.health_checker = HealthChecker(self.logger, self.monitor)
        
        # Training metrics
        self.training_metrics = {}
        self.error_counts = {}
        
        # Start monitoring
        self.monitor.start_monitoring()
        
    def log_training_step(self, step: int, metrics: Dict[str, float], context: Dict[str, Any] = None):
        """Log training step with metrics."""
        try:
            # Store metrics
            self.training_metrics[step] = {
                'timestamp': time.time(),
                'metrics': metrics,
                'context': context or {}
            }
            
            # Log to file
            self.logger.logger.info(
                f"Step {step}: " + 
                ", ".join([f"{k}={v:.6f}" for k, v in metrics.items()])
            )
            
            # Check system health periodically
            if step % 100 == 0:
                self.health_checker.alert_if_unhealthy()
                
        except Exception as e:
            self.logger.log_error(e, {'step': step, 'metrics': metrics}, 'training_monitor')
    
    def log_model_error(self, error: Exception, step: int, context: Dict[str, Any] = None):
        """Log model training error with recovery suggestions."""
        error_type = type(error).__name__
        
        # Count errors by type
        self.error_counts[error_type] = self.error_counts.get(error_type, 0) + 1
        
        # Enhanced context
        enhanced_context = {
            'step': step,
            'error_count': self.error_counts[error_type],
            'system_metrics': asdict(self.monitor.get_current_metrics() or PerformanceMetrics(0, 0, 0))
        }
        
        if context:
            enhanced_context.update(context)
        
        # Log with suggestions
        suggestions = self._get_error_suggestions(error_type, enhanced_context)
        enhanced_context['recovery_suggestions'] = suggestions
        
        self.logger.log_error(error, enhanced_context, 'model_training')
        
        # Log suggestions to console for immediate action
        if suggestions:
            print("\n=== RECOVERY SUGGESTIONS ===")
            for suggestion in suggestions:
                print(f"• {suggestion}")
            print("=============================\n")
    
    def _get_error_suggestions(self, error_type: str, context: Dict[str, Any]) -> List[str]:
        """Get recovery suggestions based on error type."""
        suggestions = []
        
        if 'OutOfMemory' in error_type:
            suggestions.extend([
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Use mixed precision training",
                "Clear GPU cache with torch.cuda.empty_cache()"
            ])
        
        elif 'NaN' in error_type or 'Inf' in error_type:
            suggestions.extend([
                "Reduce learning rate",
                "Enable gradient clipping",
                "Check input data for NaN/Inf values",
                "Use more stable numerical methods"
            ])
        
        elif 'Runtime' in error_type:
            if context.get('system_metrics', {}).get('cpu_percent', 0) > 90:
                suggestions.append("High CPU usage detected - consider reducing computation")
            
            if context.get('system_metrics', {}).get('memory_mb', 0) > 8000:
                suggestions.append("High memory usage - consider memory optimization")
        
        # Recurring error suggestions
        if context.get('error_count', 0) > 5:
            suggestions.extend([
                "Consider changing model architecture",
                "Validate input data pipeline",
                "Check for numerical instability",
                "Consider using different optimization settings"
            ])
        
        return suggestions
    
    @contextmanager
    def training_step_context(self, step: int):
        """Context manager for training step monitoring."""
        start_time = time.time()
        start_memory = None
        
        if HAS_TORCH and torch.cuda.is_available():
            torch.cuda.synchronize()
            start_memory = torch.cuda.memory_allocated()
        
        try:
            yield
        except Exception as e:
            self.log_model_error(e, step)
            raise
        finally:
            # Log timing
            elapsed = time.time() - start_time
            
            memory_used = 0
            if start_memory and HAS_TORCH and torch.cuda.is_available():
                torch.cuda.synchronize()
                memory_used = (torch.cuda.memory_allocated() - start_memory) / (1024**2)
            
            self.logger.logger.debug(f"Step {step} completed in {elapsed:.3f}s, memory: +{memory_used:.1f}MB")
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive training summary."""
        summary = {
            'total_steps': len(self.training_metrics),
            'error_counts': self.error_counts,
            'system_metrics_summary': self.monitor.get_metrics_summary(),
            'latest_health': self.health_checker.check_system_health()
        }
        
        # Training metrics summary
        if self.training_metrics:
            latest_step = max(self.training_metrics.keys())
            summary['latest_step'] = latest_step
            summary['latest_metrics'] = self.training_metrics[latest_step]['metrics']
        
        return summary
    
    def shutdown(self):
        """Shutdown monitoring gracefully."""
        self.monitor.stop_monitoring()
        self.logger.shutdown()