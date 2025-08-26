"""Comprehensive model monitoring and health tracking system.

This module provides advanced monitoring capabilities for neural operators including:
- Real-time performance tracking
- Model health monitoring
- Resource usage tracking
- Anomaly detection
- Automated alerting
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Any, Callable, Tuple
import time
import threading
import queue
import logging
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import json
import warnings


class AlertSeverity(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class ModelMetrics:
    """Model performance and health metrics."""
    timestamp: float
    forward_time: Optional[float] = None
    backward_time: Optional[float] = None
    memory_used: Optional[float] = None
    memory_cached: Optional[float] = None
    cpu_usage: Optional[float] = None
    gpu_utilization: Optional[float] = None
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    loss_value: Optional[float] = None
    gradient_norm: Optional[float] = None
    parameter_norm: Optional[float] = None
    numerical_issues: List[str] = None
    
    def __post_init__(self):
        if self.numerical_issues is None:
            self.numerical_issues = []


@dataclass
class Alert:
    """Model monitoring alert."""
    timestamp: float
    severity: AlertSeverity
    message: str
    category: str
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class PerformanceTracker:
    """Track model performance metrics over time."""
    
    def __init__(self, window_size: int = 1000, alert_threshold: float = 2.0):
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
        # Metrics storage
        self.forward_times = deque(maxlen=window_size)
        self.memory_usage = deque(maxlen=window_size)
        self.gradient_norms = deque(maxlen=window_size)
        
        # Statistics
        self.total_forward_passes = 0
        self.total_training_time = 0.0
        self.start_time = time.time()
        
        # Alert callbacks
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
    def record_forward_pass(self, forward_time: float, memory_mb: float = None):
        """Record forward pass metrics."""
        self.forward_times.append(forward_time)
        if memory_mb is not None:
            self.memory_usage.append(memory_mb)
        self.total_forward_passes += 1
        
        # Check for performance anomalies
        self._check_performance_anomalies(forward_time)
    
    def record_gradient_norm(self, grad_norm: float):
        """Record gradient norm for training monitoring."""
        self.gradient_norms.append(grad_norm)
        
        # Check for gradient issues
        if grad_norm > 100.0:
            self._emit_alert(
                AlertSeverity.WARNING,
                f"Large gradient norm detected: {grad_norm:.2e}",
                "gradient_norm"
            )
        elif grad_norm < 1e-8:
            self._emit_alert(
                AlertSeverity.WARNING,
                f"Very small gradient norm detected: {grad_norm:.2e}",
                "gradient_norm"
            )
    
    def _check_performance_anomalies(self, current_time: float):
        """Check for performance anomalies."""
        if len(self.forward_times) < 10:
            return
        
        # Calculate statistics
        recent_times = list(self.forward_times)[-10:]
        avg_recent = sum(recent_times) / len(recent_times)
        
        if len(self.forward_times) > 50:
            overall_times = list(self.forward_times)[:-10]
            avg_overall = sum(overall_times) / len(overall_times)
            
            # Check if current performance is significantly worse
            if avg_recent > self.alert_threshold * avg_overall:
                self._emit_alert(
                    AlertSeverity.WARNING,
                    f"Performance degradation detected: {avg_recent:.3f}s vs {avg_overall:.3f}s baseline",
                    "performance_degradation"
                )
    
    def _emit_alert(self, severity: AlertSeverity, message: str, category: str, metadata: Dict = None):
        """Emit alert to registered callbacks."""
        alert = Alert(
            timestamp=time.time(),
            severity=severity,
            message=message,
            category=category,
            metadata=metadata or {}
        )
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                logging.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary statistics."""
        summary = {
            'total_forward_passes': self.total_forward_passes,
            'uptime_seconds': time.time() - self.start_time,
            'average_forward_time': sum(self.forward_times) / len(self.forward_times) if self.forward_times else 0,
            'recent_forward_time': sum(list(self.forward_times)[-10:]) / 10 if len(self.forward_times) >= 10 else 0,
            'memory_usage_mb': {
                'current': self.memory_usage[-1] if self.memory_usage else 0,
                'average': sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                'peak': max(self.memory_usage) if self.memory_usage else 0
            },
            'gradient_norms': {
                'current': self.gradient_norms[-1] if self.gradient_norms else 0,
                'average': sum(self.gradient_norms) / len(self.gradient_norms) if self.gradient_norms else 0,
                'max': max(self.gradient_norms) if self.gradient_norms else 0,
                'min': min(self.gradient_norms) if self.gradient_norms else 0
            }
        }
        
        return summary


class ModelHealthMonitor:
    """Monitor model health and detect anomalies."""
    
    def __init__(self, check_interval: float = 10.0):
        self.check_interval = check_interval
        self.logger = logging.getLogger(__name__)
        
        # Health status tracking
        self.last_health_check = None
        self.health_history = deque(maxlen=100)
        self.anomalies_detected = defaultdict(int)
        
        # Background monitoring
        self.monitoring_active = False
        self.monitoring_thread = None
        self.stop_event = threading.Event()
        
        # Alert system
        self.alert_callbacks: List[Callable[[Alert], None]] = []
        
    def start_monitoring(self, model: nn.Module):
        """Start background health monitoring."""
        if self.monitoring_active:
            return
        
        self.model = model
        self.monitoring_active = True
        self.stop_event.clear()
        
        self.monitoring_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self.monitoring_thread.start()
        
        self.logger.info("Model health monitoring started")
    
    def stop_monitoring(self):
        """Stop background health monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitoring_thread:
            self.monitoring_thread.join(timeout=5.0)
        
        self.logger.info("Model health monitoring stopped")
    
    def _monitoring_loop(self):
        """Background monitoring loop."""
        while not self.stop_event.wait(self.check_interval):
            try:
                health_status = self.check_model_health(self.model)
                self.health_history.append(health_status)
                
                # Analyze health trends
                self._analyze_health_trends()
                
            except Exception as e:
                self.logger.error(f"Health monitoring error: {e}")
    
    def check_model_health(self, model: nn.Module) -> Dict[str, Any]:
        """Comprehensive model health check."""
        health_status = {
            'timestamp': time.time(),
            'overall_status': 'healthy',
            'issues': [],
            'parameter_health': {},
            'memory_health': {},
            'numerical_health': {}
        }
        
        try:
            # Check parameter health
            param_issues = self._check_parameter_health(model)
            health_status['parameter_health'] = param_issues
            if param_issues['issues']:
                health_status['issues'].extend(param_issues['issues'])
            
            # Check memory health
            memory_issues = self._check_memory_health()
            health_status['memory_health'] = memory_issues
            if memory_issues['issues']:
                health_status['issues'].extend(memory_issues['issues'])
            
            # Check numerical health
            numerical_issues = self._check_numerical_health(model)
            health_status['numerical_health'] = numerical_issues
            if numerical_issues['issues']:
                health_status['issues'].extend(numerical_issues['issues'])
            
            # Determine overall status
            if any('critical' in issue.lower() for issue in health_status['issues']):
                health_status['overall_status'] = 'critical'
            elif health_status['issues']:
                health_status['overall_status'] = 'warning'
            
            # Emit alerts for critical issues
            if health_status['overall_status'] == 'critical':
                self._emit_alert(
                    AlertSeverity.CRITICAL,
                    f"Critical model health issues: {health_status['issues']}",
                    "model_health"
                )
            
        except Exception as e:
            health_status['overall_status'] = 'error'
            health_status['issues'].append(f"Health check failed: {str(e)}")
            
        self.last_health_check = health_status
        return health_status
    
    def _check_parameter_health(self, model: nn.Module) -> Dict[str, Any]:
        """Check parameter health."""
        param_health = {
            'total_parameters': 0,
            'nan_parameters': 0,
            'inf_parameters': 0,
            'zero_parameters': 0,
            'large_parameters': 0,
            'gradient_issues': 0,
            'issues': []
        }
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                param_health['total_parameters'] += param.numel()
                
                # Check for NaN parameters
                nan_count = torch.isnan(param).sum().item()
                if nan_count > 0:
                    param_health['nan_parameters'] += nan_count
                    param_health['issues'].append(f"NaN parameters in {name}: {nan_count}")
                
                # Check for infinite parameters
                inf_count = torch.isinf(param).sum().item()
                if inf_count > 0:
                    param_health['inf_parameters'] += inf_count
                    param_health['issues'].append(f"Inf parameters in {name}: {inf_count}")
                
                # Check for zero parameters (potential dead neurons)
                zero_count = (param.abs() < 1e-10).sum().item()
                zero_ratio = zero_count / param.numel()
                if zero_ratio > 0.5:
                    param_health['zero_parameters'] += zero_count
                    param_health['issues'].append(f"Many zero parameters in {name}: {zero_ratio:.1%}")
                
                # Check for very large parameters
                large_count = (param.abs() > 100).sum().item()
                if large_count > 0:
                    param_health['large_parameters'] += large_count
                    param_health['issues'].append(f"Large parameters in {name}: {large_count}")
                
                # Check gradients if available
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        param_health['gradient_issues'] += 1
                        param_health['issues'].append(f"NaN gradients in {name}")
                    
                    grad_norm = torch.norm(param.grad).item()
                    if grad_norm > 1000:
                        param_health['gradient_issues'] += 1
                        param_health['issues'].append(f"Large gradients in {name}: {grad_norm:.2e}")
        
        return param_health
    
    def _check_memory_health(self) -> Dict[str, Any]:
        """Check memory health."""
        memory_health = {
            'issues': []
        }
        
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / (1024**3)  # GB
            reserved = torch.cuda.memory_reserved() / (1024**3)   # GB
            
            memory_health.update({
                'cuda_allocated_gb': allocated,
                'cuda_reserved_gb': reserved,
                'cuda_memory_fraction': allocated / (torch.cuda.get_device_properties(0).total_memory / (1024**3))
            })
            
            # Check for memory issues
            if allocated > 0.9 * (torch.cuda.get_device_properties(0).total_memory / (1024**3)):
                memory_health['issues'].append(f"High CUDA memory usage: {allocated:.1f}GB")
            
            if reserved - allocated > 2.0:  # More than 2GB cached
                memory_health['issues'].append(f"High CUDA memory cache: {reserved-allocated:.1f}GB")
        
        return memory_health
    
    def _check_numerical_health(self, model: nn.Module) -> Dict[str, Any]:
        """Check numerical health and stability."""
        numerical_health = {
            'issues': []
        }
        
        # Check model outputs for typical inputs
        try:
            model.eval()
            with torch.no_grad():
                # Create dummy input
                if hasattr(model, 'input_dim'):
                    dummy_input = torch.randn(2, 32, 32, model.input_dim)
                else:
                    dummy_input = torch.randn(2, 32, 32, 3)  # Default
                
                try:
                    output = model(dummy_input)
                    
                    # Check output for issues
                    if torch.isnan(output).any():
                        numerical_health['issues'].append("Model outputs contain NaN")
                    
                    if torch.isinf(output).any():
                        numerical_health['issues'].append("Model outputs contain Inf")
                    
                    # Check output magnitude
                    output_magnitude = torch.abs(output).max().item()
                    if output_magnitude > 1e6:
                        numerical_health['issues'].append(f"Very large model outputs: {output_magnitude:.2e}")
                    
                except Exception as e:
                    numerical_health['issues'].append(f"Model forward pass failed: {str(e)}")
            
        except Exception as e:
            numerical_health['issues'].append(f"Numerical health check failed: {str(e)}")
        
        return numerical_health
    
    def _analyze_health_trends(self):
        """Analyze health trends over time."""
        if len(self.health_history) < 5:
            return
        
        recent_statuses = [h['overall_status'] for h in list(self.health_history)[-5:]]
        
        # Check for degrading health trend
        if recent_statuses.count('warning') >= 3:
            self._emit_alert(
                AlertSeverity.WARNING,
                "Persistent model health warnings detected",
                "health_trend"
            )
        
        if recent_statuses.count('critical') >= 2:
            self._emit_alert(
                AlertSeverity.CRITICAL,
                "Multiple critical health issues detected",
                "health_trend"
            )
    
    def _emit_alert(self, severity: AlertSeverity, message: str, category: str, metadata: Dict = None):
        """Emit health monitoring alert."""
        alert = Alert(
            timestamp=time.time(),
            severity=severity,
            message=message,
            category=category,
            metadata=metadata or {}
        )
        
        for callback in self.alert_callbacks:
            try:
                callback(alert)
            except Exception as e:
                self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Alert], None]):
        """Add alert callback function."""
        self.alert_callbacks.append(callback)
    
    def get_health_summary(self) -> Dict[str, Any]:
        """Get health monitoring summary."""
        return {
            'monitoring_active': self.monitoring_active,
            'last_health_check': self.last_health_check,
            'health_history_length': len(self.health_history),
            'anomalies_detected': dict(self.anomalies_detected),
            'alert_callbacks_registered': len(self.alert_callbacks)
        }


class ComprehensiveModelMonitor:
    """Comprehensive monitoring system combining performance and health monitoring."""
    
    def __init__(
        self,
        performance_window_size: int = 1000,
        health_check_interval: float = 10.0,
        enable_file_logging: bool = True,
        log_file_path: str = "model_monitoring.log"
    ):
        self.performance_tracker = PerformanceTracker(window_size=performance_window_size)
        self.health_monitor = ModelHealthMonitor(check_interval=health_check_interval)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        if enable_file_logging:
            handler = logging.FileHandler(log_file_path)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Alert system
        self.alerts_history = deque(maxlen=1000)
        
        # Register alert callbacks
        self.performance_tracker.add_alert_callback(self._handle_alert)
        self.health_monitor.add_alert_callback(self._handle_alert)
        
        self.monitoring_start_time = None
    
    def start_monitoring(self, model: nn.Module):
        """Start comprehensive monitoring."""
        self.monitoring_start_time = time.time()
        self.health_monitor.start_monitoring(model)
        self.logger.info("Comprehensive model monitoring started")
    
    def stop_monitoring(self):
        """Stop all monitoring."""
        self.health_monitor.stop_monitoring()
        self.logger.info("Comprehensive model monitoring stopped")
    
    def record_forward_pass(self, forward_time: float, input_shape: Tuple, output_shape: Tuple, 
                          memory_mb: float = None, loss_value: float = None):
        """Record forward pass metrics."""
        self.performance_tracker.record_forward_pass(forward_time, memory_mb)
        
        # Log detailed metrics
        self.logger.debug(f"Forward pass: {forward_time:.3f}s, input_shape={input_shape}, "
                         f"output_shape={output_shape}, memory={memory_mb}MB, loss={loss_value}")
    
    def record_training_step(self, loss: float, grad_norm: float):
        """Record training step metrics."""
        self.performance_tracker.record_gradient_norm(grad_norm)
        
        # Check for training anomalies
        if loss != loss:  # NaN check
            self._handle_alert(Alert(
                timestamp=time.time(),
                severity=AlertSeverity.CRITICAL,
                message="NaN loss detected during training",
                category="training_anomaly"
            ))
    
    def _handle_alert(self, alert: Alert):
        """Handle monitoring alerts."""
        self.alerts_history.append(alert)
        
        # Log alert
        log_level = {
            AlertSeverity.INFO: logging.INFO,
            AlertSeverity.WARNING: logging.WARNING,
            AlertSeverity.ERROR: logging.ERROR,
            AlertSeverity.CRITICAL: logging.CRITICAL
        }.get(alert.severity, logging.WARNING)
        
        self.logger.log(log_level, f"[{alert.category}] {alert.message}")
        
        # Handle critical alerts
        if alert.severity == AlertSeverity.CRITICAL:
            self.logger.critical(f"CRITICAL ALERT: {alert.message}")
            # Could implement emergency actions here (e.g., model checkpointing)
    
    def get_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive monitoring report."""
        return {
            'monitoring_info': {
                'start_time': self.monitoring_start_time,
                'uptime_seconds': time.time() - self.monitoring_start_time if self.monitoring_start_time else 0,
                'monitoring_active': self.health_monitor.monitoring_active
            },
            'performance_summary': self.performance_tracker.get_performance_summary(),
            'health_summary': self.health_monitor.get_health_summary(),
            'alerts_summary': {
                'total_alerts': len(self.alerts_history),
                'recent_alerts': list(self.alerts_history)[-10:] if self.alerts_history else [],
                'alerts_by_severity': {
                    severity.value: sum(1 for alert in self.alerts_history if alert.severity == severity)
                    for severity in AlertSeverity
                },
                'alerts_by_category': {
                    category: sum(1 for alert in self.alerts_history if alert.category == category)
                    for category in set(alert.category for alert in self.alerts_history)
                }
            }
        }
    
    def save_report(self, filepath: str):
        """Save monitoring report to file."""
        report = self.get_comprehensive_report()
        
        # Convert non-serializable objects
        def serialize_alert(alert):
            return {
                'timestamp': alert.timestamp,
                'severity': alert.severity.value,
                'message': alert.message,
                'category': alert.category,
                'metadata': alert.metadata
            }
        
        # Serialize alerts
        if 'alerts_summary' in report and 'recent_alerts' in report['alerts_summary']:
            report['alerts_summary']['recent_alerts'] = [
                serialize_alert(alert) for alert in report['alerts_summary']['recent_alerts']
            ]
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Monitoring report saved to {filepath}")


def create_qisa_monitor(
    model: nn.Module,
    monitoring_level: str = "standard"  # "basic", "standard", "comprehensive"
) -> ComprehensiveModelMonitor:
    """Create monitoring system optimized for QISA models.
    
    Args:
        model: QISA model to monitor
        monitoring_level: Level of monitoring detail
        
    Returns:
        Configured ComprehensiveModelMonitor
    """
    if monitoring_level == "basic":
        monitor = ComprehensiveModelMonitor(
            performance_window_size=100,
            health_check_interval=30.0,
            enable_file_logging=False
        )
    elif monitoring_level == "standard":
        monitor = ComprehensiveModelMonitor(
            performance_window_size=1000,
            health_check_interval=10.0,
            enable_file_logging=True
        )
    elif monitoring_level == "comprehensive":
        monitor = ComprehensiveModelMonitor(
            performance_window_size=5000,
            health_check_interval=5.0,
            enable_file_logging=True,
            log_file_path="qisa_comprehensive_monitoring.log"
        )
    else:
        raise ValueError(f"Unknown monitoring level: {monitoring_level}")
    
    monitor.start_monitoring(model)
    return monitor