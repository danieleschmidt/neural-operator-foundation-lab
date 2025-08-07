"""Health monitoring and system diagnostics for neural operator training."""

import torch
import psutil
import time
import logging
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass
from enum import Enum
import threading
from pathlib import Path
import json


class HealthStatus(Enum):
    """Health status levels."""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    FAILURE = "failure"


@dataclass
class HealthMetric:
    """Individual health metric."""
    name: str
    value: float
    threshold_warning: float
    threshold_critical: float
    unit: str
    description: str
    timestamp: float
    
    @property
    def status(self) -> HealthStatus:
        """Determine status based on thresholds."""
        if self.value >= self.threshold_critical:
            return HealthStatus.CRITICAL
        elif self.value >= self.threshold_warning:
            return HealthStatus.WARNING
        else:
            return HealthStatus.HEALTHY


@dataclass
class SystemHealth:
    """Overall system health assessment."""
    status: HealthStatus
    metrics: Dict[str, HealthMetric]
    alerts: List[str]
    recommendations: List[str]
    timestamp: float


class HealthChecker:
    """Comprehensive health monitoring for neural operator training."""
    
    def __init__(self, 
                 check_interval: float = 30.0,
                 log_file: Optional[str] = None,
                 enable_gpu_monitoring: bool = True):
        self.check_interval = check_interval
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Health metrics storage
        self.health_history: List[SystemHealth] = []
        self.alert_callbacks: List[Callable] = []
        
        # Monitoring state
        self.monitoring = False
        self.monitor_thread: Optional[threading.Thread] = None
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if log_file:
            handler = logging.FileHandler(log_file)
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
        
        # Define metric configurations
        self._setup_metric_configs()
    
    def _setup_metric_configs(self):
        """Setup metric configurations with thresholds."""
        self.metric_configs = {
            # CPU metrics
            'cpu_usage_percent': {
                'threshold_warning': 80.0,
                'threshold_critical': 95.0,
                'unit': '%',
                'description': 'CPU utilization percentage'
            },
            'cpu_temperature': {
                'threshold_warning': 70.0,
                'threshold_critical': 85.0,
                'unit': '°C',
                'description': 'CPU temperature'
            },
            
            # Memory metrics
            'memory_usage_percent': {
                'threshold_warning': 85.0,
                'threshold_critical': 95.0,
                'unit': '%',
                'description': 'System memory usage percentage'
            },
            'memory_available_gb': {
                'threshold_warning': 2.0,
                'threshold_critical': 1.0,
                'unit': 'GB',
                'description': 'Available system memory',
                'invert': True  # Lower values are worse
            },
            
            # GPU metrics (if available)
            'gpu_usage_percent': {
                'threshold_warning': 90.0,
                'threshold_critical': 98.0,
                'unit': '%',
                'description': 'GPU utilization percentage'
            },
            'gpu_memory_percent': {
                'threshold_warning': 90.0,
                'threshold_critical': 98.0,
                'unit': '%',
                'description': 'GPU memory usage percentage'
            },
            'gpu_temperature': {
                'threshold_warning': 80.0,
                'threshold_critical': 90.0,
                'unit': '°C',
                'description': 'GPU temperature'
            },
            
            # Training metrics
            'loss_nan_count': {
                'threshold_warning': 1.0,
                'threshold_critical': 5.0,
                'unit': 'count',
                'description': 'Number of NaN losses detected'
            },
            'gradient_norm': {
                'threshold_warning': 10.0,
                'threshold_critical': 100.0,
                'unit': '',
                'description': 'Gradient norm magnitude'
            },
            
            # Disk metrics
            'disk_usage_percent': {
                'threshold_warning': 85.0,
                'threshold_critical': 95.0,
                'unit': '%',
                'description': 'Disk usage percentage'
            },
            'disk_available_gb': {
                'threshold_warning': 10.0,
                'threshold_critical': 5.0,
                'unit': 'GB',
                'description': 'Available disk space',
                'invert': True
            }
        }
    
    def collect_system_metrics(self) -> Dict[str, float]:
        """Collect current system metrics."""
        metrics = {}
        current_time = time.time()
        
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            metrics['cpu_usage_percent'] = cpu_percent
            
            # CPU temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    cpu_temps = []
                    for name, entries in temps.items():
                        if 'cpu' in name.lower() or 'core' in name.lower():
                            cpu_temps.extend([entry.current for entry in entries])
                    
                    if cpu_temps:
                        metrics['cpu_temperature'] = max(cpu_temps)
            except (AttributeError, OSError):
                pass
            
            # Memory metrics
            memory = psutil.virtual_memory()
            metrics['memory_usage_percent'] = memory.percent
            metrics['memory_available_gb'] = memory.available / (1024**3)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            metrics['disk_usage_percent'] = (disk.used / disk.total) * 100
            metrics['disk_available_gb'] = disk.free / (1024**3)
            
            # GPU metrics (if available and enabled)
            if self.enable_gpu_monitoring and torch.cuda.is_available():
                gpu_metrics = self._collect_gpu_metrics()
                metrics.update(gpu_metrics)
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
        
        return metrics
    
    def _collect_gpu_metrics(self) -> Dict[str, float]:
        """Collect GPU-specific metrics."""
        metrics = {}
        
        try:
            # GPU utilization
            if hasattr(torch.cuda, 'utilization'):
                metrics['gpu_usage_percent'] = torch.cuda.utilization()
            
            # GPU memory
            if torch.cuda.is_available():
                memory_allocated = torch.cuda.memory_allocated()
                memory_reserved = torch.cuda.memory_reserved()
                memory_total = torch.cuda.get_device_properties(0).total_memory
                
                metrics['gpu_memory_percent'] = (memory_reserved / memory_total) * 100
                metrics['gpu_memory_allocated_gb'] = memory_allocated / (1024**3)
                metrics['gpu_memory_reserved_gb'] = memory_reserved / (1024**3)
            
            # GPU temperature (if available)
            try:
                import pynvml
                pynvml.nvmlInit()
                
                device_count = pynvml.nvmlDeviceGetCount()
                if device_count > 0:
                    handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
                    metrics['gpu_temperature'] = temp
            except ImportError:
                self.logger.debug("pynvml not available for GPU temperature monitoring")
            except Exception as e:
                self.logger.debug(f"Could not get GPU temperature: {e}")
        
        except Exception as e:
            self.logger.error(f"Error collecting GPU metrics: {e}")
        
        return metrics
    
    def collect_training_metrics(self, trainer=None, **kwargs) -> Dict[str, float]:
        """Collect training-specific metrics."""
        metrics = {}
        
        try:
            # Loss-related metrics
            if 'loss' in kwargs:
                loss_value = kwargs['loss']
                if torch.is_tensor(loss_value):
                    loss_value = loss_value.item()
                
                if not (torch.isnan(torch.tensor(loss_value)) or torch.isinf(torch.tensor(loss_value))):
                    metrics['current_loss'] = loss_value
                else:
                    metrics['loss_nan_count'] = metrics.get('loss_nan_count', 0) + 1
            
            # Gradient metrics
            if trainer and hasattr(trainer, 'model'):
                total_norm = 0
                param_count = 0
                
                for p in trainer.model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                        param_count += 1
                
                if param_count > 0:
                    metrics['gradient_norm'] = (total_norm ** 0.5)
            
            # Learning rate
            if trainer and hasattr(trainer, 'optimizer'):
                metrics['learning_rate'] = trainer.optimizer.param_groups[0]['lr']
            
        except Exception as e:
            self.logger.error(f"Error collecting training metrics: {e}")
        
        return metrics
    
    def assess_health(self, metrics: Dict[str, float], 
                     training_metrics: Dict[str, float] = None) -> SystemHealth:
        """Assess overall system health."""
        current_time = time.time()
        all_metrics = {**metrics}
        
        if training_metrics:
            all_metrics.update(training_metrics)
        
        # Create health metrics
        health_metrics = {}
        overall_status = HealthStatus.HEALTHY
        alerts = []
        recommendations = []
        
        for metric_name, value in all_metrics.items():
            if metric_name in self.metric_configs:
                config = self.metric_configs[metric_name]
                
                # Handle inverted metrics (where lower is worse)
                if config.get('invert', False):
                    warning_threshold = config['threshold_warning']
                    critical_threshold = config['threshold_critical']
                    
                    if value <= critical_threshold:
                        status = HealthStatus.CRITICAL
                    elif value <= warning_threshold:
                        status = HealthStatus.WARNING
                    else:
                        status = HealthStatus.HEALTHY
                else:
                    # Normal metrics (where higher is worse)
                    health_metric = HealthMetric(
                        name=metric_name,
                        value=value,
                        threshold_warning=config['threshold_warning'],
                        threshold_critical=config['threshold_critical'],
                        unit=config['unit'],
                        description=config['description'],
                        timestamp=current_time
                    )
                    status = health_metric.status
                
                health_metrics[metric_name] = health_metric
                
                # Update overall status (take worst)
                if status.value == HealthStatus.CRITICAL.value:
                    overall_status = HealthStatus.CRITICAL
                elif status.value == HealthStatus.WARNING.value and overall_status != HealthStatus.CRITICAL:
                    overall_status = HealthStatus.WARNING
                
                # Generate alerts and recommendations
                if status == HealthStatus.CRITICAL:
                    alerts.append(f"CRITICAL: {metric_name} = {value:.2f} {config['unit']}")
                    recommendations.extend(self._get_recommendations(metric_name, status))
                elif status == HealthStatus.WARNING:
                    alerts.append(f"WARNING: {metric_name} = {value:.2f} {config['unit']}")
                    recommendations.extend(self._get_recommendations(metric_name, status))
        
        return SystemHealth(
            status=overall_status,
            metrics=health_metrics,
            alerts=alerts,
            recommendations=list(set(recommendations)),  # Remove duplicates
            timestamp=current_time
        )
    
    def _get_recommendations(self, metric_name: str, status: HealthStatus) -> List[str]:
        """Get recommendations for specific metric issues."""
        recommendations = []
        
        if metric_name == 'cpu_usage_percent':
            if status == HealthStatus.CRITICAL:
                recommendations.extend([
                    "Reduce batch size to lower CPU usage",
                    "Enable multiprocessing for data loading",
                    "Consider using GPU acceleration if available"
                ])
            elif status == HealthStatus.WARNING:
                recommendations.append("Monitor CPU usage and consider optimizations")
        
        elif metric_name == 'memory_usage_percent' or metric_name == 'memory_available_gb':
            if status == HealthStatus.CRITICAL:
                recommendations.extend([
                    "Immediately reduce batch size",
                    "Enable gradient checkpointing",
                    "Clear unused variables and call torch.cuda.empty_cache()"
                ])
            elif status == HealthStatus.WARNING:
                recommendations.extend([
                    "Consider reducing batch size",
                    "Monitor memory usage trends"
                ])
        
        elif metric_name.startswith('gpu_'):
            if 'memory' in metric_name:
                if status == HealthStatus.CRITICAL:
                    recommendations.extend([
                        "Reduce batch size immediately",
                        "Enable mixed precision training",
                        "Use gradient accumulation instead of large batches"
                    ])
                elif status == HealthStatus.WARNING:
                    recommendations.extend([
                        "Consider reducing batch size",
                        "Enable mixed precision if not already enabled"
                    ])
            
            elif 'temperature' in metric_name:
                if status == HealthStatus.CRITICAL:
                    recommendations.extend([
                        "Reduce GPU load immediately",
                        "Check GPU cooling system",
                        "Reduce batch size or training intensity"
                    ])
        
        elif metric_name == 'gradient_norm':
            if status == HealthStatus.CRITICAL:
                recommendations.extend([
                    "Enable gradient clipping",
                    "Reduce learning rate",
                    "Check for numerical instabilities"
                ])
            elif status == HealthStatus.WARNING:
                recommendations.append("Monitor gradient norm trends")
        
        elif metric_name == 'loss_nan_count':
            if status in [HealthStatus.WARNING, HealthStatus.CRITICAL]:
                recommendations.extend([
                    "Reduce learning rate",
                    "Enable gradient clipping",
                    "Check input data for NaN values",
                    "Add numerical stability to loss function"
                ])
        
        elif metric_name.startswith('disk_'):
            if status == HealthStatus.CRITICAL:
                recommendations.extend([
                    "Free up disk space immediately",
                    "Clean temporary files and logs",
                    "Reduce checkpoint frequency"
                ])
            elif status == HealthStatus.WARNING:
                recommendations.append("Monitor disk usage and plan cleanup")
        
        return recommendations
    
    def start_monitoring(self, trainer=None):
        """Start continuous health monitoring."""
        if self.monitoring:
            self.logger.warning("Health monitoring is already running")
            return
        
        self.monitoring = True
        self.monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(trainer,),
            daemon=True
        )
        self.monitor_thread.start()
        
        self.logger.info(f"Started health monitoring with {self.check_interval}s interval")
    
    def stop_monitoring(self):
        """Stop continuous health monitoring."""
        if not self.monitoring:
            return
        
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        
        self.logger.info("Stopped health monitoring")
    
    def _monitoring_loop(self, trainer):
        """Continuous monitoring loop."""
        while self.monitoring:
            try:
                # Collect metrics
                system_metrics = self.collect_system_metrics()
                training_metrics = {}
                
                if trainer:
                    training_metrics = self.collect_training_metrics(trainer)
                
                # Assess health
                health = self.assess_health(system_metrics, training_metrics)
                
                # Store history
                self.health_history.append(health)
                
                # Limit history size
                if len(self.health_history) > 1000:
                    self.health_history = self.health_history[-1000:]
                
                # Log critical issues
                if health.status == HealthStatus.CRITICAL:
                    self.logger.error(f"CRITICAL HEALTH ISSUE: {', '.join(health.alerts)}")
                elif health.status == HealthStatus.WARNING:
                    self.logger.warning(f"Health warning: {', '.join(health.alerts)}")
                
                # Execute alert callbacks
                for callback in self.alert_callbacks:
                    try:
                        callback(health)
                    except Exception as e:
                        self.logger.error(f"Alert callback failed: {e}")
                
                # Sleep until next check
                time.sleep(self.check_interval)
            
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                time.sleep(self.check_interval)
    
    def add_alert_callback(self, callback: Callable):
        """Add callback function for health alerts."""
        self.alert_callbacks.append(callback)
    
    def get_current_health(self, trainer=None) -> SystemHealth:
        """Get current health status."""
        system_metrics = self.collect_system_metrics()
        training_metrics = self.collect_training_metrics(trainer) if trainer else {}
        
        return self.assess_health(system_metrics, training_metrics)
    
    def get_health_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get health summary for the specified time period."""
        if not self.health_history:
            return {"message": "No health data available"}
        
        cutoff_time = time.time() - (hours * 3600)
        recent_health = [h for h in self.health_history if h.timestamp >= cutoff_time]
        
        if not recent_health:
            return {"message": f"No health data available for last {hours} hours"}
        
        # Count status occurrences
        status_counts = {}
        total_alerts = 0
        unique_recommendations = set()
        
        for health in recent_health:
            status = health.status.value
            status_counts[status] = status_counts.get(status, 0) + 1
            total_alerts += len(health.alerts)
            unique_recommendations.update(health.recommendations)
        
        return {
            "time_period_hours": hours,
            "total_checks": len(recent_health),
            "status_distribution": status_counts,
            "total_alerts": total_alerts,
            "alerts_per_check": total_alerts / len(recent_health),
            "unique_recommendations": list(unique_recommendations),
            "current_status": self.health_history[-1].status.value,
            "recent_alerts": self.health_history[-1].alerts
        }
    
    def export_health_report(self, filepath: str, hours: int = 24):
        """Export detailed health report."""
        summary = self.get_health_summary(hours)
        current_health = self.get_current_health()
        
        report = {
            "export_timestamp": time.time(),
            "summary": summary,
            "current_health": {
                "status": current_health.status.value,
                "alerts": current_health.alerts,
                "recommendations": current_health.recommendations,
                "metrics": {
                    name: {
                        "value": metric.value,
                        "status": metric.status.value,
                        "unit": metric.unit,
                        "description": metric.description
                    }
                    for name, metric in current_health.metrics.items()
                }
            },
            "recent_history": [
                {
                    "timestamp": h.timestamp,
                    "status": h.status.value,
                    "alerts_count": len(h.alerts),
                    "recommendations_count": len(h.recommendations)
                }
                for h in self.health_history[-100:]  # Last 100 checks
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"Health report exported to {filepath}")


def create_training_health_monitor(trainer, check_interval: float = 30.0) -> HealthChecker:
    """Create and start a health monitor for training."""
    log_file = None
    if hasattr(trainer, 'config') and hasattr(trainer.config, 'save_dir'):
        log_file = str(Path(trainer.config.save_dir) / 'health.log')
    
    health_checker = HealthChecker(
        check_interval=check_interval,
        log_file=log_file
    )
    
    # Add alert callback to trainer if it has error handler
    if hasattr(trainer, 'error_handler'):
        def health_alert_callback(health: SystemHealth):
            if health.status == HealthStatus.CRITICAL:
                context = {
                    'trainer': trainer,
                    'health_alerts': health.alerts,
                    'recommendations': health.recommendations
                }
                
                # Log critical health issues as errors
                trainer.logger.error(f"Critical health issues detected: {health.alerts}")
                
                # Could potentially trigger recovery actions
                for rec in health.recommendations:
                    if 'reduce batch size' in rec.lower():
                        trainer.error_handler._reduce_batch_size(context)
                    elif 'reduce learning rate' in rec.lower():
                        trainer.error_handler._reduce_learning_rate(context)
        
        health_checker.add_alert_callback(health_alert_callback)
    
    health_checker.start_monitoring(trainer)
    return health_checker