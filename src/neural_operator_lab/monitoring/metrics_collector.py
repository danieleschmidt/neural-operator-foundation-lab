"""Metrics collection and system monitoring for neural operators."""

import time
import psutil
import threading
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import logging

logger = logging.getLogger(__name__)


@dataclass
class SystemMetrics:
    """System resource metrics."""
    
    timestamp: float = field(default_factory=time.time)
    cpu_percent: float = 0.0
    memory_percent: float = 0.0
    memory_used_gb: float = 0.0
    memory_available_gb: float = 0.0
    gpu_memory_used_gb: Optional[float] = None
    gpu_memory_total_gb: Optional[float] = None
    gpu_utilization: Optional[float] = None
    disk_usage_percent: float = 0.0
    network_bytes_sent: int = 0
    network_bytes_recv: int = 0


@dataclass 
class TrainingMetrics:
    """Training-specific metrics."""
    
    timestamp: float = field(default_factory=time.time)
    epoch: int = 0
    batch: int = 0
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    learning_rate: float = 0.0
    batch_size: int = 0
    throughput_samples_per_sec: float = 0.0
    gradient_norm: Optional[float] = None
    memory_peak_gb: Optional[float] = None


class MetricsCollector:
    """Comprehensive metrics collection for neural operator training."""
    
    def __init__(
        self,
        collection_interval: float = 1.0,
        max_history: int = 1000,
        enable_gpu_monitoring: bool = True
    ):
        self.collection_interval = collection_interval
        self.max_history = max_history
        self.enable_gpu_monitoring = enable_gpu_monitoring
        
        # Metrics storage
        self.system_metrics: deque = deque(maxlen=max_history)
        self.training_metrics: deque = deque(maxlen=max_history)
        self.custom_metrics: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=max_history)
        )
        
        # Collection state
        self._collecting = False
        self._collection_thread: Optional[threading.Thread] = None
        self._last_network_stats = None
        
        # GPU monitoring setup
        self._gpu_available = False
        if enable_gpu_monitoring:
            try:
                import torch
                self._gpu_available = torch.cuda.is_available()
                if self._gpu_available:
                    logger.info("GPU monitoring enabled")
            except ImportError:
                logger.warning("PyTorch not available, GPU monitoring disabled")
    
    def start_collection(self) -> None:
        """Start automatic metrics collection."""
        if self._collecting:
            logger.warning("Metrics collection already running")
            return
            
        self._collecting = True
        self._collection_thread = threading.Thread(
            target=self._collection_loop,
            daemon=True
        )
        self._collection_thread.start()
        logger.info("Started metrics collection")
    
    def stop_collection(self) -> None:
        """Stop automatic metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Stopped metrics collection")
    
    def _collection_loop(self) -> None:
        """Main metrics collection loop."""
        while self._collecting:
            try:
                metrics = self._collect_system_metrics()
                self.system_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def _collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = None
        gpu_memory_total = None
        gpu_utilization = None
        
        if self._gpu_available:
            try:
                import torch
                if torch.cuda.device_count() > 0:
                    gpu_memory_used = torch.cuda.memory_allocated(0) / 1e9
                    gpu_memory_total = torch.cuda.memory_reserved(0) / 1e9
                    # GPU utilization would require nvidia-ml-py
            except Exception as e:
                logger.debug(f"GPU metrics collection failed: {e}")
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network stats
        network = psutil.net_io_counters()
        
        return SystemMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / 1e9,
            memory_available_gb=memory.available / 1e9,
            gpu_memory_used_gb=gpu_memory_used,
            gpu_memory_total_gb=gpu_memory_total,
            gpu_utilization=gpu_utilization,
            disk_usage_percent=disk.percent,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv
        )
    
    def record_training_metrics(
        self,
        epoch: int,
        batch: int,
        train_loss: float,
        val_loss: Optional[float] = None,
        learning_rate: float = 0.0,
        batch_size: int = 0,
        throughput_samples_per_sec: float = 0.0,
        gradient_norm: Optional[float] = None
    ) -> None:
        """Record training-specific metrics."""
        
        # Calculate memory peak if GPU available
        memory_peak_gb = None
        if self._gpu_available:
            try:
                import torch
                memory_peak_gb = torch.cuda.max_memory_allocated(0) / 1e9
            except Exception:
                pass
        
        metrics = TrainingMetrics(
            epoch=epoch,
            batch=batch,
            train_loss=train_loss,
            val_loss=val_loss,
            learning_rate=learning_rate,
            batch_size=batch_size,
            throughput_samples_per_sec=throughput_samples_per_sec,
            gradient_norm=gradient_norm,
            memory_peak_gb=memory_peak_gb
        )
        
        self.training_metrics.append(metrics)
    
    def record_custom_metric(
        self,
        name: str,
        value: Union[int, float],
        timestamp: Optional[float] = None
    ) -> None:
        """Record a custom metric."""
        timestamp = timestamp or time.time()
        self.custom_metrics[name].append((timestamp, value))
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def get_latest_training_metrics(self) -> Optional[TrainingMetrics]:
        """Get the most recent training metrics."""
        return self.training_metrics[-1] if self.training_metrics else None
    
    def get_system_metrics_history(
        self,
        last_n: Optional[int] = None
    ) -> List[SystemMetrics]:
        """Get system metrics history."""
        metrics_list = list(self.system_metrics)
        if last_n:
            return metrics_list[-last_n:]
        return metrics_list
    
    def get_training_metrics_history(
        self,
        last_n: Optional[int] = None
    ) -> List[TrainingMetrics]:
        """Get training metrics history."""
        metrics_list = list(self.training_metrics)
        if last_n:
            return metrics_list[-last_n:]
        return metrics_list
    
    def get_custom_metrics(self, name: str) -> List[tuple]:
        """Get custom metrics by name."""
        return list(self.custom_metrics[name])
    
    def get_average_metrics(
        self,
        window_size: int = 10
    ) -> Dict[str, float]:
        """Calculate average metrics over a window."""
        if not self.system_metrics:
            return {}
        
        recent_metrics = list(self.system_metrics)[-window_size:]
        
        return {
            'avg_cpu_percent': sum(m.cpu_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_percent': sum(m.memory_percent for m in recent_metrics) / len(recent_metrics),
            'avg_memory_used_gb': sum(m.memory_used_gb for m in recent_metrics) / len(recent_metrics),
        }
    
    def export_metrics(self, format: str = 'dict') -> Dict[str, Any]:
        """Export all metrics in specified format."""
        data = {
            'system_metrics': [
                {
                    'timestamp': m.timestamp,
                    'cpu_percent': m.cpu_percent,
                    'memory_percent': m.memory_percent,
                    'memory_used_gb': m.memory_used_gb,
                    'gpu_memory_used_gb': m.gpu_memory_used_gb,
                    'gpu_utilization': m.gpu_utilization
                }
                for m in self.system_metrics
            ],
            'training_metrics': [
                {
                    'timestamp': m.timestamp,
                    'epoch': m.epoch,
                    'batch': m.batch,
                    'train_loss': m.train_loss,
                    'val_loss': m.val_loss,
                    'learning_rate': m.learning_rate,
                    'throughput_samples_per_sec': m.throughput_samples_per_sec
                }
                for m in self.training_metrics
            ],
            'custom_metrics': {
                name: list(values)
                for name, values in self.custom_metrics.items()
            }
        }
        
        return data
    
    def clear_metrics(self) -> None:
        """Clear all collected metrics."""
        self.system_metrics.clear()
        self.training_metrics.clear()
        self.custom_metrics.clear()
        logger.info("Cleared all metrics")
    
    def __enter__(self):
        """Context manager entry."""
        self.start_collection()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_collection()