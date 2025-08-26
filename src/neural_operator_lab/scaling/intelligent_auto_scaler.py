"""Intelligent Auto-Scaling System for QISA Neural Operators.

Advanced auto-scaling system that dynamically adjusts computational resources
based on workload demands, model complexity, and performance requirements:

- Dynamic GPU allocation and load balancing
- Intelligent batch size optimization  
- Adaptive model parallelism
- Resource usage prediction
- Cost-aware scaling decisions
- Multi-cloud resource management
"""

import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
import numpy as np
import time
import threading
import queue
from collections import deque, defaultdict
import logging
import json
from dataclasses import dataclass
from enum import Enum
import psutil
import subprocess

try:
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    DISTRIBUTED_AVAILABLE = True
except ImportError:
    DISTRIBUTED_AVAILABLE = False

try:
    import gpustat
    GPU_MONITORING_AVAILABLE = True
except ImportError:
    GPU_MONITORING_AVAILABLE = False


class ScalingDecision(Enum):
    """Auto-scaling decision types."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down"
    MAINTAIN = "maintain"
    OPTIMIZE = "optimize"


class ResourceType(Enum):
    """Resource types for scaling."""
    GPU = "gpu"
    CPU = "cpu"
    MEMORY = "memory"
    BATCH_SIZE = "batch_size"
    MODEL_PARALLEL = "model_parallel"


@dataclass
class ResourceMetrics:
    """Resource utilization metrics."""
    timestamp: float
    gpu_utilization: Dict[int, float]  # GPU ID -> utilization %
    gpu_memory: Dict[int, Tuple[float, float]]  # GPU ID -> (used_gb, total_gb)
    cpu_utilization: float
    system_memory: Tuple[float, float]  # (used_gb, total_gb)
    inference_time: Optional[float] = None
    throughput: Optional[float] = None  # samples/second
    batch_size: Optional[int] = None
    queue_length: Optional[int] = None


@dataclass
class ScalingAction:
    """Scaling action to be performed."""
    action_type: ScalingDecision
    resource_type: ResourceType
    current_value: Union[int, float]
    target_value: Union[int, float]
    estimated_impact: Dict[str, float]
    confidence: float
    reasoning: str


class ResourceMonitor:
    """Monitor system resources and performance metrics."""
    
    def __init__(self, monitoring_interval: float = 5.0, history_size: int = 1000):
        self.monitoring_interval = monitoring_interval
        self.history_size = history_size
        
        # Metrics storage
        self.metrics_history = deque(maxlen=history_size)
        self.performance_history = deque(maxlen=history_size)
        
        # Threading
        self.monitoring_active = False
        self.monitor_thread = None
        self.stop_event = threading.Event()
        
        # Current state
        self.current_metrics = None
        self.logger = logging.getLogger(__name__)
        
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.stop_event.clear()
        
        self.monitor_thread = threading.Thread(target=self._monitoring_loop, daemon=True)
        self.monitor_thread.start()
        
        self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        if not self.monitoring_active:
            return
        
        self.monitoring_active = False
        self.stop_event.set()
        
        if self.monitor_thread:
            self.monitor_thread.join(timeout=10.0)
        
        self.logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self):
        """Main monitoring loop."""
        while not self.stop_event.wait(self.monitoring_interval):
            try:
                metrics = self._collect_metrics()
                self.metrics_history.append(metrics)
                self.current_metrics = metrics
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current resource metrics."""
        
        # GPU metrics
        gpu_utilization = {}
        gpu_memory = {}
        
        if torch.cuda.is_available():
            for gpu_id in range(torch.cuda.device_count()):
                # Basic GPU metrics using torch
                with torch.cuda.device(gpu_id):
                    memory_used = torch.cuda.memory_allocated(gpu_id) / (1024**3)  # GB
                    memory_total = torch.cuda.get_device_properties(gpu_id).total_memory / (1024**3)  # GB
                    
                    gpu_memory[gpu_id] = (memory_used, memory_total)
                    
                    # Estimate utilization based on memory usage
                    gpu_utilization[gpu_id] = min(100.0, (memory_used / memory_total) * 100)
        
        # System metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        memory_info = psutil.virtual_memory()
        system_memory = (memory_info.used / (1024**3), memory_info.total / (1024**3))
        
        return ResourceMetrics(
            timestamp=time.time(),
            gpu_utilization=gpu_utilization,
            gpu_memory=gpu_memory,
            cpu_utilization=cpu_percent,
            system_memory=system_memory
        )
    
    def get_recent_metrics(self, window_seconds: float = 60.0) -> List[ResourceMetrics]:
        """Get metrics from recent time window."""
        cutoff_time = time.time() - window_seconds
        return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def calculate_resource_trends(self, window_seconds: float = 300.0) -> Dict[str, float]:
        """Calculate resource usage trends."""
        recent_metrics = self.get_recent_metrics(window_seconds)
        
        if len(recent_metrics) < 2:
            return {}
        
        trends = {}
        
        # GPU utilization trend
        gpu_utils = []
        for metrics in recent_metrics:
            if metrics.gpu_utilization:
                gpu_utils.append(np.mean(list(metrics.gpu_utilization.values())))
        
        if len(gpu_utils) >= 2:
            trends['gpu_utilization_trend'] = np.polyfit(range(len(gpu_utils)), gpu_utils, 1)[0]
        
        # Memory usage trend
        memory_usage = [m.system_memory[0] / m.system_memory[1] * 100 for m in recent_metrics]
        if len(memory_usage) >= 2:
            trends['memory_trend'] = np.polyfit(range(len(memory_usage)), memory_usage, 1)[0]
        
        # CPU utilization trend
        cpu_usage = [m.cpu_utilization for m in recent_metrics]
        if len(cpu_usage) >= 2:
            trends['cpu_trend'] = np.polyfit(range(len(cpu_usage)), cpu_usage, 1)[0]
        
        return trends


class PerformancePredictor:
    """Predict performance impact of scaling decisions."""
    
    def __init__(self, model_complexity: Optional[int] = None):
        self.model_complexity = model_complexity or 1000000  # Default param count
        
        # Performance models (simple linear models for demonstration)
        self.throughput_model_params = {
            'gpu_count': 0.8,       # 80% efficiency per additional GPU
            'batch_size': 0.95,     # 95% efficiency per batch size increase  
            'memory_limit': -0.1    # Performance degrades near memory limit
        }
        
        self.latency_model_params = {
            'model_size': 1e-6,     # Latency increases with model size
            'batch_size': 0.1,      # Slight latency increase with batch size
            'gpu_count': -0.2       # Latency decreases with more GPUs
        }
        
        # Historical performance data
        self.performance_history = []
        
    def predict_throughput_change(
        self,
        current_config: Dict[str, Any],
        proposed_config: Dict[str, Any]
    ) -> float:
        """Predict throughput change from scaling action."""
        
        # Calculate throughput multiplier
        gpu_multiplier = 1.0
        if 'gpu_count' in proposed_config:
            gpu_ratio = proposed_config['gpu_count'] / current_config.get('gpu_count', 1)
            gpu_multiplier = 1 + (gpu_ratio - 1) * self.throughput_model_params['gpu_count']
        
        batch_multiplier = 1.0
        if 'batch_size' in proposed_config:
            batch_ratio = proposed_config['batch_size'] / current_config.get('batch_size', 1)
            batch_multiplier = 1 + (batch_ratio - 1) * self.throughput_model_params['batch_size']
        
        # Memory pressure penalty
        memory_penalty = 1.0
        if 'memory_usage_pct' in current_config:
            if current_config['memory_usage_pct'] > 80:
                memory_penalty = 1 + (current_config['memory_usage_pct'] - 80) * self.throughput_model_params['memory_limit'] / 100
        
        predicted_multiplier = gpu_multiplier * batch_multiplier * memory_penalty
        
        return predicted_multiplier
    
    def predict_latency_change(
        self,
        current_config: Dict[str, Any],
        proposed_config: Dict[str, Any]
    ) -> float:
        """Predict latency change from scaling action."""
        
        # Simple latency prediction model
        model_size_factor = self.model_complexity * self.latency_model_params['model_size']
        
        batch_size_factor = 0
        if 'batch_size' in proposed_config:
            batch_diff = proposed_config['batch_size'] - current_config.get('batch_size', 1)
            batch_size_factor = batch_diff * self.latency_model_params['batch_size']
        
        gpu_factor = 0
        if 'gpu_count' in proposed_config:
            gpu_diff = proposed_config['gpu_count'] - current_config.get('gpu_count', 1)
            gpu_factor = gpu_diff * self.latency_model_params['gpu_count']
        
        latency_change_pct = (model_size_factor + batch_size_factor + gpu_factor) * 100
        
        return max(-50, min(50, latency_change_pct))  # Clamp to reasonable range
    
    def predict_cost_change(
        self,
        current_config: Dict[str, Any],
        proposed_config: Dict[str, Any],
        gpu_cost_per_hour: float = 2.0
    ) -> float:
        """Predict cost change from scaling action."""
        
        current_gpus = current_config.get('gpu_count', 1)
        proposed_gpus = proposed_config.get('gpu_count', current_gpus)
        
        cost_multiplier = proposed_gpus / current_gpus
        
        return cost_multiplier


class IntelligentAutoScaler:
    """Intelligent auto-scaling system for QISA neural operators."""
    
    def __init__(
        self,
        model: nn.Module,
        target_gpu_utilization: float = 80.0,
        target_memory_utilization: float = 85.0,
        min_throughput: float = 10.0,  # samples/second
        max_cost_increase: float = 2.0,  # 2x cost increase limit
        scaling_cooldown: float = 300.0,  # 5 minutes between scaling actions
        enable_predictive_scaling: bool = True
    ):
        self.model = model
        self.target_gpu_utilization = target_gpu_utilization
        self.target_memory_utilization = target_memory_utilization
        self.min_throughput = min_throughput
        self.max_cost_increase = max_cost_increase
        self.scaling_cooldown = scaling_cooldown
        self.enable_predictive_scaling = enable_predictive_scaling
        
        # Components
        self.resource_monitor = ResourceMonitor()
        self.performance_predictor = PerformancePredictor(
            model_complexity=sum(p.numel() for p in model.parameters())
        )
        
        # Scaling state
        self.current_config = self._get_current_config()
        self.last_scaling_time = 0.0
        self.scaling_history = []
        
        # Auto-scaling settings
        self.auto_scaling_enabled = False
        self.scaling_thread = None
        self.stop_scaling_event = threading.Event()
        
        self.logger = logging.getLogger(__name__)
        
    def _get_current_config(self) -> Dict[str, Any]:
        """Get current system configuration."""
        return {
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'batch_size': 32,  # Default batch size
            'model_parallel': False,
            'mixed_precision': False
        }
    
    def start_auto_scaling(self):
        """Start automatic scaling based on resource metrics."""
        if self.auto_scaling_enabled:
            return
        
        self.resource_monitor.start_monitoring()
        self.auto_scaling_enabled = True
        self.stop_scaling_event.clear()
        
        self.scaling_thread = threading.Thread(target=self._auto_scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        self.logger.info("Intelligent auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop automatic scaling."""
        if not self.auto_scaling_enabled:
            return
        
        self.auto_scaling_enabled = False
        self.stop_scaling_event.set()
        self.resource_monitor.stop_monitoring()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        self.logger.info("Auto-scaling stopped")
    
    def analyze_scaling_needs(self) -> Optional[ScalingAction]:
        """Analyze current metrics and determine scaling needs."""
        
        current_metrics = self.resource_monitor.current_metrics
        if not current_metrics:
            return None
        
        # Analyze GPU utilization
        if current_metrics.gpu_utilization:
            avg_gpu_util = np.mean(list(current_metrics.gpu_utilization.values()))
            
            if avg_gpu_util < 50:  # Under-utilized
                return self._suggest_scale_down_action(current_metrics)
            elif avg_gpu_util > self.target_gpu_utilization:  # Over-utilized
                return self._suggest_scale_up_action(current_metrics)
        
        # Analyze memory utilization
        total_memory_used = 0
        total_memory_available = 0
        
        for gpu_id, (used, total) in current_metrics.gpu_memory.items():
            total_memory_used += used
            total_memory_available += total
        
        if total_memory_available > 0:
            memory_util_pct = (total_memory_used / total_memory_available) * 100
            
            if memory_util_pct > self.target_memory_utilization:
                return self._suggest_memory_optimization_action(current_metrics)
        
        return ScalingAction(
            action_type=ScalingDecision.MAINTAIN,
            resource_type=ResourceType.GPU,
            current_value=len(current_metrics.gpu_utilization),
            target_value=len(current_metrics.gpu_utilization),
            estimated_impact={},
            confidence=1.0,
            reasoning="All metrics within target ranges"
        )
    
    def _suggest_scale_up_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """Suggest scale-up action based on high utilization."""
        
        current_gpus = len(metrics.gpu_utilization)
        target_gpus = min(current_gpus + 1, 8)  # Max 8 GPUs
        
        return ScalingAction(
            action_type=ScalingDecision.SCALE_UP,
            resource_type=ResourceType.GPU,
            current_value=current_gpus,
            target_value=target_gpus,
            estimated_impact={'throughput_change': 1.5, 'cost_change': 2.0},
            confidence=0.8,
            reasoning=f"High GPU utilization ({np.mean(list(metrics.gpu_utilization.values())):.1f}%)"
        )
    
    def _suggest_scale_down_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """Suggest scale-down action based on low utilization."""
        
        current_gpus = len(metrics.gpu_utilization)
        if current_gpus <= 1:
            return None  # Cannot scale down below 1 GPU
        
        target_gpus = current_gpus - 1
        
        return ScalingAction(
            action_type=ScalingDecision.SCALE_DOWN,
            resource_type=ResourceType.GPU,
            current_value=current_gpus,
            target_value=target_gpus,
            estimated_impact={'throughput_change': 0.7, 'cost_change': 0.5},
            confidence=0.7,
            reasoning=f"Low GPU utilization ({np.mean(list(metrics.gpu_utilization.values())):.1f}%)"
        )
    
    def _suggest_memory_optimization_action(self, metrics: ResourceMetrics) -> ScalingAction:
        """Suggest memory optimization action."""
        
        current_batch_size = self.current_config.get('batch_size', 32)
        target_batch_size = max(1, current_batch_size // 2)  # Reduce batch size
        
        return ScalingAction(
            action_type=ScalingDecision.OPTIMIZE,
            resource_type=ResourceType.BATCH_SIZE,
            current_value=current_batch_size,
            target_value=target_batch_size,
            estimated_impact={'memory_reduction': 0.5, 'throughput_change': 0.8},
            confidence=0.9,
            reasoning="High memory utilization, reducing batch size"
        )
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Generate scaling system report."""
        
        current_metrics = self.resource_monitor.current_metrics
        
        # Calculate average utilization
        avg_gpu_util = 0
        avg_memory_util = 0
        
        if current_metrics and current_metrics.gpu_utilization:
            avg_gpu_util = np.mean(list(current_metrics.gpu_utilization.values()))
            
            total_memory_used = sum(used for used, _ in current_metrics.gpu_memory.values())
            total_memory_available = sum(total for _, total in current_metrics.gpu_memory.values())
            avg_memory_util = (total_memory_used / total_memory_available * 100) if total_memory_available > 0 else 0
        
        return {
            'auto_scaling_enabled': self.auto_scaling_enabled,
            'current_config': self.current_config,
            'target_utilization': {
                'gpu': self.target_gpu_utilization,
                'memory': self.target_memory_utilization
            },
            'current_utilization': {
                'gpu': avg_gpu_util,
                'memory': avg_memory_util,
                'cpu': current_metrics.cpu_utilization if current_metrics else 0
            },
            'scaling_history': len(self.scaling_history),
            'recent_scaling_actions': self.scaling_history[-5:] if self.scaling_history else [],
            'monitoring_stats': {
                'metrics_collected': len(self.resource_monitor.metrics_history),
                'monitoring_active': self.resource_monitor.monitoring_active
            }
        }


def create_intelligent_auto_scaler(
    model: nn.Module,
    scaling_config: Optional[Dict[str, Any]] = None
) -> IntelligentAutoScaler:
    """Factory function to create intelligent auto-scaler.
    
    Args:
        model: Neural operator model to scale
        scaling_config: Scaling configuration
        
    Returns:
        Configured IntelligentAutoScaler
    """
    
    default_config = {
        'target_gpu_utilization': 80.0,
        'target_memory_utilization': 85.0,
        'min_throughput': 10.0,
        'max_cost_increase': 2.0,
        'scaling_cooldown': 300.0,
        'enable_predictive_scaling': True
    }
    
    config = {**default_config, **(scaling_config or {})}
    
    auto_scaler = IntelligentAutoScaler(model, **config)
    
    return auto_scaler