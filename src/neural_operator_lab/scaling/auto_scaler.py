"""Auto-scaling system for neural operator workloads."""

import torch
import time
import logging
import threading
import psutil
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from collections import deque, defaultdict


class ScalingAction(Enum):
    """Types of scaling actions."""
    SCALE_UP = "scale_up"
    SCALE_DOWN = "scale_down" 
    NO_ACTION = "no_action"
    EMERGENCY_SCALE_DOWN = "emergency_scale_down"


class ResourceType(Enum):
    """Types of resources to scale."""
    BATCH_SIZE = "batch_size"
    MODEL_PARALLEL = "model_parallel"
    DATA_PARALLEL = "data_parallel"
    PRECISION = "precision"
    COMPUTE_NODES = "compute_nodes"


@dataclass
class ResourceMetrics:
    """Current resource utilization metrics."""
    cpu_percent: float
    memory_percent: float
    gpu_percent: float = 0.0
    gpu_memory_percent: float = 0.0
    disk_io_percent: float = 0.0
    network_io_mbps: float = 0.0
    throughput_samples_per_sec: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    queue_length: int = 0
    timestamp: float = field(default_factory=time.time)


@dataclass 
class ScalingThresholds:
    """Thresholds for scaling decisions."""
    # Scale up thresholds
    cpu_scale_up: float = 80.0
    memory_scale_up: float = 85.0
    gpu_scale_up: float = 85.0
    gpu_memory_scale_up: float = 90.0
    throughput_scale_up: float = 0.8  # Ratio of target throughput
    latency_scale_up: float = 1000.0  # ms
    queue_scale_up: int = 100
    
    # Scale down thresholds  
    cpu_scale_down: float = 30.0
    memory_scale_down: float = 40.0
    gpu_scale_down: float = 30.0
    gpu_memory_scale_down: float = 40.0
    throughput_scale_down: float = 0.3  # Ratio of target throughput
    latency_scale_down: float = 100.0  # ms
    queue_scale_down: int = 10
    
    # Emergency thresholds
    memory_emergency: float = 98.0
    gpu_memory_emergency: float = 98.0
    error_rate_emergency: float = 0.1  # 10% error rate


@dataclass
class ScalingPolicy:
    """Configuration for scaling behavior."""
    # General settings
    enabled: bool = True
    check_interval: float = 30.0  # seconds
    cooldown_period: float = 300.0  # seconds between scaling actions
    
    # Scaling limits
    min_batch_size: int = 1
    max_batch_size: int = 512
    min_workers: int = 1
    max_workers: int = 16
    min_replicas: int = 1
    max_replicas: int = 10
    
    # Scaling factors
    scale_up_factor: float = 1.5
    scale_down_factor: float = 0.7
    emergency_scale_down_factor: float = 0.5
    
    # Decision parameters
    decision_history_size: int = 10
    consensus_threshold: float = 0.6  # Fraction of recent decisions that must agree
    predictive_scaling: bool = True
    aggressive_scaling: bool = False
    
    # Thresholds
    thresholds: ScalingThresholds = field(default_factory=ScalingThresholds)


class MetricsCollector:
    """Collects system and application metrics."""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
    
    def collect_system_metrics(self) -> ResourceMetrics:
        """Collect current system resource metrics."""
        try:
            # CPU and memory
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            disk_io_percent = min(100, (disk_io.read_bytes + disk_io.write_bytes) / (1024**3) * 10)
            
            # Network I/O
            network_io = psutil.net_io_counters()
            network_mbps = (network_io.bytes_sent + network_io.bytes_recv) / (1024**2)
            
            # GPU metrics
            gpu_percent = 0.0
            gpu_memory_percent = 0.0
            
            if torch.cuda.is_available():
                try:
                    gpu_memory_used = torch.cuda.memory_allocated()
                    gpu_memory_total = torch.cuda.get_device_properties(0).total_memory
                    gpu_memory_percent = (gpu_memory_used / gpu_memory_total) * 100
                    
                    # Try to get GPU utilization
                    try:
                        import pynvml
                        pynvml.nvmlInit()
                        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                        utilization = pynvml.nvmlDeviceGetUtilizationRates(handle)
                        gpu_percent = utilization.gpu
                    except ImportError:
                        gpu_percent = gpu_memory_percent  # Fallback approximation
                
                except Exception as e:
                    self.logger.debug(f"Could not collect GPU metrics: {e}")
            
            return ResourceMetrics(
                cpu_percent=cpu_percent,
                memory_percent=memory_percent,
                gpu_percent=gpu_percent,
                gpu_memory_percent=gpu_memory_percent,
                disk_io_percent=disk_io_percent,
                network_io_mbps=network_mbps
            )
        
        except Exception as e:
            self.logger.error(f"Error collecting system metrics: {e}")
            return ResourceMetrics(cpu_percent=0, memory_percent=0)


class ScalingDecisionEngine:
    """Makes scaling decisions based on metrics and policies."""
    
    def __init__(self, policy: ScalingPolicy):
        self.policy = policy
        self.decision_history = deque(maxlen=policy.decision_history_size)
        self.last_scaling_time = defaultdict(float)
        self.logger = logging.getLogger(__name__)
    
    def should_scale(self, metrics: ResourceMetrics, 
                    resource_type: ResourceType) -> ScalingAction:
        """Determine if scaling action is needed."""
        # Check cooldown period
        if self._in_cooldown(resource_type):
            return ScalingAction.NO_ACTION
        
        # Emergency scaling
        emergency_action = self._check_emergency_conditions(metrics)
        if emergency_action != ScalingAction.NO_ACTION:
            return emergency_action
        
        # Regular scaling decision
        scale_up_score = self._calculate_scale_up_score(metrics)
        scale_down_score = self._calculate_scale_down_score(metrics)
        
        # Make decision based on scores
        if scale_up_score > scale_down_score and scale_up_score > 0.5:
            action = ScalingAction.SCALE_UP
        elif scale_down_score > scale_up_score and scale_down_score > 0.5:
            action = ScalingAction.SCALE_DOWN
        else:
            action = ScalingAction.NO_ACTION
        
        # Apply consensus filtering
        if self.policy.consensus_threshold > 0:
            action = self._apply_consensus_filter(action, resource_type)
        
        # Record decision
        self.decision_history.append({
            'timestamp': time.time(),
            'resource_type': resource_type,
            'action': action,
            'metrics': metrics,
            'scale_up_score': scale_up_score,
            'scale_down_score': scale_down_score
        })
        
        return action
    
    def _in_cooldown(self, resource_type: ResourceType) -> bool:
        """Check if resource is in cooldown period."""
        last_time = self.last_scaling_time[resource_type]
        return (time.time() - last_time) < self.policy.cooldown_period
    
    def _check_emergency_conditions(self, metrics: ResourceMetrics) -> ScalingAction:
        """Check for emergency conditions requiring immediate action."""
        thresholds = self.policy.thresholds
        
        # Emergency memory conditions
        if (metrics.memory_percent >= thresholds.memory_emergency or
            metrics.gpu_memory_percent >= thresholds.gpu_memory_emergency):
            self.logger.warning(f"Emergency memory condition detected: "
                              f"RAM={metrics.memory_percent}%, GPU={metrics.gpu_memory_percent}%")
            return ScalingAction.EMERGENCY_SCALE_DOWN
        
        # High error rate
        if metrics.error_rate >= thresholds.error_rate_emergency:
            self.logger.warning(f"High error rate detected: {metrics.error_rate}")
            return ScalingAction.EMERGENCY_SCALE_DOWN
        
        return ScalingAction.NO_ACTION
    
    def _calculate_scale_up_score(self, metrics: ResourceMetrics) -> float:
        """Calculate score for scaling up (0-1)."""
        thresholds = self.policy.thresholds
        score = 0.0
        factors = 0
        
        # CPU utilization
        if metrics.cpu_percent > thresholds.cpu_scale_up:
            score += min(1.0, (metrics.cpu_percent - thresholds.cpu_scale_up) / 
                        (100 - thresholds.cpu_scale_up))
            factors += 1
        
        # Memory utilization
        if metrics.memory_percent > thresholds.memory_scale_up:
            score += min(1.0, (metrics.memory_percent - thresholds.memory_scale_up) / 
                        (100 - thresholds.memory_scale_up))
            factors += 1
        
        # GPU utilization
        if metrics.gpu_percent > thresholds.gpu_scale_up:
            score += min(1.0, (metrics.gpu_percent - thresholds.gpu_scale_up) / 
                        (100 - thresholds.gpu_scale_up))
            factors += 1
        
        # GPU memory
        if metrics.gpu_memory_percent > thresholds.gpu_memory_scale_up:
            score += min(1.0, (metrics.gpu_memory_percent - thresholds.gpu_memory_scale_up) / 
                        (100 - thresholds.gpu_memory_scale_up))
            factors += 1
        
        # Latency
        if metrics.latency_ms > thresholds.latency_scale_up:
            score += min(1.0, (metrics.latency_ms - thresholds.latency_scale_up) / 
                        (thresholds.latency_scale_up * 2))
            factors += 1
        
        # Queue length
        if metrics.queue_length > thresholds.queue_scale_up:
            score += min(1.0, (metrics.queue_length - thresholds.queue_scale_up) / 
                        thresholds.queue_scale_up)
            factors += 1
        
        return score / max(1, factors)
    
    def _calculate_scale_down_score(self, metrics: ResourceMetrics) -> float:
        """Calculate score for scaling down (0-1)."""
        thresholds = self.policy.thresholds
        score = 0.0
        factors = 0
        
        # CPU utilization
        if metrics.cpu_percent < thresholds.cpu_scale_down:
            score += min(1.0, (thresholds.cpu_scale_down - metrics.cpu_percent) / 
                        thresholds.cpu_scale_down)
            factors += 1
        
        # Memory utilization
        if metrics.memory_percent < thresholds.memory_scale_down:
            score += min(1.0, (thresholds.memory_scale_down - metrics.memory_percent) / 
                        thresholds.memory_scale_down)
            factors += 1
        
        # GPU utilization
        if metrics.gpu_percent < thresholds.gpu_scale_down:
            score += min(1.0, (thresholds.gpu_scale_down - metrics.gpu_percent) / 
                        thresholds.gpu_scale_down)
            factors += 1
        
        # Latency (good latency supports scaling down)
        if metrics.latency_ms < thresholds.latency_scale_down:
            score += min(1.0, (thresholds.latency_scale_down - metrics.latency_ms) / 
                        thresholds.latency_scale_down)
            factors += 1
        
        # Queue length
        if metrics.queue_length < thresholds.queue_scale_down:
            score += min(1.0, (thresholds.queue_scale_down - metrics.queue_length) / 
                        max(1, thresholds.queue_scale_down))
            factors += 1
        
        return score / max(1, factors)
    
    def _apply_consensus_filter(self, action: ScalingAction, 
                               resource_type: ResourceType) -> ScalingAction:
        """Apply consensus filtering to reduce oscillations."""
        if len(self.decision_history) < 3:
            return action
        
        # Count recent decisions for this resource type
        recent_decisions = [
            d['action'] for d in list(self.decision_history)[-5:]
            if d['resource_type'] == resource_type
        ]
        
        if not recent_decisions:
            return action
        
        # Count votes for each action
        action_counts = defaultdict(int)
        for decision in recent_decisions:
            action_counts[decision] += 1
        
        total_decisions = len(recent_decisions)
        consensus_needed = int(total_decisions * self.policy.consensus_threshold)
        
        # Check if current action has consensus
        if action_counts[action] >= consensus_needed:
            return action
        
        # Check if any action has consensus
        for act, count in action_counts.items():
            if count >= consensus_needed:
                return act
        
        # No consensus, default to no action
        return ScalingAction.NO_ACTION


class AutoScaler:
    """Main auto-scaling coordinator."""
    
    def __init__(self, policy: ScalingPolicy = None):
        self.policy = policy or ScalingPolicy()
        self.metrics_collector = MetricsCollector()
        self.decision_engine = ScalingDecisionEngine(self.policy)
        
        # State
        self.running = False
        self.scaling_thread: Optional[threading.Thread] = None
        
        # Callbacks for scaling actions
        self.scaling_callbacks: Dict[ResourceType, List[Callable]] = defaultdict(list)
        
        # Metrics history
        self.metrics_history = deque(maxlen=1000)
        
        # Logging
        self.logger = logging.getLogger(__name__)
        
        # Current resource states
        self.current_batch_size = 32
        self.current_workers = 1
        self.current_replicas = 1
        self.current_precision = 'fp32'
    
    def register_scaling_callback(self, resource_type: ResourceType, 
                                callback: Callable[[ScalingAction, ResourceMetrics], bool]):
        """Register callback for scaling actions.
        
        Args:
            resource_type: Type of resource to scale
            callback: Function that implements the scaling action
                     Should return True if scaling was successful
        """
        self.scaling_callbacks[resource_type].append(callback)
        self.logger.info(f"Registered scaling callback for {resource_type.value}")
    
    def start(self):
        """Start auto-scaling monitoring."""
        if self.running:
            self.logger.warning("Auto-scaler is already running")
            return
        
        if not self.policy.enabled:
            self.logger.info("Auto-scaling is disabled")
            return
        
        self.running = True
        self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
        self.scaling_thread.start()
        
        self.logger.info(f"Started auto-scaler with {self.policy.check_interval}s interval")
    
    def stop(self):
        """Stop auto-scaling monitoring."""
        if not self.running:
            return
        
        self.running = False
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        self.logger.info("Stopped auto-scaler")
    
    def _scaling_loop(self):
        """Main scaling monitoring loop."""
        while self.running:
            try:
                # Collect current metrics
                metrics = self.metrics_collector.collect_system_metrics()
                
                # Add application-specific metrics if available
                metrics = self._enrich_metrics(metrics)
                
                # Store metrics history
                self.metrics_history.append(metrics)
                
                # Check each resource type for scaling opportunities
                for resource_type in ResourceType:
                    if resource_type in self.scaling_callbacks:
                        action = self.decision_engine.should_scale(metrics, resource_type)
                        
                        if action != ScalingAction.NO_ACTION:
                            self._execute_scaling_action(resource_type, action, metrics)
                
                # Sleep until next check
                time.sleep(self.policy.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.policy.check_interval)
    
    def _enrich_metrics(self, base_metrics: ResourceMetrics) -> ResourceMetrics:
        """Enrich metrics with application-specific data."""
        # This would be extended to collect application-specific metrics
        # For now, return base metrics
        return base_metrics
    
    def _execute_scaling_action(self, resource_type: ResourceType, 
                              action: ScalingAction, metrics: ResourceMetrics):
        """Execute a scaling action."""
        self.logger.info(f"Executing {action.value} for {resource_type.value}")
        
        # Record scaling attempt
        self.decision_engine.last_scaling_time[resource_type] = time.time()
        
        success = False
        
        # Execute callbacks for this resource type
        for callback in self.scaling_callbacks[resource_type]:
            try:
                if callback(action, metrics):
                    success = True
                    break  # Stop after first successful callback
            except Exception as e:
                self.logger.error(f"Scaling callback failed: {e}")
        
        if success:
            self.logger.info(f"Successfully executed {action.value} for {resource_type.value}")
        else:
            self.logger.warning(f"Failed to execute {action.value} for {resource_type.value}")
    
    def get_scaling_report(self) -> Dict[str, Any]:
        """Get comprehensive scaling report."""
        if not self.metrics_history:
            return {"message": "No metrics available"}
        
        recent_metrics = list(self.metrics_history)[-20:]
        
        # Calculate trends
        cpu_trend = [m.cpu_percent for m in recent_metrics]
        memory_trend = [m.memory_percent for m in recent_metrics]
        gpu_trend = [m.gpu_percent for m in recent_metrics]
        
        # Scaling history from decision engine
        scaling_history = list(self.decision_engine.decision_history)[-20:]
        
        return {
            'current_state': {
                'batch_size': self.current_batch_size,
                'workers': self.current_workers,
                'replicas': self.current_replicas,
                'precision': self.current_precision
            },
            'current_metrics': {
                'cpu_percent': recent_metrics[-1].cpu_percent if recent_metrics else 0,
                'memory_percent': recent_metrics[-1].memory_percent if recent_metrics else 0,
                'gpu_percent': recent_metrics[-1].gpu_percent if recent_metrics else 0,
                'gpu_memory_percent': recent_metrics[-1].gpu_memory_percent if recent_metrics else 0,
            },
            'trends': {
                'cpu_avg': np.mean(cpu_trend) if cpu_trend else 0,
                'memory_avg': np.mean(memory_trend) if memory_trend else 0,
                'gpu_avg': np.mean(gpu_trend) if gpu_trend else 0,
                'cpu_trend': cpu_trend,
                'memory_trend': memory_trend,
                'gpu_trend': gpu_trend
            },
            'scaling_history': [
                {
                    'timestamp': h.get('timestamp', 0),
                    'resource_type': h.get('resource_type', {}).value if h.get('resource_type') else 'unknown',
                    'action': h.get('action', {}).value if h.get('action') else 'unknown',
                    'scale_up_score': h.get('scale_up_score', 0),
                    'scale_down_score': h.get('scale_down_score', 0)
                }
                for h in scaling_history
            ],
            'policy': {
                'enabled': self.policy.enabled,
                'check_interval': self.policy.check_interval,
                'cooldown_period': self.policy.cooldown_period,
                'aggressive_scaling': self.policy.aggressive_scaling
            }
        }
    
    def manual_scale(self, resource_type: ResourceType, action: ScalingAction) -> bool:
        """Manually trigger scaling action."""
        if not self.running:
            self.logger.error("Auto-scaler must be running for manual scaling")
            return False
        
        current_metrics = self.metrics_collector.collect_system_metrics()
        self._execute_scaling_action(resource_type, action, current_metrics)
        
        return True
    
    def update_policy(self, **kwargs):
        """Update scaling policy parameters."""
        for key, value in kwargs.items():
            if hasattr(self.policy, key):
                setattr(self.policy, key, value)
                self.logger.info(f"Updated policy {key} to {value}")
            elif hasattr(self.policy.thresholds, key):
                setattr(self.policy.thresholds, key, value)
                self.logger.info(f"Updated threshold {key} to {value}")
            else:
                self.logger.warning(f"Unknown policy parameter: {key}")


def create_training_auto_scaler(trainer) -> AutoScaler:
    """Create auto-scaler configured for training workloads."""
    
    # Create policy optimized for training
    policy = ScalingPolicy(
        check_interval=30.0,
        cooldown_period=180.0,  # Shorter cooldown for training
        min_batch_size=1,
        max_batch_size=min(512, trainer.config.batch_size * 8),
        scale_up_factor=1.3,
        scale_down_factor=0.8,
        aggressive_scaling=True,  # More responsive for training
        predictive_scaling=True
    )
    
    # Adjust thresholds for training
    policy.thresholds.memory_scale_up = 80.0  # More conservative for training
    policy.thresholds.gpu_memory_scale_up = 85.0
    policy.thresholds.memory_scale_down = 50.0
    
    scaler = AutoScaler(policy)
    
    # Register batch size scaling callback
    def batch_size_callback(action: ScalingAction, metrics: ResourceMetrics) -> bool:
        try:
            current_bs = trainer.config.batch_size
            
            if action == ScalingAction.SCALE_UP:
                new_bs = min(policy.max_batch_size, int(current_bs * policy.scale_up_factor))
            elif action == ScalingAction.SCALE_DOWN:
                new_bs = max(policy.min_batch_size, int(current_bs * policy.scale_down_factor))
            elif action == ScalingAction.EMERGENCY_SCALE_DOWN:
                new_bs = max(policy.min_batch_size, int(current_bs * policy.emergency_scale_down_factor))
            else:
                return False
            
            if new_bs != current_bs:
                trainer.config.batch_size = new_bs
                scaler.current_batch_size = new_bs
                trainer.logger.info(f"Auto-scaled batch size from {current_bs} to {new_bs}")
                return True
            
            return False
            
        except Exception as e:
            trainer.logger.error(f"Batch size scaling failed: {e}")
            return False
    
    scaler.register_scaling_callback(ResourceType.BATCH_SIZE, batch_size_callback)
    
    return scaler