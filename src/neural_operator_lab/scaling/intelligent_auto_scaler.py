"""Intelligent Auto-Scaling System

Advanced auto-scaling with load balancing, resource optimization,
and intelligent scaling decisions based on workload patterns.
"""

import torch
import torch.nn as nn
import torch.distributed as dist
import time
import threading
import queue
import json
from typing import Dict, List, Any, Optional, Tuple, Callable, Union
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from contextlib import contextmanager
import logging
import numpy as np
from collections import deque, defaultdict
import asyncio
import multiprocessing as mp

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import ray
    HAS_RAY = True
except ImportError:
    HAS_RAY = False

from neural_operator_lab.base import NeuralOperatorBase


@dataclass
class ResourceMetrics:
    """System resource metrics."""
    timestamp: float
    cpu_percent: float
    memory_percent: float
    gpu_memory_percent: float
    gpu_utilization: float
    network_io_mbps: float
    disk_io_mbps: float
    temperature_celsius: float
    power_consumption_watts: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'cpu_percent': self.cpu_percent,
            'memory_percent': self.memory_percent,
            'gpu_memory_percent': self.gpu_memory_percent,
            'gpu_utilization': self.gpu_utilization,
            'network_io_mbps': self.network_io_mbps,
            'disk_io_mbps': self.disk_io_mbps,
            'temperature_celsius': self.temperature_celsius,
            'power_consumption_watts': self.power_consumption_watts
        }


@dataclass
class WorkloadMetrics:
    """Training workload metrics."""
    timestamp: float
    batch_size: int
    throughput_samples_per_sec: float
    loss_value: float
    gradient_norm: float
    learning_rate: float
    epoch: int
    step: int
    estimated_remaining_time: float
    queue_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'timestamp': self.timestamp,
            'batch_size': self.batch_size,
            'throughput_samples_per_sec': self.throughput_samples_per_sec,
            'loss_value': self.loss_value,
            'gradient_norm': self.gradient_norm,
            'learning_rate': self.learning_rate,
            'epoch': self.epoch,
            'step': self.step,
            'estimated_remaining_time': self.estimated_remaining_time,
            'queue_size': self.queue_size
        }


@dataclass
class ScalingDecision:
    """Auto-scaling decision."""
    action: str  # "scale_up", "scale_down", "maintain", "migrate"
    target_replicas: int
    target_batch_size: int
    reason: str
    confidence: float
    estimated_performance_gain: float
    estimated_cost_change: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action': self.action,
            'target_replicas': self.target_replicas,
            'target_batch_size': self.target_batch_size,
            'reason': self.reason,
            'confidence': self.confidence,
            'estimated_performance_gain': self.estimated_performance_gain,
            'estimated_cost_change': self.estimated_cost_change
        }


class ResourceMonitor:
    """Advanced system resource monitoring."""
    
    def __init__(self, monitoring_interval: float = 1.0):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=1000)
        self.monitoring_active = False
        self.monitor_thread = None
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def start_monitoring(self):
        """Start resource monitoring in background thread."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
            self.monitor_thread.start()
            self.logger.info("Resource monitoring started")
    
    def stop_monitoring(self):
        """Stop resource monitoring."""
        self.monitoring_active = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=5.0)
        self.logger.info("Resource monitoring stopped")
    
    def _monitor_loop(self):
        """Main monitoring loop."""
        while self.monitoring_active:
            try:
                metrics = self._collect_metrics()
                with self._lock:
                    self.metrics_history.append(metrics)
                time.sleep(self.monitoring_interval)
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
    
    def _collect_metrics(self) -> ResourceMetrics:
        """Collect current system metrics."""
        timestamp = time.time()
        
        # CPU and memory
        cpu_percent = 0.0
        memory_percent = 0.0
        network_io_mbps = 0.0
        disk_io_mbps = 0.0
        temperature = 0.0
        power_consumption = 0.0
        
        if HAS_PSUTIL:
            cpu_percent = psutil.cpu_percent()
            memory_percent = psutil.virtual_memory().percent
            
            # Network I/O
            net_io = psutil.net_io_counters()
            if hasattr(self, '_last_net_io'):
                bytes_sent_diff = net_io.bytes_sent - self._last_net_io.bytes_sent
                bytes_recv_diff = net_io.bytes_recv - self._last_net_io.bytes_recv
                time_diff = timestamp - self._last_net_timestamp
                network_io_mbps = (bytes_sent_diff + bytes_recv_diff) / (1024 * 1024 * time_diff)
            
            self._last_net_io = net_io
            self._last_net_timestamp = timestamp
            
            # Disk I/O
            disk_io = psutil.disk_io_counters()
            if hasattr(self, '_last_disk_io'):
                bytes_read_diff = disk_io.read_bytes - self._last_disk_io.read_bytes
                bytes_write_diff = disk_io.write_bytes - self._last_disk_io.write_bytes
                time_diff = timestamp - self._last_disk_timestamp
                disk_io_mbps = (bytes_read_diff + bytes_write_diff) / (1024 * 1024 * time_diff)
            
            self._last_disk_io = disk_io
            self._last_disk_timestamp = timestamp
            
            # Temperature (if available)
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    temperature = np.mean([temp.current for sensor_temps in temps.values() 
                                         for temp in sensor_temps])
            except:
                pass
        
        # GPU metrics
        gpu_memory_percent = 0.0
        gpu_utilization = 0.0
        
        if torch.cuda.is_available():
            gpu_memory_percent = (torch.cuda.memory_allocated() / 
                                torch.cuda.max_memory_allocated()) * 100
            
            # Try to get GPU utilization using nvidia-ml-py if available
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                gpu_utilization = util.gpu
            except:
                pass
        
        return ResourceMetrics(
            timestamp=timestamp,
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            gpu_memory_percent=gpu_memory_percent,
            gpu_utilization=gpu_utilization,
            network_io_mbps=network_io_mbps,
            disk_io_mbps=disk_io_mbps,
            temperature_celsius=temperature,
            power_consumption_watts=power_consumption
        )
    
    def get_recent_metrics(self, window_seconds: float = 60.0) -> List[ResourceMetrics]:
        """Get recent metrics within time window."""
        cutoff_time = time.time() - window_seconds
        
        with self._lock:
            return [m for m in self.metrics_history if m.timestamp >= cutoff_time]
    
    def get_average_metrics(self, window_seconds: float = 60.0) -> Optional[ResourceMetrics]:
        """Get average metrics over time window."""
        recent_metrics = self.get_recent_metrics(window_seconds)
        
        if not recent_metrics:
            return None
        
        # Calculate averages
        avg_metrics = ResourceMetrics(
            timestamp=time.time(),
            cpu_percent=np.mean([m.cpu_percent for m in recent_metrics]),
            memory_percent=np.mean([m.memory_percent for m in recent_metrics]),
            gpu_memory_percent=np.mean([m.gpu_memory_percent for m in recent_metrics]),
            gpu_utilization=np.mean([m.gpu_utilization for m in recent_metrics]),
            network_io_mbps=np.mean([m.network_io_mbps for m in recent_metrics]),
            disk_io_mbps=np.mean([m.disk_io_mbps for m in recent_metrics]),
            temperature_celsius=np.mean([m.temperature_celsius for m in recent_metrics]),
            power_consumption_watts=np.mean([m.power_consumption_watts for m in recent_metrics])
        )
        
        return avg_metrics


class LoadBalancer:
    """Intelligent load balancer for distributed training."""
    
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.worker_queues: Dict[int, queue.Queue] = {}
        self.worker_metrics: Dict[int, List[float]] = defaultdict(list)
        self.worker_loads: Dict[int, float] = defaultdict(float)
        self._lock = threading.Lock()
        
        self.logger = logging.getLogger(__name__)
    
    def add_worker(self, worker_id: int, queue_size: int = 100):
        """Add a worker to the load balancer."""
        with self._lock:
            if worker_id not in self.worker_queues:
                self.worker_queues[worker_id] = queue.Queue(maxsize=queue_size)
                self.worker_loads[worker_id] = 0.0
                self.logger.info(f"Added worker {worker_id}")
    
    def remove_worker(self, worker_id: int):
        """Remove a worker from the load balancer."""
        with self._lock:
            if worker_id in self.worker_queues:
                del self.worker_queues[worker_id]
                del self.worker_loads[worker_id]
                if worker_id in self.worker_metrics:
                    del self.worker_metrics[worker_id]
                self.logger.info(f"Removed worker {worker_id}")
    
    def get_best_worker(self) -> Optional[int]:
        """Get the worker with lowest load."""
        with self._lock:
            if not self.worker_queues:
                return None
            
            # Find worker with minimum load
            best_worker = min(self.worker_loads.keys(), 
                            key=lambda w: self.worker_loads[w])
            
            return best_worker
    
    def submit_task(self, task: Any) -> bool:
        """Submit task to best available worker."""
        worker_id = self.get_best_worker()
        
        if worker_id is None:
            return False
        
        try:
            self.worker_queues[worker_id].put(task, block=False)
            with self._lock:
                self.worker_loads[worker_id] += 1.0
            return True
        except queue.Full:
            return False
    
    def update_worker_performance(self, worker_id: int, processing_time: float):
        """Update worker performance metrics."""
        with self._lock:
            self.worker_metrics[worker_id].append(processing_time)
            
            # Keep only recent metrics
            if len(self.worker_metrics[worker_id]) > 100:
                self.worker_metrics[worker_id] = self.worker_metrics[worker_id][-100:]
            
            # Update load based on average processing time
            if self.worker_metrics[worker_id]:
                avg_time = np.mean(self.worker_metrics[worker_id])
                queue_size = self.worker_queues[worker_id].qsize()
                self.worker_loads[worker_id] = avg_time * queue_size
    
    def get_load_distribution(self) -> Dict[int, float]:
        """Get current load distribution across workers."""
        with self._lock:
            return self.worker_loads.copy()
    
    def rebalance_load(self):
        """Rebalance load across workers."""
        with self._lock:
            if len(self.worker_queues) < 2:
                return
            
            # Find overloaded and underloaded workers
            loads = list(self.worker_loads.values())
            mean_load = np.mean(loads)
            std_load = np.std(loads)
            
            overloaded_workers = [
                wid for wid, load in self.worker_loads.items()
                if load > mean_load + std_load
            ]
            
            underloaded_workers = [
                wid for wid, load in self.worker_loads.items()
                if load < mean_load - std_load
            ]
            
            # Move tasks from overloaded to underloaded workers
            for overloaded_id in overloaded_workers:
                if not underloaded_workers:
                    break
                
                underloaded_id = underloaded_workers[0]
                overloaded_queue = self.worker_queues[overloaded_id]
                underloaded_queue = self.worker_queues[underloaded_id]
                
                # Move some tasks
                tasks_to_move = min(3, overloaded_queue.qsize() // 2)
                moved_tasks = []
                
                for _ in range(tasks_to_move):
                    try:
                        task = overloaded_queue.get(block=False)
                        moved_tasks.append(task)
                    except queue.Empty:
                        break
                
                for task in moved_tasks:
                    try:
                        underloaded_queue.put(task, block=False)
                    except queue.Full:
                        # Put back in original queue if can't move
                        overloaded_queue.put(task)
                
                # Update loads
                self.worker_loads[overloaded_id] -= len(moved_tasks)
                self.worker_loads[underloaded_id] += len(moved_tasks)
                
                if self.worker_loads[underloaded_id] >= mean_load:
                    underloaded_workers.remove(underloaded_id)


class PredictiveScaler:
    """Predictive auto-scaler using workload patterns."""
    
    def __init__(self, prediction_window: int = 100):
        self.prediction_window = prediction_window
        self.workload_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        
        # Simple moving average for prediction
        self.performance_predictor = self._create_predictor()
        
        self.logger = logging.getLogger(__name__)
    
    def _create_predictor(self):
        """Create a simple performance predictor."""
        # In a real implementation, this could be a more sophisticated ML model
        return {
            'throughput_weights': deque(maxlen=self.prediction_window),
            'resource_weights': deque(maxlen=self.prediction_window),
            'loss_weights': deque(maxlen=self.prediction_window)
        }
    
    def update_workload_metrics(self, metrics: WorkloadMetrics):
        """Update workload metrics for prediction."""
        self.workload_history.append(metrics)
        
        # Update predictor weights
        if len(self.workload_history) >= 2:
            prev_metrics = self.workload_history[-2]
            throughput_change = metrics.throughput_samples_per_sec / prev_metrics.throughput_samples_per_sec
            
            self.performance_predictor['throughput_weights'].append(throughput_change)
    
    def predict_performance(
        self, 
        target_batch_size: int, 
        target_replicas: int,
        resource_metrics: ResourceMetrics
    ) -> Tuple[float, float]:
        """Predict performance for given configuration.
        
        Returns:
            Tuple of (predicted_throughput, confidence)
        """
        if len(self.workload_history) < 5:
            return 0.0, 0.0
        
        recent_metrics = list(self.workload_history)[-10:]
        
        # Simple throughput prediction based on batch size scaling
        current_throughput = np.mean([m.throughput_samples_per_sec for m in recent_metrics])
        current_batch_size = np.mean([m.batch_size for m in recent_metrics])
        
        # Estimate throughput scaling with batch size
        batch_scale_factor = target_batch_size / current_batch_size if current_batch_size > 0 else 1.0
        
        # Account for diminishing returns
        if batch_scale_factor > 1:
            batch_scale_factor = batch_scale_factor ** 0.8  # Diminishing returns
        
        # Account for multiple replicas
        replica_efficiency = min(target_replicas * 0.9, target_replicas)  # 90% efficiency per replica
        
        # Resource utilization factor
        resource_factor = 1.0
        if resource_metrics.gpu_utilization > 90:
            resource_factor = 0.8  # Bottleneck
        elif resource_metrics.gpu_utilization < 50:
            resource_factor = 1.2  # Underutilized
        
        predicted_throughput = current_throughput * batch_scale_factor * replica_efficiency * resource_factor
        
        # Confidence based on prediction window size and variance
        throughput_variance = np.var([m.throughput_samples_per_sec for m in recent_metrics])
        confidence = min(1.0, len(recent_metrics) / 10) * (1.0 / (1.0 + throughput_variance))
        
        return predicted_throughput, confidence
    
    def recommend_scaling(
        self,
        current_workload: WorkloadMetrics,
        resource_metrics: ResourceMetrics,
        performance_target: Dict[str, float]
    ) -> ScalingDecision:
        """Recommend scaling action based on current state and targets."""
        
        # Default decision
        decision = ScalingDecision(
            action="maintain",
            target_replicas=1,
            target_batch_size=current_workload.batch_size,
            reason="No scaling needed",
            confidence=0.5,
            estimated_performance_gain=0.0,
            estimated_cost_change=0.0
        )
        
        target_throughput = performance_target.get('throughput_samples_per_sec', 0)
        target_latency_ms = performance_target.get('latency_ms', 1000)
        
        current_throughput = current_workload.throughput_samples_per_sec
        
        # Check if scaling is needed
        if target_throughput > 0 and current_throughput < target_throughput * 0.8:
            # Need to scale up
            
            # Determine scaling strategy
            if resource_metrics.gpu_utilization < 70:
                # Increase batch size
                new_batch_size = min(current_workload.batch_size * 2, 512)
                predicted_throughput, confidence = self.predict_performance(
                    new_batch_size, 1, resource_metrics
                )
                
                decision.action = "scale_up"
                decision.target_batch_size = new_batch_size
                decision.target_replicas = 1
                decision.reason = f"Increase batch size to {new_batch_size} for better GPU utilization"
                decision.confidence = confidence
                decision.estimated_performance_gain = (predicted_throughput - current_throughput) / current_throughput
                
            else:
                # Add more replicas
                new_replicas = 2
                predicted_throughput, confidence = self.predict_performance(
                    current_workload.batch_size, new_replicas, resource_metrics
                )
                
                decision.action = "scale_up"
                decision.target_batch_size = current_workload.batch_size
                decision.target_replicas = new_replicas
                decision.reason = f"Add replica due to high GPU utilization ({resource_metrics.gpu_utilization:.1f}%)"
                decision.confidence = confidence
                decision.estimated_performance_gain = (predicted_throughput - current_throughput) / current_throughput
                decision.estimated_cost_change = 1.0  # Double cost
        
        elif current_throughput > target_throughput * 1.2 and resource_metrics.gpu_utilization < 40:
            # Can scale down
            new_batch_size = max(current_workload.batch_size // 2, 1)
            predicted_throughput, confidence = self.predict_performance(
                new_batch_size, 1, resource_metrics
            )
            
            if predicted_throughput >= target_throughput * 0.9:
                decision.action = "scale_down"
                decision.target_batch_size = new_batch_size
                decision.target_replicas = 1
                decision.reason = f"Reduce batch size to {new_batch_size} to save resources"
                decision.confidence = confidence
                decision.estimated_performance_gain = (predicted_throughput - current_throughput) / current_throughput
                decision.estimated_cost_change = -0.5  # Save resources
        
        self.scaling_history.append(decision)
        return decision


class IntelligentAutoScaler:
    """Main intelligent auto-scaling orchestrator."""
    
    def __init__(
        self,
        resource_monitor: Optional[ResourceMonitor] = None,
        load_balancer: Optional[LoadBalancer] = None,
        predictive_scaler: Optional[PredictiveScaler] = None,
        scaling_interval: float = 30.0
    ):
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.load_balancer = load_balancer or LoadBalancer()
        self.predictive_scaler = predictive_scaler or PredictiveScaler()
        
        self.scaling_interval = scaling_interval
        self.scaling_active = False
        self.scaling_thread = None
        
        # Performance targets
        self.performance_targets = {
            'throughput_samples_per_sec': 100.0,
            'latency_ms': 100.0,
            'gpu_utilization_target': 80.0,
            'memory_utilization_max': 90.0
        }
        
        # Scaling cooldown to prevent oscillation
        self.last_scaling_time = 0
        self.scaling_cooldown = 120.0  # 2 minutes
        
        self.logger = logging.getLogger(__name__)
    
    def set_performance_targets(self, targets: Dict[str, float]):
        """Set performance targets for auto-scaling."""
        self.performance_targets.update(targets)
        self.logger.info(f"Updated performance targets: {self.performance_targets}")
    
    def start_auto_scaling(self):
        """Start auto-scaling in background."""
        if not self.scaling_active:
            self.scaling_active = True
            self.resource_monitor.start_monitoring()
            
            self.scaling_thread = threading.Thread(target=self._scaling_loop, daemon=True)
            self.scaling_thread.start()
            
            self.logger.info("Auto-scaling started")
    
    def stop_auto_scaling(self):
        """Stop auto-scaling."""
        self.scaling_active = False
        self.resource_monitor.stop_monitoring()
        
        if self.scaling_thread:
            self.scaling_thread.join(timeout=10.0)
        
        self.logger.info("Auto-scaling stopped")
    
    def _scaling_loop(self):
        """Main auto-scaling decision loop."""
        while self.scaling_active:
            try:
                # Check cooldown period
                if time.time() - self.last_scaling_time < self.scaling_cooldown:
                    time.sleep(self.scaling_interval)
                    continue
                
                # Get current metrics
                resource_metrics = self.resource_monitor.get_average_metrics(60.0)
                if resource_metrics is None:
                    time.sleep(self.scaling_interval)
                    continue
                
                # Check if we have recent workload metrics
                if not self.predictive_scaler.workload_history:
                    time.sleep(self.scaling_interval)
                    continue
                
                current_workload = self.predictive_scaler.workload_history[-1]
                
                # Get scaling recommendation
                scaling_decision = self.predictive_scaler.recommend_scaling(
                    current_workload, resource_metrics, self.performance_targets
                )
                
                # Execute scaling decision if confidence is high enough
                if scaling_decision.confidence > 0.7 and scaling_decision.action != "maintain":
                    self._execute_scaling_decision(scaling_decision)
                    self.last_scaling_time = time.time()
                
                # Rebalance load periodically
                self.load_balancer.rebalance_load()
                
                time.sleep(self.scaling_interval)
                
            except Exception as e:
                self.logger.error(f"Error in scaling loop: {e}")
                time.sleep(self.scaling_interval)
    
    def _execute_scaling_decision(self, decision: ScalingDecision):
        """Execute a scaling decision."""
        self.logger.info(f"Executing scaling decision: {decision.action}")
        self.logger.info(f"Reason: {decision.reason}")
        self.logger.info(f"Confidence: {decision.confidence:.2f}")
        self.logger.info(f"Expected performance gain: {decision.estimated_performance_gain:.2%}")
        
        if decision.action == "scale_up":
            if decision.target_replicas > 1:
                # Add workers
                current_workers = len(self.load_balancer.worker_queues)
                for i in range(current_workers, decision.target_replicas):
                    self.load_balancer.add_worker(i)
            
        elif decision.action == "scale_down":
            # Remove workers
            current_workers = len(self.load_balancer.worker_queues)
            workers_to_remove = max(0, current_workers - decision.target_replicas)
            
            for i in range(workers_to_remove):
                worker_id = max(self.load_balancer.worker_queues.keys())
                self.load_balancer.remove_worker(worker_id)
        
        # Note: Batch size changes would need to be communicated to the training loop
        # This is implementation-specific and would depend on the training framework
    
    def update_workload_metrics(self, metrics: WorkloadMetrics):
        """Update current workload metrics."""
        self.predictive_scaler.update_workload_metrics(metrics)
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        resource_metrics = self.resource_monitor.get_average_metrics(60.0)
        load_distribution = self.load_balancer.get_load_distribution()
        
        recent_workload = None
        if self.predictive_scaler.workload_history:
            recent_workload = self.predictive_scaler.workload_history[-1].to_dict()
        
        recent_scaling = None
        if self.predictive_scaler.scaling_history:
            recent_scaling = self.predictive_scaler.scaling_history[-1].to_dict()
        
        return {
            'scaling_active': self.scaling_active,
            'performance_targets': self.performance_targets,
            'resource_metrics': resource_metrics.to_dict() if resource_metrics else None,
            'load_distribution': load_distribution,
            'recent_workload_metrics': recent_workload,
            'recent_scaling_decision': recent_scaling,
            'cooldown_remaining': max(0, self.scaling_cooldown - (time.time() - self.last_scaling_time))
        }
    
    def force_scaling_decision(
        self, 
        action: str, 
        target_replicas: int = 1, 
        target_batch_size: Optional[int] = None
    ) -> bool:
        """Force a specific scaling decision."""
        if time.time() - self.last_scaling_time < 10.0:  # Minimum 10s between forced decisions
            return False
        
        current_workload = None
        if self.predictive_scaler.workload_history:
            current_workload = self.predictive_scaler.workload_history[-1]
            current_batch_size = current_workload.batch_size
        else:
            current_batch_size = 32
        
        decision = ScalingDecision(
            action=action,
            target_replicas=target_replicas,
            target_batch_size=target_batch_size or current_batch_size,
            reason="Manual scaling decision",
            confidence=1.0,
            estimated_performance_gain=0.0,
            estimated_cost_change=0.0
        )
        
        self._execute_scaling_decision(decision)
        self.last_scaling_time = time.time()
        
        return True


@contextmanager
def auto_scaling_context(
    performance_targets: Dict[str, float],
    scaling_interval: float = 30.0
):
    """Context manager for automatic scaling during training."""
    
    auto_scaler = IntelligentAutoScaler(scaling_interval=scaling_interval)
    auto_scaler.set_performance_targets(performance_targets)
    auto_scaler.start_auto_scaling()
    
    try:
        yield auto_scaler
    finally:
        auto_scaler.stop_auto_scaling()