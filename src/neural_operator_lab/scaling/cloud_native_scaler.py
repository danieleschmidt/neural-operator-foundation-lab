"""Cloud-native auto-scaling for neural operator workloads."""

import torch
import torch.distributed as dist
import time
import json
import logging
import threading
import asyncio
import subprocess
import os
from typing import Dict, List, Any, Optional, Callable, NamedTuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import numpy as np
from collections import deque, defaultdict
from datetime import datetime, timedelta


class CloudProvider(Enum):
    """Supported cloud providers."""
    AWS = "aws"
    GCP = "gcp"
    AZURE = "azure"
    KUBERNETES = "kubernetes"
    LOCAL = "local"


class ScalingStrategy(Enum):
    """Scaling strategies."""
    REACTIVE = "reactive"
    PREDICTIVE = "predictive"
    HYBRID = "hybrid"
    ML_BASED = "ml_based"


@dataclass
class CloudResource:
    """Cloud resource specification."""
    provider: CloudProvider
    instance_type: str
    cpu_cores: int
    memory_gb: int
    gpu_count: int = 0
    gpu_type: str = ""
    cost_per_hour: float = 0.0
    availability_zone: str = ""
    spot_instance: bool = False


@dataclass
class WorkloadMetrics:
    """Comprehensive workload metrics."""
    timestamp: float
    throughput_samples_per_sec: float
    latency_p50_ms: float
    latency_p95_ms: float
    latency_p99_ms: float
    cpu_utilization: float
    memory_utilization: float
    gpu_utilization: float
    gpu_memory_utilization: float
    error_rate: float
    queue_depth: int
    active_workers: int
    pending_requests: int
    cost_per_sample: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class ScalingPolicy:
    """Auto-scaling policy configuration."""
    min_instances: int = 1
    max_instances: int = 100
    target_cpu_utilization: float = 70.0
    target_gpu_utilization: float = 80.0
    target_latency_ms: float = 100.0
    scale_up_threshold: float = 80.0
    scale_down_threshold: float = 30.0
    scale_up_cooldown_sec: int = 300
    scale_down_cooldown_sec: int = 600
    enable_spot_instances: bool = True
    max_cost_per_hour: float = 10.0
    enable_preemption_handling: bool = True


class WorkloadPredictor:
    """ML-based workload prediction for proactive scaling."""
    
    def __init__(self, history_window: int = 1000):
        self.history_window = history_window
        self.metrics_history = deque(maxlen=history_window)
        self.prediction_model = None
        self.logger = logging.getLogger(__name__)
        
    def add_metrics(self, metrics: WorkloadMetrics):
        """Add metrics to history for learning."""
        self.metrics_history.append(metrics)
        
        # Retrain model periodically
        if len(self.metrics_history) >= 100 and len(self.metrics_history) % 50 == 0:
            self._train_prediction_model()
    
    def _train_prediction_model(self):
        """Train simple prediction model on historical data."""
        if len(self.metrics_history) < 50:
            return
        
        try:
            # Extract features and targets
            features = []
            targets = []
            
            for i in range(len(self.metrics_history) - 10):
                # Features: last 10 time points
                feature_window = []
                for j in range(10):
                    metrics = self.metrics_history[i + j]
                    feature_window.extend([
                        metrics.throughput_samples_per_sec,
                        metrics.cpu_utilization,
                        metrics.gpu_utilization,
                        metrics.latency_p95_ms,
                        metrics.queue_depth
                    ])
                features.append(feature_window)
                
                # Target: throughput in next time step
                next_metrics = self.metrics_history[i + 10]
                targets.append(next_metrics.throughput_samples_per_sec)
            
            # Simple linear regression (would use sklearn in production)
            features = np.array(features)
            targets = np.array(targets)
            
            if features.size > 0 and targets.size > 0:
                # Pseudo-inverse for linear regression
                self.prediction_model = np.linalg.pinv(features) @ targets
                self.logger.info("Updated workload prediction model")
        
        except Exception as e:
            self.logger.warning(f"Failed to train prediction model: {e}")
    
    def predict_workload(self, horizon_minutes: int = 5) -> Dict[str, float]:
        """Predict workload metrics for the next horizon."""
        if self.prediction_model is None or len(self.metrics_history) < 10:
            return {}
        
        try:
            # Extract recent features
            recent_features = []
            for i in range(10):
                metrics = self.metrics_history[-(10-i)]
                recent_features.extend([
                    metrics.throughput_samples_per_sec,
                    metrics.cpu_utilization,
                    metrics.gpu_utilization,
                    metrics.latency_p95_ms,
                    metrics.queue_depth
                ])
            
            # Predict
            predicted_throughput = float(np.dot(recent_features, self.prediction_model))
            
            # Estimate other metrics based on throughput
            current_metrics = self.metrics_history[-1]
            throughput_ratio = predicted_throughput / max(current_metrics.throughput_samples_per_sec, 1)
            
            return {
                'predicted_throughput': max(0, predicted_throughput),
                'predicted_cpu_utilization': min(100, current_metrics.cpu_utilization * throughput_ratio),
                'predicted_gpu_utilization': min(100, current_metrics.gpu_utilization * throughput_ratio),
                'predicted_latency_ms': current_metrics.latency_p95_ms / max(throughput_ratio, 0.1),
                'confidence': min(1.0, len(self.metrics_history) / self.history_window)
            }
        
        except Exception as e:
            self.logger.warning(f"Prediction failed: {e}")
            return {}


class CloudResourceManager:
    """Manage cloud resources across providers."""
    
    def __init__(self, provider: CloudProvider, credentials: Dict[str, str] = None):
        self.provider = provider
        self.credentials = credentials or {}
        self.active_instances = {}
        self.pending_instances = set()
        self.logger = logging.getLogger(__name__)
        
        # Initialize provider-specific clients
        self._init_provider_client()
    
    def _init_provider_client(self):
        """Initialize cloud provider client."""
        if self.provider == CloudProvider.KUBERNETES:
            self._init_kubernetes_client()
        elif self.provider == CloudProvider.AWS:
            self._init_aws_client()
        elif self.provider == CloudProvider.GCP:
            self._init_gcp_client()
        elif self.provider == CloudProvider.AZURE:
            self._init_azure_client()
    
    def _init_kubernetes_client(self):
        """Initialize Kubernetes client."""
        try:
            # Would use kubernetes python client in production
            self.k8s_namespace = os.environ.get('K8S_NAMESPACE', 'default')
            self.k8s_deployment = os.environ.get('K8S_DEPLOYMENT', 'neural-operator')
        except Exception as e:
            self.logger.warning(f"Kubernetes client init failed: {e}")
    
    def _init_aws_client(self):
        """Initialize AWS client."""
        try:
            # Would use boto3 in production
            self.aws_region = os.environ.get('AWS_REGION', 'us-west-2')
        except Exception as e:
            self.logger.warning(f"AWS client init failed: {e}")
    
    def _init_gcp_client(self):
        """Initialize GCP client."""
        try:
            # Would use google-cloud-compute in production
            self.gcp_project = os.environ.get('GCP_PROJECT', '')
            self.gcp_zone = os.environ.get('GCP_ZONE', 'us-central1-a')
        except Exception as e:
            self.logger.warning(f"GCP client init failed: {e}")
    
    def _init_azure_client(self):
        """Initialize Azure client."""
        try:
            # Would use azure-mgmt-compute in production
            self.azure_subscription = os.environ.get('AZURE_SUBSCRIPTION_ID', '')
            self.azure_resource_group = os.environ.get('AZURE_RESOURCE_GROUP', '')
        except Exception as e:
            self.logger.warning(f"Azure client init failed: {e}")
    
    async def scale_up(self, target_instances: int, resource_spec: CloudResource) -> List[str]:
        """Scale up to target number of instances."""
        current_count = len(self.active_instances)
        instances_to_add = target_instances - current_count
        
        if instances_to_add <= 0:
            return []
        
        self.logger.info(f"Scaling up: adding {instances_to_add} instances")
        
        new_instances = []
        for i in range(instances_to_add):
            try:
                instance_id = await self._launch_instance(resource_spec)
                if instance_id:
                    new_instances.append(instance_id)
                    self.pending_instances.add(instance_id)
            except Exception as e:
                self.logger.error(f"Failed to launch instance {i}: {e}")
        
        return new_instances
    
    async def scale_down(self, target_instances: int) -> List[str]:
        """Scale down to target number of instances."""
        current_count = len(self.active_instances)
        instances_to_remove = current_count - target_instances
        
        if instances_to_remove <= 0:
            return []
        
        self.logger.info(f"Scaling down: removing {instances_to_remove} instances")
        
        # Select instances to terminate (prefer spot instances)
        instances_to_terminate = []
        sorted_instances = sorted(
            self.active_instances.items(),
            key=lambda x: (not x[1].get('spot', False), x[1].get('launch_time', 0))
        )
        
        for instance_id, info in sorted_instances[:instances_to_remove]:
            instances_to_terminate.append(instance_id)
        
        # Terminate selected instances
        terminated = []
        for instance_id in instances_to_terminate:
            try:
                success = await self._terminate_instance(instance_id)
                if success:
                    terminated.append(instance_id)
                    del self.active_instances[instance_id]
            except Exception as e:
                self.logger.error(f"Failed to terminate instance {instance_id}: {e}")
        
        return terminated
    
    async def _launch_instance(self, resource_spec: CloudResource) -> Optional[str]:
        """Launch a new instance based on provider."""
        if self.provider == CloudProvider.KUBERNETES:
            return await self._launch_k8s_pod(resource_spec)
        elif self.provider == CloudProvider.AWS:
            return await self._launch_aws_instance(resource_spec)
        elif self.provider == CloudProvider.GCP:
            return await self._launch_gcp_instance(resource_spec)
        elif self.provider == CloudProvider.AZURE:
            return await self._launch_azure_instance(resource_spec)
        else:
            return await self._launch_local_worker(resource_spec)
    
    async def _launch_k8s_pod(self, resource_spec: CloudResource) -> Optional[str]:
        """Launch Kubernetes pod."""
        try:
            # Scale deployment
            cmd = [
                'kubectl', 'scale', 'deployment', self.k8s_deployment,
                '--replicas', str(len(self.active_instances) + 1),
                '-n', self.k8s_namespace
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                instance_id = f"k8s-pod-{int(time.time())}"
                self.active_instances[instance_id] = {
                    'provider': 'kubernetes',
                    'resource_spec': asdict(resource_spec),
                    'launch_time': time.time()
                }
                return instance_id
        
        except Exception as e:
            self.logger.error(f"Failed to launch K8s pod: {e}")
        
        return None
    
    async def _launch_aws_instance(self, resource_spec: CloudResource) -> Optional[str]:
        """Launch AWS EC2 instance."""
        # Placeholder for AWS EC2 launch
        instance_id = f"aws-{resource_spec.instance_type}-{int(time.time())}"
        self.active_instances[instance_id] = {
            'provider': 'aws',
            'resource_spec': asdict(resource_spec),
            'launch_time': time.time()
        }
        return instance_id
    
    async def _launch_gcp_instance(self, resource_spec: CloudResource) -> Optional[str]:
        """Launch GCP Compute Engine instance."""
        # Placeholder for GCP instance launch
        instance_id = f"gcp-{resource_spec.instance_type}-{int(time.time())}"
        self.active_instances[instance_id] = {
            'provider': 'gcp',
            'resource_spec': asdict(resource_spec),
            'launch_time': time.time()
        }
        return instance_id
    
    async def _launch_azure_instance(self, resource_spec: CloudResource) -> Optional[str]:
        """Launch Azure VM."""
        # Placeholder for Azure VM launch
        instance_id = f"azure-{resource_spec.instance_type}-{int(time.time())}"
        self.active_instances[instance_id] = {
            'provider': 'azure',
            'resource_spec': asdict(resource_spec),
            'launch_time': time.time()
        }
        return instance_id
    
    async def _launch_local_worker(self, resource_spec: CloudResource) -> Optional[str]:
        """Launch local worker process."""
        # For local testing
        instance_id = f"local-worker-{int(time.time())}"
        self.active_instances[instance_id] = {
            'provider': 'local',
            'resource_spec': asdict(resource_spec),
            'launch_time': time.time()
        }
        return instance_id
    
    async def _terminate_instance(self, instance_id: str) -> bool:
        """Terminate instance based on provider."""
        if instance_id not in self.active_instances:
            return False
        
        provider = self.active_instances[instance_id]['provider']
        
        try:
            if provider == 'kubernetes':
                return await self._terminate_k8s_pod(instance_id)
            elif provider == 'aws':
                return await self._terminate_aws_instance(instance_id)
            elif provider == 'gcp':
                return await self._terminate_gcp_instance(instance_id)
            elif provider == 'azure':
                return await self._terminate_azure_instance(instance_id)
            else:
                return await self._terminate_local_worker(instance_id)
        
        except Exception as e:
            self.logger.error(f"Failed to terminate {instance_id}: {e}")
            return False
    
    async def _terminate_k8s_pod(self, instance_id: str) -> bool:
        """Terminate Kubernetes pod."""
        try:
            cmd = [
                'kubectl', 'scale', 'deployment', self.k8s_deployment,
                '--replicas', str(len(self.active_instances) - 1),
                '-n', self.k8s_namespace
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            await process.communicate()
            return process.returncode == 0
        
        except Exception:
            return False
    
    async def _terminate_aws_instance(self, instance_id: str) -> bool:
        """Terminate AWS instance."""
        # Placeholder
        return True
    
    async def _terminate_gcp_instance(self, instance_id: str) -> bool:
        """Terminate GCP instance."""
        # Placeholder
        return True
    
    async def _terminate_azure_instance(self, instance_id: str) -> bool:
        """Terminate Azure instance."""
        # Placeholder
        return True
    
    async def _terminate_local_worker(self, instance_id: str) -> bool:
        """Terminate local worker."""
        # Placeholder
        return True


class CloudNativeAutoScaler:
    """Comprehensive cloud-native auto-scaler for neural operators."""
    
    def __init__(
        self,
        provider: CloudProvider,
        resource_spec: CloudResource,
        policy: ScalingPolicy,
        strategy: ScalingStrategy = ScalingStrategy.HYBRID
    ):
        self.provider = provider
        self.resource_spec = resource_spec
        self.policy = policy
        self.strategy = strategy
        
        # Components
        self.resource_manager = CloudResourceManager(provider)
        self.workload_predictor = WorkloadPredictor()
        
        # State
        self.metrics_history = deque(maxlen=1000)
        self.scaling_history = deque(maxlen=100)
        self.last_scale_action = 0
        self.current_instances = policy.min_instances
        
        # Monitoring
        self.logger = logging.getLogger(__name__)
        self.monitoring_active = False
        self.monitoring_task = None
        
    async def start_monitoring(self, monitoring_interval: float = 30.0):
        """Start auto-scaling monitoring."""
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(
            self._monitoring_loop(monitoring_interval)
        )
        self.logger.info("Started cloud-native auto-scaling")
    
    async def stop_monitoring(self):
        """Stop auto-scaling monitoring."""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
            try:
                await self.monitoring_task
            except asyncio.CancelledError:
                pass
        self.logger.info("Stopped cloud-native auto-scaling")
    
    async def _monitoring_loop(self, interval: float):
        """Main monitoring and scaling loop."""
        while self.monitoring_active:
            try:
                # Collect metrics
                metrics = await self._collect_workload_metrics()
                if metrics:
                    self.metrics_history.append(metrics)
                    self.workload_predictor.add_metrics(metrics)
                    
                    # Make scaling decision
                    scaling_action = await self._make_scaling_decision(metrics)
                    
                    # Execute scaling if needed
                    if scaling_action:
                        await self._execute_scaling_action(scaling_action)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(interval)
    
    async def _collect_workload_metrics(self) -> Optional[WorkloadMetrics]:
        """Collect current workload metrics."""
        try:
            # In production, this would collect from monitoring systems
            # For now, return mock metrics
            return WorkloadMetrics(
                timestamp=time.time(),
                throughput_samples_per_sec=100.0,
                latency_p50_ms=50.0,
                latency_p95_ms=95.0,
                latency_p99_ms=150.0,
                cpu_utilization=60.0,
                memory_utilization=70.0,
                gpu_utilization=80.0,
                gpu_memory_utilization=75.0,
                error_rate=0.01,
                queue_depth=10,
                active_workers=self.current_instances,
                pending_requests=5
            )
        except Exception as e:
            self.logger.error(f"Failed to collect metrics: {e}")
            return None
    
    async def _make_scaling_decision(self, current_metrics: WorkloadMetrics) -> Optional[Dict[str, Any]]:
        """Make intelligent scaling decision based on strategy."""
        now = time.time()
        
        # Check cooldown
        if now - self.last_scale_action < self.policy.scale_up_cooldown_sec:
            return None
        
        if self.strategy == ScalingStrategy.REACTIVE:
            return await self._reactive_scaling_decision(current_metrics)
        elif self.strategy == ScalingStrategy.PREDICTIVE:
            return await self._predictive_scaling_decision(current_metrics)
        elif self.strategy == ScalingStrategy.HYBRID:
            return await self._hybrid_scaling_decision(current_metrics)
        elif self.strategy == ScalingStrategy.ML_BASED:
            return await self._ml_based_scaling_decision(current_metrics)
        
        return None
    
    async def _reactive_scaling_decision(self, metrics: WorkloadMetrics) -> Optional[Dict[str, Any]]:
        """Reactive scaling based on current metrics."""
        # Scale up conditions
        if (metrics.cpu_utilization > self.policy.scale_up_threshold or
            metrics.gpu_utilization > self.policy.scale_up_threshold or
            metrics.latency_p95_ms > self.policy.target_latency_ms * 1.5):
            
            if self.current_instances < self.policy.max_instances:
                return {
                    'action': 'scale_up',
                    'target_instances': min(
                        self.current_instances + max(1, self.current_instances // 4),
                        self.policy.max_instances
                    ),
                    'reason': 'High resource utilization or latency'
                }
        
        # Scale down conditions
        if (metrics.cpu_utilization < self.policy.scale_down_threshold and
            metrics.gpu_utilization < self.policy.scale_down_threshold and
            metrics.queue_depth < 5):
            
            if self.current_instances > self.policy.min_instances:
                return {
                    'action': 'scale_down',
                    'target_instances': max(
                        self.current_instances - max(1, self.current_instances // 4),
                        self.policy.min_instances
                    ),
                    'reason': 'Low resource utilization'
                }
        
        return None
    
    async def _predictive_scaling_decision(self, metrics: WorkloadMetrics) -> Optional[Dict[str, Any]]:
        """Predictive scaling based on workload forecasting."""
        predictions = self.workload_predictor.predict_workload(horizon_minutes=5)
        
        if not predictions or predictions.get('confidence', 0) < 0.7:
            # Fall back to reactive if prediction is unreliable
            return await self._reactive_scaling_decision(metrics)
        
        predicted_cpu = predictions.get('predicted_cpu_utilization', metrics.cpu_utilization)
        predicted_gpu = predictions.get('predicted_gpu_utilization', metrics.gpu_utilization)
        
        # Proactive scale up
        if (predicted_cpu > self.policy.scale_up_threshold or
            predicted_gpu > self.policy.scale_up_threshold):
            
            if self.current_instances < self.policy.max_instances:
                return {
                    'action': 'scale_up',
                    'target_instances': min(
                        self.current_instances + max(1, self.current_instances // 3),
                        self.policy.max_instances
                    ),
                    'reason': f'Predicted high utilization: CPU={predicted_cpu:.1f}%, GPU={predicted_gpu:.1f}%'
                }
        
        # Proactive scale down
        if (predicted_cpu < self.policy.scale_down_threshold and
            predicted_gpu < self.policy.scale_down_threshold):
            
            if self.current_instances > self.policy.min_instances:
                return {
                    'action': 'scale_down',
                    'target_instances': max(
                        self.current_instances - max(1, self.current_instances // 3),
                        self.policy.min_instances
                    ),
                    'reason': f'Predicted low utilization: CPU={predicted_cpu:.1f}%, GPU={predicted_gpu:.1f}%'
                }
        
        return None
    
    async def _hybrid_scaling_decision(self, metrics: WorkloadMetrics) -> Optional[Dict[str, Any]]:
        """Hybrid approach combining reactive and predictive."""
        # Get both reactive and predictive decisions
        reactive = await self._reactive_scaling_decision(metrics)
        predictive = await self._predictive_scaling_decision(metrics)
        
        # Prioritize reactive for immediate issues
        if reactive and reactive['action'] == 'scale_up':
            return reactive
        
        # Use predictive for proactive scaling
        if predictive:
            return predictive
        
        # Fall back to reactive
        return reactive
    
    async def _ml_based_scaling_decision(self, metrics: WorkloadMetrics) -> Optional[Dict[str, Any]]:
        """ML-based scaling using advanced models."""
        # Placeholder for more sophisticated ML models
        return await self._hybrid_scaling_decision(metrics)
    
    async def _execute_scaling_action(self, action: Dict[str, Any]):
        """Execute the scaling action."""
        try:
            action_type = action['action']
            target_instances = action['target_instances']
            reason = action['reason']
            
            self.logger.info(f"Executing {action_type} to {target_instances} instances: {reason}")
            
            if action_type == 'scale_up':
                new_instances = await self.resource_manager.scale_up(target_instances, self.resource_spec)
                self.logger.info(f"Launched instances: {new_instances}")
            elif action_type == 'scale_down':
                terminated_instances = await self.resource_manager.scale_down(target_instances)
                self.logger.info(f"Terminated instances: {terminated_instances}")
            
            # Update state
            self.current_instances = target_instances
            self.last_scale_action = time.time()
            
            # Record scaling action
            self.scaling_history.append({
                'timestamp': time.time(),
                'action': action_type,
                'from_instances': len(self.resource_manager.active_instances),
                'to_instances': target_instances,
                'reason': reason
            })
            
        except Exception as e:
            self.logger.error(f"Failed to execute scaling action: {e}")
    
    def get_scaling_status(self) -> Dict[str, Any]:
        """Get current scaling status and metrics."""
        return {
            'current_instances': self.current_instances,
            'active_instances': len(self.resource_manager.active_instances),
            'pending_instances': len(self.resource_manager.pending_instances),
            'last_scale_action': self.last_scale_action,
            'recent_metrics': list(self.metrics_history)[-5:] if self.metrics_history else [],
            'recent_scaling': list(self.scaling_history)[-5:] if self.scaling_history else [],
            'provider': self.provider.value,
            'strategy': self.strategy.value
        }