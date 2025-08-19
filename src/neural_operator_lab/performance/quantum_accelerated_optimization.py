"""Quantum-accelerated optimization and auto-scaling for neural operators."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Any, Tuple, Callable
import logging
import time
import psutil
import threading
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from queue import Queue, Empty
import gc
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QuantumOptimizationConfig:
    """Configuration for quantum-accelerated optimization."""
    # Quantum-inspired parameters
    enable_quantum_optimization: bool = True
    quantum_annealing_steps: int = 1000
    quantum_temperature_schedule: str = "exponential"  # "linear", "exponential", "adaptive"
    entanglement_strength: float = 0.1
    
    # Performance optimization
    enable_auto_scaling: bool = True
    target_utilization: float = 0.8
    scale_up_threshold: float = 0.9
    scale_down_threshold: float = 0.3
    min_replicas: int = 1
    max_replicas: int = 8
    
    # Memory optimization
    enable_dynamic_batching: bool = True
    adaptive_precision: bool = True
    gradient_checkpointing: bool = True
    memory_pool_size: int = 1024 * 1024 * 1024  # 1GB
    
    # Distributed optimization
    enable_distributed_optimization: bool = True
    num_optimization_workers: int = mp.cpu_count()
    async_optimization: bool = True
    
    # Caching and prefetching
    enable_intelligent_caching: bool = True
    cache_size_gb: float = 2.0
    prefetch_batches: int = 2
    
    # Auto-tuning
    enable_hyperparameter_tuning: bool = True
    tuning_budget: int = 100  # Number of trials
    parallel_trials: int = 4


class QuantumAnnealingOptimizer:
    """Quantum-inspired annealing optimizer for neural networks."""
    
    def __init__(self, model: nn.Module, config: QuantumOptimizationConfig):
        self.model = model
        self.config = config
        
        # Quantum state variables
        self.quantum_states = {}
        self.energy_history = []
        self.temperature = 1.0
        self.current_step = 0
        
        # Initialize quantum states for each parameter
        self._initialize_quantum_states()
        
    def _initialize_quantum_states(self):
        """Initialize quantum states for model parameters."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                # Create superposition of parameter states
                self.quantum_states[name] = {
                    'amplitude': torch.ones_like(param.data),
                    'phase': torch.zeros_like(param.data),
                    'entanglement_partners': [],
                    'energy_contribution': 0.0
                }
    
    def compute_system_energy(self, loss: torch.Tensor) -> float:
        """Compute total system energy (loss + regularization)."""
        energy = loss.item()
        
        # Add quantum regularization terms
        for name, state in self.quantum_states.items():
            amplitude_penalty = torch.norm(state['amplitude'] - 1.0).item()
            phase_penalty = torch.norm(state['phase']).item() * 0.1
            energy += amplitude_penalty + phase_penalty
        
        return energy
    
    def update_temperature(self):
        """Update temperature according to annealing schedule."""
        progress = self.current_step / self.config.quantum_annealing_steps
        
        if self.config.quantum_temperature_schedule == "linear":
            self.temperature = 1.0 - progress
        elif self.config.quantum_temperature_schedule == "exponential":
            self.temperature = np.exp(-5.0 * progress)
        elif self.config.quantum_temperature_schedule == "adaptive":
            # Adaptive temperature based on energy landscape
            if len(self.energy_history) >= 10:
                recent_variance = np.var(self.energy_history[-10:])
                self.temperature = max(0.01, min(1.0, recent_variance))
        
        self.temperature = max(0.001, self.temperature)  # Minimum temperature
    
    def quantum_step(self, loss: torch.Tensor, optimizer: torch.optim.Optimizer) -> Dict[str, float]:
        """Perform quantum-inspired optimization step."""
        current_energy = self.compute_system_energy(loss)
        self.energy_history.append(current_energy)
        
        # Update temperature
        self.update_temperature()
        
        # Quantum tunneling through local minima
        if self.current_step > 0 and len(self.energy_history) >= 2:
            energy_change = current_energy - self.energy_history[-2]
            
            # Accept moves based on quantum probability
            acceptance_prob = np.exp(-energy_change / (self.temperature + 1e-8))
            
            if energy_change > 0 and np.random.random() < acceptance_prob:
                # Quantum tunneling: accept uphill move
                logger.debug(f"Quantum tunneling accepted with probability {acceptance_prob:.4f}")
                
                # Apply quantum fluctuations to parameters
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Add quantum noise proportional to temperature
                        quantum_noise = torch.randn_like(param.grad) * self.temperature * 0.01
                        param.grad.add_(quantum_noise)
        
        # Update quantum states
        self._update_quantum_states(optimizer)
        
        # Regular optimizer step
        optimizer.step()
        
        self.current_step += 1
        
        return {
            'energy': current_energy,
            'temperature': self.temperature,
            'quantum_step': self.current_step,
            'tunneling_probability': acceptance_prob if 'acceptance_prob' in locals() else 0.0
        }
    
    def _update_quantum_states(self, optimizer: torch.optim.Optimizer):
        """Update quantum states based on gradients."""
        for name, param in self.model.named_parameters():
            if param.requires_grad and param.grad is not None and name in self.quantum_states:
                state = self.quantum_states[name]
                
                # Update amplitude based on gradient magnitude
                grad_norm = torch.norm(param.grad)
                state['amplitude'] *= torch.exp(-grad_norm * 0.01)
                
                # Update phase based on gradient direction
                if grad_norm > 1e-8:
                    phase_update = param.grad / grad_norm * 0.1
                    state['phase'] += phase_update
                
                # Normalize amplitude
                state['amplitude'] = torch.clamp(state['amplitude'], 0.1, 2.0)


class AdaptiveResourceManager:
    """Intelligent resource management and auto-scaling."""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.current_replicas = config.min_replicas
        self.resource_history = []
        self.scaling_events = []
        
        # Resource monitoring
        self.cpu_utilization = 0.0
        self.memory_utilization = 0.0
        self.gpu_utilization = 0.0
        
        # Auto-scaling state
        self.last_scale_time = 0.0
        self.scale_cooldown = 30.0  # seconds
        
        # Start resource monitoring thread
        self.monitoring_active = True
        self.monitor_thread = threading.Thread(target=self._monitor_resources, daemon=True)
        self.monitor_thread.start()
    
    def _monitor_resources(self):
        """Continuously monitor system resources."""
        while self.monitoring_active:
            try:
                # CPU utilization
                self.cpu_utilization = psutil.cpu_percent(interval=1.0)
                
                # Memory utilization
                memory = psutil.virtual_memory()
                self.memory_utilization = memory.percent / 100.0
                
                # GPU utilization (if available)
                if torch.cuda.is_available():
                    self.gpu_utilization = torch.cuda.utilization() / 100.0
                
                # Record metrics
                resource_metrics = {
                    'timestamp': time.time(),
                    'cpu': self.cpu_utilization,
                    'memory': self.memory_utilization,
                    'gpu': self.gpu_utilization,
                    'replicas': self.current_replicas
                }
                
                self.resource_history.append(resource_metrics)
                
                # Keep only recent history
                if len(self.resource_history) > 300:  # 5 minutes at 1s intervals
                    self.resource_history.pop(0)
                
                # Check scaling conditions
                self._check_scaling_conditions()
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(1.0)
    
    def _check_scaling_conditions(self):
        """Check if scaling up or down is needed."""
        if not self.config.enable_auto_scaling:
            return
        
        current_time = time.time()
        if current_time - self.last_scale_time < self.scale_cooldown:
            return
        
        if len(self.resource_history) < 10:
            return
        
        # Average utilization over last 10 seconds
        recent_metrics = self.resource_history[-10:]
        avg_cpu = np.mean([m['cpu'] for m in recent_metrics]) / 100.0
        avg_memory = np.mean([m['memory'] for m in recent_metrics])
        avg_gpu = np.mean([m['gpu'] for m in recent_metrics])
        
        # Overall utilization (weighted average)
        overall_utilization = (avg_cpu * 0.4 + avg_memory * 0.4 + avg_gpu * 0.2)
        
        # Scale up conditions
        if (overall_utilization > self.config.scale_up_threshold and 
            self.current_replicas < self.config.max_replicas):
            self._scale_up()
        
        # Scale down conditions
        elif (overall_utilization < self.config.scale_down_threshold and 
              self.current_replicas > self.config.min_replicas):
            self._scale_down()
    
    def _scale_up(self):
        """Scale up resources."""
        old_replicas = self.current_replicas
        self.current_replicas = min(self.current_replicas + 1, self.config.max_replicas)
        
        self.scaling_events.append({
            'timestamp': time.time(),
            'action': 'scale_up',
            'from_replicas': old_replicas,
            'to_replicas': self.current_replicas,
            'reason': 'high_utilization'
        })
        
        self.last_scale_time = time.time()
        logger.info(f"Scaled up from {old_replicas} to {self.current_replicas} replicas")
    
    def _scale_down(self):
        """Scale down resources."""
        old_replicas = self.current_replicas
        self.current_replicas = max(self.current_replicas - 1, self.config.min_replicas)
        
        self.scaling_events.append({
            'timestamp': time.time(),
            'action': 'scale_down',
            'from_replicas': old_replicas,
            'to_replicas': self.current_replicas,
            'reason': 'low_utilization'
        })
        
        self.last_scale_time = time.time()
        logger.info(f"Scaled down from {old_replicas} to {self.current_replicas} replicas")
    
    def get_optimal_batch_size(self, base_batch_size: int) -> int:
        """Get dynamically adjusted batch size based on resources."""
        if not self.config.enable_dynamic_batching:
            return base_batch_size
        
        # Adjust batch size based on memory utilization
        memory_factor = max(0.5, 1.0 - self.memory_utilization)
        adjusted_batch_size = int(base_batch_size * memory_factor)
        
        # Ensure minimum batch size
        return max(1, adjusted_batch_size)
    
    def cleanup(self):
        """Cleanup resources."""
        self.monitoring_active = False
        if self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=2.0)


class IntelligentCache:
    """Intelligent caching system with LRU and predictive prefetching."""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.cache = {}
        self.access_history = []
        self.cache_stats = {'hits': 0, 'misses': 0}
        self.max_cache_size = int(config.cache_size_gb * 1024 * 1024 * 1024)  # Convert to bytes
        self.current_cache_size = 0
        
        # Access pattern learning
        self.access_patterns = {}
        
    def _compute_cache_key(self, inputs: torch.Tensor, metadata: Dict[str, Any]) -> str:
        """Compute cache key for inputs."""
        # Use tensor hash and metadata
        tensor_hash = hash(tuple(inputs.shape)) ^ hash(inputs.dtype)
        metadata_hash = hash(str(sorted(metadata.items())))
        return f"{tensor_hash}_{metadata_hash}"
    
    def get(self, cache_key: str) -> Optional[torch.Tensor]:
        """Get item from cache."""
        if cache_key in self.cache:
            # Move to end (LRU)
            item = self.cache.pop(cache_key)
            self.cache[cache_key] = item
            self.cache_stats['hits'] += 1
            self._record_access(cache_key)
            return item['data']
        
        self.cache_stats['misses'] += 1
        return None
    
    def put(self, cache_key: str, data: torch.Tensor, metadata: Dict[str, Any]):
        """Put item in cache."""
        item_size = data.numel() * data.element_size()
        
        # Check if we need to evict items
        while (self.current_cache_size + item_size > self.max_cache_size and 
               len(self.cache) > 0):
            self._evict_lru()
        
        # Add new item
        self.cache[cache_key] = {
            'data': data.clone(),
            'metadata': metadata,
            'size': item_size,
            'timestamp': time.time()
        }
        
        self.current_cache_size += item_size
        self._record_access(cache_key)
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if not self.cache:
            return
        
        # Remove first item (least recently used)
        lru_key = next(iter(self.cache))
        removed_item = self.cache.pop(lru_key)
        self.current_cache_size -= removed_item['size']
    
    def _record_access(self, cache_key: str):
        """Record access for pattern learning."""
        self.access_history.append({
            'key': cache_key,
            'timestamp': time.time()
        })
        
        # Keep only recent history
        if len(self.access_history) > 10000:
            self.access_history = self.access_history[-5000:]
    
    def predict_next_access(self) -> List[str]:
        """Predict next likely cache accesses."""
        if len(self.access_history) < 10:
            return []
        
        # Simple pattern recognition: look for sequences
        recent_accesses = [item['key'] for item in self.access_history[-10:]]
        
        # Find common sequences
        predictions = []
        for i in range(len(recent_accesses) - 2):
            sequence = tuple(recent_accesses[i:i+3])
            if sequence in self.access_patterns:
                next_key = self.access_patterns[sequence]
                if next_key not in predictions:
                    predictions.append(next_key)
        
        return predictions[:3]  # Return top 3 predictions
    
    def get_cache_statistics(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        total_requests = self.cache_stats['hits'] + self.cache_stats['misses']
        hit_rate = self.cache_stats['hits'] / total_requests if total_requests > 0 else 0.0
        
        return {
            'hit_rate': hit_rate,
            'total_requests': total_requests,
            'cache_size_mb': self.current_cache_size / (1024 * 1024),
            'cache_items': len(self.cache)
        }


class ParallelOptimizationEngine:
    """Parallel optimization engine using multiple processes."""
    
    def __init__(self, config: QuantumOptimizationConfig):
        self.config = config
        self.num_workers = config.num_optimization_workers
        
        # Work queues
        self.task_queue = mp.Queue()
        self.result_queue = mp.Queue()
        
        # Worker pool
        self.workers = []
        self.active = True
        
        # Start workers
        self._start_workers()
    
    def _start_workers(self):
        """Start optimization worker processes."""
        for i in range(self.num_workers):
            worker = mp.Process(
                target=self._worker_loop,
                args=(self.task_queue, self.result_queue),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.num_workers} optimization workers")
    
    def _worker_loop(self, task_queue: mp.Queue, result_queue: mp.Queue):
        """Worker process loop."""
        while True:
            try:
                task = task_queue.get(timeout=1.0)
                if task is None:  # Poison pill
                    break
                
                # Execute optimization task
                result = self._execute_optimization_task(task)
                result_queue.put(result)
                
            except Empty:
                continue
            except Exception as e:
                result_queue.put({
                    'task_id': task.get('task_id', 'unknown'),
                    'error': str(e),
                    'success': False
                })
    
    def _execute_optimization_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single optimization task."""
        task_type = task['type']
        
        if task_type == 'hyperparameter_search':
            return self._hyperparameter_search_task(task)
        elif task_type == 'gradient_optimization':
            return self._gradient_optimization_task(task)
        elif task_type == 'architecture_search':
            return self._architecture_search_task(task)
        else:
            return {
                'task_id': task['task_id'],
                'error': f'Unknown task type: {task_type}',
                'success': False
            }
    
    def _hyperparameter_search_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute hyperparameter search task."""
        # Simplified hyperparameter search
        params = task['parameter_space']
        
        # Random search
        best_score = float('-inf')
        best_params = {}
        
        for _ in range(task.get('num_trials', 10)):
            # Sample random parameters
            trial_params = {}
            for param_name, param_range in params.items():
                if isinstance(param_range, tuple):
                    trial_params[param_name] = np.random.uniform(param_range[0], param_range[1])
                elif isinstance(param_range, list):
                    trial_params[param_name] = np.random.choice(param_range)
            
            # Simulate evaluation (in real implementation, would train model)
            score = np.random.uniform(0.7, 0.95)  # Mock score
            
            if score > best_score:
                best_score = score
                best_params = trial_params.copy()
        
        return {
            'task_id': task['task_id'],
            'best_params': best_params,
            'best_score': best_score,
            'success': True
        }
    
    def _gradient_optimization_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute gradient optimization task."""
        # Placeholder for advanced gradient optimization
        return {
            'task_id': task['task_id'],
            'optimized_gradients': 'placeholder',
            'success': True
        }
    
    def _architecture_search_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Execute architecture search task."""
        # Placeholder for neural architecture search
        return {
            'task_id': task['task_id'],
            'best_architecture': {'layers': [512, 256, 128]},
            'success': True
        }
    
    def submit_task(self, task: Dict[str, Any]) -> str:
        """Submit optimization task."""
        task_id = f"task_{int(time.time() * 1000000)}"
        task['task_id'] = task_id
        
        self.task_queue.put(task)
        return task_id
    
    def get_result(self, timeout: float = 1.0) -> Optional[Dict[str, Any]]:
        """Get optimization result."""
        try:
            return self.result_queue.get(timeout=timeout)
        except Empty:
            return None
    
    def cleanup(self):
        """Cleanup worker processes."""
        self.active = False
        
        # Send poison pills to workers
        for _ in self.workers:
            self.task_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=2.0)
            if worker.is_alive():
                worker.terminate()
        
        self.workers.clear()


class QuantumAcceleratedOptimizer:
    """Main quantum-accelerated optimization system."""
    
    def __init__(self, model: nn.Module, config: Optional[QuantumOptimizationConfig] = None):
        self.model = model
        self.config = config or QuantumOptimizationConfig()
        
        # Initialize components
        self.quantum_optimizer = QuantumAnnealingOptimizer(model, self.config)
        self.resource_manager = AdaptiveResourceManager(self.config)
        self.intelligent_cache = IntelligentCache(self.config)
        self.parallel_engine = ParallelOptimizationEngine(self.config)
        
        # Optimization state
        self.optimization_history = []
        self.performance_metrics = {}
        
        # Auto-tuning state
        self.tuning_active = False
        self.tuning_results = []
        
    def optimize_training_step(self, batch: Any, loss_fn: Callable, 
                             optimizer: torch.optim.Optimizer) -> Dict[str, Any]:
        """Optimize a single training step with all enhancements."""
        step_start_time = time.time()
        
        # Get optimal batch size
        if hasattr(batch, '__len__'):
            current_batch_size = len(batch[0]) if isinstance(batch, (list, tuple)) else batch.size(0)
            optimal_batch_size = self.resource_manager.get_optimal_batch_size(current_batch_size)
            
            # Adjust batch if necessary
            if optimal_batch_size != current_batch_size:
                batch = self._adjust_batch_size(batch, optimal_batch_size)
        
        # Check cache
        cache_key = self._generate_cache_key(batch)
        cached_result = self.intelligent_cache.get(cache_key)
        
        if cached_result is not None:
            logger.debug("Using cached computation result")
            return {'cached': True, 'result': cached_result}
        
        # Quantum-enhanced optimization step
        outputs = self.model(batch[0])
        loss = loss_fn(outputs, batch[1])
        
        # Quantum annealing step
        quantum_metrics = self.quantum_optimizer.quantum_step(loss, optimizer)
        
        # Cache result
        if self.config.enable_intelligent_caching:
            self.intelligent_cache.put(
                cache_key, outputs.detach(),
                {'loss': loss.item(), 'timestamp': time.time()}
            )
        
        # Collect performance metrics
        step_metrics = {
            'loss': loss.item(),
            'step_time': time.time() - step_start_time,
            'memory_usage': torch.cuda.memory_allocated() / 1e9 if torch.cuda.is_available() else 0,
            **quantum_metrics
        }
        
        self.optimization_history.append(step_metrics)
        
        # Trigger parallel optimizations if needed
        if self.config.enable_distributed_optimization:
            self._schedule_parallel_optimizations(step_metrics)
        
        return step_metrics
    
    def _adjust_batch_size(self, batch: Any, target_size: int) -> Any:
        """Adjust batch size dynamically."""
        if isinstance(batch, (list, tuple)):
            inputs, targets = batch[0], batch[1]
            current_size = inputs.size(0)
            
            if target_size < current_size:
                # Reduce batch size
                return (inputs[:target_size], targets[:target_size])
            elif target_size > current_size:
                # Increase batch size (repeat samples)
                repeat_factor = target_size // current_size + 1
                expanded_inputs = inputs.repeat(repeat_factor, *([1] * (inputs.dim() - 1)))
                expanded_targets = targets.repeat(repeat_factor, *([1] * (targets.dim() - 1)))
                return (expanded_inputs[:target_size], expanded_targets[:target_size])
        
        return batch
    
    def _generate_cache_key(self, batch: Any) -> str:
        """Generate cache key for batch."""
        if isinstance(batch, (list, tuple)):
            inputs = batch[0]
            return f"batch_{hash(tuple(inputs.shape))}_{inputs.dtype}_{inputs.device}"
        return f"batch_{hash(str(batch))}"
    
    def _schedule_parallel_optimizations(self, step_metrics: Dict[str, float]):
        """Schedule parallel optimization tasks."""
        # Example: hyperparameter tuning task
        if (len(self.optimization_history) % 100 == 0 and 
            self.config.enable_hyperparameter_tuning):
            
            task = {
                'type': 'hyperparameter_search',
                'parameter_space': {
                    'learning_rate': (1e-5, 1e-2),
                    'batch_size': [16, 32, 64, 128],
                    'weight_decay': (1e-6, 1e-2)
                },
                'num_trials': 20
            }
            
            task_id = self.parallel_engine.submit_task(task)
            logger.debug(f"Submitted hyperparameter search task: {task_id}")
    
    def start_auto_tuning(self):
        """Start automatic hyperparameter tuning."""
        if self.tuning_active:
            return
        
        self.tuning_active = True
        tuning_thread = threading.Thread(target=self._auto_tuning_loop, daemon=True)
        tuning_thread.start()
        
        logger.info("Started automatic hyperparameter tuning")
    
    def _auto_tuning_loop(self):
        """Automatic tuning loop."""
        while self.tuning_active:
            try:
                # Check for completed tuning results
                result = self.parallel_engine.get_result(timeout=5.0)
                if result and result.get('success', False):
                    self.tuning_results.append(result)
                    logger.info(f"Tuning result: {result.get('best_score', 'unknown')}")
                
                # Submit new tuning tasks periodically
                if len(self.tuning_results) < self.config.tuning_budget:
                    task = {
                        'type': 'hyperparameter_search',
                        'parameter_space': {
                            'quantum_temperature': (0.01, 2.0),
                            'entanglement_strength': (0.001, 1.0),
                            'annealing_steps': [100, 500, 1000, 2000]
                        },
                        'num_trials': 10
                    }
                    self.parallel_engine.submit_task(task)
                
                time.sleep(10.0)  # Check every 10 seconds
                
            except Exception as e:
                logger.error(f"Auto-tuning error: {e}")
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        if not self.optimization_history:
            return {'error': 'No optimization history available'}
        
        recent_metrics = self.optimization_history[-100:] if len(self.optimization_history) >= 100 else self.optimization_history
        
        report = {
            'total_steps': len(self.optimization_history),
            'average_loss': np.mean([m['loss'] for m in recent_metrics]),
            'average_step_time': np.mean([m['step_time'] for m in recent_metrics]),
            'quantum_metrics': {
                'average_temperature': np.mean([m.get('temperature', 0) for m in recent_metrics]),
                'average_energy': np.mean([m.get('energy', 0) for m in recent_metrics]),
            },
            'resource_metrics': {
                'current_replicas': self.resource_manager.current_replicas,
                'scaling_events': len(self.resource_manager.scaling_events),
            },
            'cache_metrics': self.intelligent_cache.get_cache_statistics(),
            'tuning_results': len(self.tuning_results),
        }
        
        return report
    
    def cleanup(self):
        """Cleanup all resources."""
        self.tuning_active = False
        self.resource_manager.cleanup()
        self.parallel_engine.cleanup()
        
        logger.info("Quantum accelerated optimizer cleaned up")


def create_quantum_accelerated_optimizer(
    model: nn.Module,
    enable_quantum_optimization: bool = True,
    enable_auto_scaling: bool = True,
    enable_intelligent_caching: bool = True,
    cache_size_gb: float = 2.0,
    num_optimization_workers: int = None
) -> QuantumAcceleratedOptimizer:
    """Create quantum-accelerated optimizer with specified configuration."""
    config = QuantumOptimizationConfig(
        enable_quantum_optimization=enable_quantum_optimization,
        enable_auto_scaling=enable_auto_scaling,
        enable_intelligent_caching=enable_intelligent_caching,
        cache_size_gb=cache_size_gb,
        num_optimization_workers=num_optimization_workers or mp.cpu_count()
    )
    
    return QuantumAcceleratedOptimizer(model, config)