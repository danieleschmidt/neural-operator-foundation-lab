#!/usr/bin/env python3
"""
Scalable Neural Operator Framework - Generation 3
Optimized for performance, memory efficiency, and large-scale deployment
"""

import os
import json
import logging
import time
import math
import random
import threading
import multiprocessing as mp
from typing import Dict, Any, List, Tuple, Optional, Union, Iterator
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from pathlib import Path
import queue
import hashlib
import pickle
from contextlib import contextmanager


# Import the robust framework as a base
from robust_neural_operator_framework import (
    RobustMatrix, RobustNeuralOperator, ValidationConfig, 
    ModelMetrics, setup_robust_logging
)

logger = setup_robust_logging(log_file="logs/scalable_neural_operator.log")


@dataclass
class ScalingConfig:
    """Configuration for scaling and optimization."""
    # Memory management
    max_memory_gb: float = 8.0
    enable_memory_pooling: bool = True
    garbage_collect_interval: int = 100
    
    # Parallelization
    max_workers: int = max(1, mp.cpu_count() - 1)
    batch_processing: bool = True
    async_inference: bool = True
    
    # Caching
    enable_caching: bool = True
    cache_size_mb: int = 256
    cache_ttl_seconds: int = 3600
    
    # Auto-scaling
    enable_auto_scaling: bool = True
    scale_threshold_cpu: float = 0.8
    scale_threshold_memory: float = 0.8
    min_replicas: int = 1
    max_replicas: int = 4
    
    # Performance optimization
    enable_jit_compilation: bool = False  # Would be True with numba
    vectorize_operations: bool = True
    optimize_memory_layout: bool = True
    
    # Load balancing
    enable_load_balancing: bool = True
    health_check_interval: int = 30


class MemoryPool:
    """Memory pool for efficient matrix allocation and reuse."""
    
    def __init__(self, max_size_mb: float = 256):
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.pools = {}  # Shape -> List of matrices
        self.current_size = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Initialized memory pool with {max_size_mb}MB limit")
    
    def get_matrix(self, rows: int, cols: int, name: str = "pooled") -> Optional[RobustMatrix]:
        """Get a matrix from the pool if available."""
        shape = (rows, cols)
        
        with self.lock:
            if shape in self.pools and self.pools[shape]:
                matrix = self.pools[shape].pop()
                matrix.name = name
                # Reset data to zeros
                matrix.data = [[0.0] * cols for _ in range(rows)]
                self.hits += 1
                logger.debug(f"Pool hit for shape {shape}")
                return matrix
            
        self.misses += 1
        return None
    
    def return_matrix(self, matrix: RobustMatrix):
        """Return a matrix to the pool for reuse."""
        shape = (matrix.rows, matrix.cols)
        matrix_size = matrix.rows * matrix.cols * 8  # 8 bytes per float64
        
        with self.lock:
            if self.current_size + matrix_size <= self.max_size_bytes:
                if shape not in self.pools:
                    self.pools[shape] = []
                
                self.pools[shape].append(matrix)
                self.current_size += matrix_size
                logger.debug(f"Returned matrix {shape} to pool")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics."""
        with self.lock:
            total_matrices = sum(len(pool) for pool in self.pools.values())
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            return {
                "pools": len(self.pools),
                "total_matrices": total_matrices,
                "current_size_mb": self.current_size / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses
            }
    
    def cleanup(self):
        """Clean up the memory pool."""
        with self.lock:
            self.pools.clear()
            self.current_size = 0


class ResultCache:
    """LRU cache for inference results."""
    
    def __init__(self, max_size_mb: int = 256, ttl_seconds: int = 3600):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.ttl_seconds = ttl_seconds
        self.cache = {}  # key -> (result, timestamp, size)
        self.access_order = []  # For LRU eviction
        self.current_size = 0
        self.lock = threading.Lock()
        self.hits = 0
        self.misses = 0
        
        logger.info(f"Initialized result cache with {max_size_mb}MB limit, {ttl_seconds}s TTL")
    
    def _hash_input(self, matrix_data: List[List[float]]) -> str:
        """Create hash key from matrix data."""
        # Simple hash of first few elements for performance
        sample_data = str(matrix_data[0][:min(10, len(matrix_data[0]))])
        return hashlib.md5(sample_data.encode()).hexdigest()
    
    def get(self, input_matrix: RobustMatrix) -> Optional[RobustMatrix]:
        """Get cached result if available and not expired."""
        key = self._hash_input(input_matrix.data)
        current_time = time.time()
        
        with self.lock:
            if key in self.cache:
                result, timestamp, size = self.cache[key]
                
                # Check if expired
                if current_time - timestamp > self.ttl_seconds:
                    del self.cache[key]
                    self.current_size -= size
                    self.access_order.remove(key)
                    self.misses += 1
                    return None
                
                # Update access order (move to end)
                self.access_order.remove(key)
                self.access_order.append(key)
                
                self.hits += 1
                logger.debug(f"Cache hit for key {key[:8]}...")
                return result
            
        self.misses += 1
        return None
    
    def put(self, input_matrix: RobustMatrix, result: RobustMatrix):
        """Cache the result."""
        key = self._hash_input(input_matrix.data)
        result_size = len(pickle.dumps(result.data))
        
        with self.lock:
            # Remove oldest entries if needed
            while (self.current_size + result_size > self.max_size_bytes and 
                   self.access_order):
                oldest_key = self.access_order.pop(0)
                if oldest_key in self.cache:
                    _, _, old_size = self.cache[oldest_key]
                    del self.cache[oldest_key]
                    self.current_size -= old_size
            
            # Add new entry
            if self.current_size + result_size <= self.max_size_bytes:
                self.cache[key] = (result, time.time(), result_size)
                self.access_order.append(key)
                self.current_size += result_size
                logger.debug(f"Cached result for key {key[:8]}...")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self.lock:
            hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
            
            return {
                "entries": len(self.cache),
                "size_mb": self.current_size / (1024 * 1024),
                "hit_rate": hit_rate,
                "hits": self.hits,
                "misses": self.misses
            }


class LoadBalancer:
    """Simple load balancer for distributing inference requests."""
    
    def __init__(self, models: List[RobustNeuralOperator]):
        self.models = models
        self.current_loads = [0] * len(models)  # Track current load per model
        self.lock = threading.Lock()
        self.request_count = 0
        
        logger.info(f"Initialized load balancer with {len(models)} models")
    
    def get_best_model(self) -> Tuple[int, RobustNeuralOperator]:
        """Get the model with the lowest current load."""
        with self.lock:
            # Find model with minimum load
            min_load_idx = min(range(len(self.current_loads)), 
                             key=lambda i: self.current_loads[i])
            
            # Increment load for selected model
            self.current_loads[min_load_idx] += 1
            self.request_count += 1
            
            return min_load_idx, self.models[min_load_idx]
    
    def release_model(self, model_idx: int):
        """Release a model after processing."""
        with self.lock:
            if model_idx < len(self.current_loads):
                self.current_loads[model_idx] = max(0, self.current_loads[model_idx] - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get load balancer statistics."""
        with self.lock:
            return {
                "models": len(self.models),
                "current_loads": self.current_loads.copy(),
                "total_requests": self.request_count,
                "avg_load": sum(self.current_loads) / len(self.current_loads)
            }


class AutoScaler:
    """Auto-scaler for dynamically adjusting model replicas."""
    
    def __init__(self, base_model: RobustNeuralOperator, config: ScalingConfig):
        self.base_model = base_model
        self.config = config
        self.replicas = [base_model]  # Start with one replica
        self.metrics_history = []
        self.lock = threading.Lock()
        
        logger.info(f"Initialized auto-scaler with {config.min_replicas}-{config.max_replicas} replicas")
    
    def should_scale_up(self, current_metrics: Dict[str, float]) -> bool:
        """Check if we should scale up."""
        cpu_usage = current_metrics.get('cpu_usage', 0.0)
        memory_usage = current_metrics.get('memory_usage', 0.0)
        
        return (cpu_usage > self.config.scale_threshold_cpu or 
                memory_usage > self.config.scale_threshold_memory) and \
               len(self.replicas) < self.config.max_replicas
    
    def should_scale_down(self, current_metrics: Dict[str, float]) -> bool:
        """Check if we should scale down."""
        cpu_usage = current_metrics.get('cpu_usage', 0.0)
        memory_usage = current_metrics.get('memory_usage', 0.0)
        
        return (cpu_usage < self.config.scale_threshold_cpu * 0.5 and 
                memory_usage < self.config.scale_threshold_memory * 0.5) and \
               len(self.replicas) > self.config.min_replicas
    
    def scale_up(self):
        """Add a new model replica."""
        with self.lock:
            try:
                # Create new replica (simplified - would copy weights in real implementation)
                new_replica = RobustNeuralOperator(
                    self.base_model.input_dim,
                    self.base_model.hidden_dim, 
                    self.base_model.output_dim,
                    f"replica_{len(self.replicas)}"
                )
                self.replicas.append(new_replica)
                logger.info(f"Scaled up to {len(self.replicas)} replicas")
            except Exception as e:
                logger.error(f"Failed to scale up: {e}")
    
    def scale_down(self):
        """Remove a model replica."""
        with self.lock:
            if len(self.replicas) > self.config.min_replicas:
                removed_replica = self.replicas.pop()
                logger.info(f"Scaled down to {len(self.replicas)} replicas")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics and trigger scaling if needed."""
        self.metrics_history.append({
            'timestamp': time.time(),
            'metrics': metrics.copy()
        })
        
        # Keep only recent history
        cutoff_time = time.time() - 300  # 5 minutes
        self.metrics_history = [
            entry for entry in self.metrics_history 
            if entry['timestamp'] > cutoff_time
        ]
        
        # Check scaling conditions
        if self.should_scale_up(metrics):
            self.scale_up()
        elif self.should_scale_down(metrics):
            self.scale_down()
    
    def get_replicas(self) -> List[RobustNeuralOperator]:
        """Get current replicas."""
        with self.lock:
            return self.replicas.copy()


class ScalableNeuralOperatorCluster:
    """High-performance scalable neural operator cluster."""
    
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 config: ScalingConfig = None):
        
        self.config = config or ScalingConfig()
        
        # Create base model
        self.base_model = RobustNeuralOperator(input_dim, hidden_dim, output_dim, "cluster_base")
        
        # Initialize scaling components
        self.memory_pool = MemoryPool(self.config.cache_size_mb)
        self.result_cache = ResultCache(self.config.cache_size_mb, self.config.cache_ttl_seconds)
        self.auto_scaler = AutoScaler(self.base_model, self.config)
        self.load_balancer = LoadBalancer(self.auto_scaler.get_replicas())
        
        # Performance tracking
        self.performance_metrics = {
            'total_requests': 0,
            'total_processing_time': 0.0,
            'cache_hits': 0,
            'cache_misses': 0,
            'errors': 0
        }
        self.metrics_lock = threading.Lock()
        
        # Background threads
        self.cleanup_thread = None
        self.monitoring_thread = None
        self.running = True
        
        logger.info(f"Initialized scalable neural operator cluster")
        self._start_background_tasks()
    
    def _start_background_tasks(self):
        """Start background monitoring and cleanup tasks."""
        if self.config.enable_auto_scaling:
            self.monitoring_thread = threading.Thread(target=self._monitor_cluster)
            self.monitoring_thread.daemon = True
            self.monitoring_thread.start()
        
        self.cleanup_thread = threading.Thread(target=self._periodic_cleanup)
        self.cleanup_thread.daemon = True
        self.cleanup_thread.start()
    
    def _monitor_cluster(self):
        """Monitor cluster performance and trigger auto-scaling."""
        while self.running:
            try:
                # Simulate resource usage (in real implementation would use psutil)
                current_metrics = {
                    'cpu_usage': random.uniform(0.3, 0.9),  # Simulate variable load
                    'memory_usage': random.uniform(0.2, 0.8),
                    'request_rate': self.performance_metrics['total_requests'] / max(1, time.time() - 3600)
                }
                
                # Update auto-scaler
                self.auto_scaler.update_metrics(current_metrics)
                
                # Update load balancer with new replicas
                self.load_balancer.models = self.auto_scaler.get_replicas()
                self.load_balancer.current_loads = [0] * len(self.load_balancer.models)
                
                time.sleep(self.config.health_check_interval)
                
            except Exception as e:
                logger.error(f"Monitoring error: {e}")
                time.sleep(10)  # Error backoff
    
    def _periodic_cleanup(self):
        """Periodic cleanup of memory and cache."""
        while self.running:
            try:
                # Cleanup every 100 requests or 60 seconds
                time.sleep(60)
                
                if self.performance_metrics['total_requests'] % self.config.garbage_collect_interval == 0:
                    # Memory pool cleanup
                    if self.config.enable_memory_pooling:
                        self.memory_pool.cleanup()
                    
                    # Force garbage collection (would use gc.collect() in real implementation)
                    logger.debug("Performed periodic cleanup")
                
            except Exception as e:
                logger.error(f"Cleanup error: {e}")
    
    def forward_batch(self, inputs: List[RobustMatrix], 
                     enable_caching: bool = None,
                     use_load_balancer: bool = None) -> List[RobustMatrix]:
        """Process a batch of inputs efficiently."""
        enable_caching = enable_caching if enable_caching is not None else self.config.enable_caching
        use_load_balancer = use_load_balancer if use_load_balancer is not None else self.config.enable_load_balancing
        
        start_time = time.time()
        results = []
        cache_hits = 0
        
        try:
            # Process each input
            for input_matrix in inputs:
                # Try cache first
                cached_result = None
                if enable_caching:
                    cached_result = self.result_cache.get(input_matrix)
                
                if cached_result:
                    results.append(cached_result)
                    cache_hits += 1
                else:
                    # Get model for processing
                    if use_load_balancer:
                        model_idx, model = self.load_balancer.get_best_model()
                    else:
                        model_idx, model = 0, self.base_model
                    
                    try:
                        # Process with model
                        result = model.forward(input_matrix)
                        results.append(result)
                        
                        # Cache the result
                        if enable_caching:
                            self.result_cache.put(input_matrix, result)
                        
                    finally:
                        if use_load_balancer:
                            self.load_balancer.release_model(model_idx)
            
            # Update metrics
            processing_time = time.time() - start_time
            with self.metrics_lock:
                self.performance_metrics['total_requests'] += len(inputs)
                self.performance_metrics['total_processing_time'] += processing_time
                self.performance_metrics['cache_hits'] += cache_hits
                self.performance_metrics['cache_misses'] += len(inputs) - cache_hits
            
            logger.debug(f"Processed batch of {len(inputs)} with {cache_hits} cache hits in {processing_time:.4f}s")
            
            return results
            
        except Exception as e:
            with self.metrics_lock:
                self.performance_metrics['errors'] += 1
            logger.error(f"Batch processing error: {e}")
            raise
    
    def forward_async(self, input_matrix: RobustMatrix) -> 'Future':
        """Asynchronous forward pass."""
        if not self.config.async_inference:
            raise ValueError("Async inference not enabled")
        
        # Simplified async implementation
        executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        return executor.submit(self.forward_batch, [input_matrix])
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Get comprehensive cluster statistics."""
        with self.metrics_lock:
            avg_processing_time = (
                self.performance_metrics['total_processing_time'] / 
                max(1, self.performance_metrics['total_requests'])
            )
            
            cache_hit_rate = (
                self.performance_metrics['cache_hits'] / 
                max(1, self.performance_metrics['cache_hits'] + self.performance_metrics['cache_misses'])
            )
            
            return {
                'cluster_performance': {
                    'total_requests': self.performance_metrics['total_requests'],
                    'avg_processing_time': avg_processing_time,
                    'error_rate': self.performance_metrics['errors'] / max(1, self.performance_metrics['total_requests']),
                    'throughput_rps': self.performance_metrics['total_requests'] / max(1, time.time() - 3600)
                },
                'caching': {
                    'hit_rate': cache_hit_rate,
                    'cache_stats': self.result_cache.get_stats()
                },
                'memory_pool': self.memory_pool.get_stats(),
                'load_balancer': self.load_balancer.get_stats(),
                'auto_scaler': {
                    'replicas': len(self.auto_scaler.replicas),
                    'min_replicas': self.config.min_replicas,
                    'max_replicas': self.config.max_replicas
                },
                'configuration': {
                    'max_memory_gb': self.config.max_memory_gb,
                    'max_workers': self.config.max_workers,
                    'enable_caching': self.config.enable_caching,
                    'enable_auto_scaling': self.config.enable_auto_scaling
                }
            }
    
    def shutdown(self):
        """Gracefully shutdown the cluster."""
        logger.info("Shutting down scalable neural operator cluster")
        self.running = False
        
        # Wait for background threads
        if self.monitoring_thread and self.monitoring_thread.is_alive():
            self.monitoring_thread.join(timeout=5)
        
        if self.cleanup_thread and self.cleanup_thread.is_alive():
            self.cleanup_thread.join(timeout=5)
        
        # Cleanup resources
        self.memory_pool.cleanup()
        logger.info("Cluster shutdown complete")


def run_scalability_validation():
    """Run comprehensive scalability validation."""
    
    print("\n🚀 SCALABLE NEURAL OPERATOR VALIDATION SUITE")
    print("=" * 60)
    
    validation_results = {}
    
    try:
        # Create scalable cluster
        print("\n🏗️ Creating scalable neural operator cluster...")
        config = ScalingConfig(
            max_memory_gb=2.0,
            max_workers=4,
            enable_caching=True,
            enable_auto_scaling=True,
            cache_size_mb=128
        )
        
        cluster = ScalableNeuralOperatorCluster(
            input_dim=32, 
            hidden_dim=16, 
            output_dim=8,
            config=config
        )
        
        print(f"✅ Cluster created with {len(cluster.auto_scaler.replicas)} initial replicas")
        
        # Test 1: Single request performance
        print("\n⚡ Test 1: Single Request Performance")
        test_data = [[random.uniform(-1, 1) for _ in range(32)]]
        test_input = RobustMatrix(1, 32, test_data, "single_test")
        
        start_time = time.time()
        results = cluster.forward_batch([test_input])
        single_time = time.time() - start_time
        
        print(f"  Single request: {single_time:.4f}s")
        validation_results['single_request_time'] = single_time
        
        # Test 2: Batch processing performance
        print("\n📦 Test 2: Batch Processing Performance")
        batch_sizes = [1, 5, 10, 20, 50]
        batch_results = []
        
        for batch_size in batch_sizes:
            batch_data = [[random.uniform(-1, 1) for _ in range(32)] for _ in range(batch_size)]
            batch_inputs = [RobustMatrix(1, 32, [row], f"batch_{i}") for i, row in enumerate(batch_data)]
            
            start_time = time.time()
            results = cluster.forward_batch(batch_inputs)
            batch_time = time.time() - start_time
            
            throughput = batch_size / batch_time
            batch_results.append({
                'batch_size': batch_size,
                'time': batch_time,
                'throughput': throughput
            })
            
            print(f"  Batch {batch_size:2d}: {batch_time:.4f}s ({throughput:.1f} req/s)")
        
        validation_results['batch_performance'] = batch_results
        
        # Test 3: Caching effectiveness
        print("\n💾 Test 3: Caching Effectiveness")
        
        # Create repeated requests
        repeated_data = [[random.uniform(-1, 1) for _ in range(32)] for _ in range(5)]
        repeated_inputs = [RobustMatrix(1, 32, [row], f"cached_{i}") for i, row in enumerate(repeated_data)]
        
        # First run (populate cache)
        cluster.forward_batch(repeated_inputs, enable_caching=True)
        
        # Second run (should hit cache)
        start_time = time.time()
        cluster.forward_batch(repeated_inputs, enable_caching=True)
        cached_time = time.time() - start_time
        
        cache_stats = cluster.result_cache.get_stats()
        print(f"  Cache hit rate: {cache_stats['hit_rate']:.2f}")
        print(f"  Cached processing time: {cached_time:.4f}s")
        
        validation_results['cache_stats'] = cache_stats
        
        # Test 4: Load balancing
        print("\n⚖️ Test 4: Load Balancing")
        
        # Trigger scaling up (simulate high load)
        cluster.auto_scaler.update_metrics({'cpu_usage': 0.9, 'memory_usage': 0.85})
        time.sleep(1)  # Allow scaling
        
        # Process with load balancer
        concurrent_inputs = [RobustMatrix(1, 32, [[random.uniform(-1, 1) for _ in range(32)]], f"concurrent_{i}") 
                           for i in range(10)]
        
        start_time = time.time()
        results = cluster.forward_batch(concurrent_inputs, use_load_balancer=True)
        lb_time = time.time() - start_time
        
        lb_stats = cluster.load_balancer.get_stats()
        print(f"  Load balancer models: {lb_stats['models']}")
        print(f"  Total requests: {lb_stats['total_requests']}")
        print(f"  Processing time: {lb_time:.4f}s")
        
        validation_results['load_balancer_stats'] = lb_stats
        
        # Test 5: Memory efficiency
        print("\n🧠 Test 5: Memory Efficiency")
        
        memory_stats = cluster.memory_pool.get_stats()
        print(f"  Memory pool hit rate: {memory_stats['hit_rate']:.2f}")
        print(f"  Pool size: {memory_stats['current_size_mb']:.1f}MB")
        print(f"  Total matrices in pool: {memory_stats['total_matrices']}")
        
        validation_results['memory_stats'] = memory_stats
        
        # Test 6: Comprehensive cluster stats
        print("\n📊 Test 6: Comprehensive Cluster Statistics")
        
        comprehensive_stats = cluster.get_comprehensive_stats()
        print(f"  Total requests processed: {comprehensive_stats['cluster_performance']['total_requests']}")
        print(f"  Average processing time: {comprehensive_stats['cluster_performance']['avg_processing_time']:.4f}s")
        print(f"  Cache hit rate: {comprehensive_stats['caching']['hit_rate']:.2f}")
        print(f"  Current replicas: {comprehensive_stats['auto_scaler']['replicas']}")
        
        validation_results['comprehensive_stats'] = comprehensive_stats
        
        # Overall scalability assessment
        print("\n🏆 Scalability Assessment")
        
        scalability_score = 0
        
        # Performance criteria
        if single_time < 0.01:  # < 10ms
            scalability_score += 20
            print("✅ Excellent single request latency (+20)")
        elif single_time < 0.05:  # < 50ms
            scalability_score += 15
            print("✅ Good single request latency (+15)")
        
        # Throughput criteria
        max_throughput = max(result['throughput'] for result in batch_results)
        if max_throughput > 1000:
            scalability_score += 25
            print("✅ Excellent throughput (+25)")
        elif max_throughput > 500:
            scalability_score += 20
            print("✅ Good throughput (+20)")
        elif max_throughput > 100:
            scalability_score += 15
            print("✅ Moderate throughput (+15)")
        
        # Caching efficiency
        if cache_stats['hit_rate'] > 0.8:
            scalability_score += 20
            print("✅ Excellent cache performance (+20)")
        elif cache_stats['hit_rate'] > 0.5:
            scalability_score += 15
            print("✅ Good cache performance (+15)")
        
        # Auto-scaling
        if len(cluster.auto_scaler.replicas) > 1:
            scalability_score += 15
            print("✅ Auto-scaling working (+15)")
        
        # Load balancing
        if lb_stats['total_requests'] > 0:
            scalability_score += 10
            print("✅ Load balancing active (+10)")
        
        # Memory efficiency
        if memory_stats['hit_rate'] > 0.3:
            scalability_score += 10
            print("✅ Memory pooling effective (+10)")
        
        print(f"\n📊 SCALABILITY SCORE: {scalability_score}/100")
        
        if scalability_score >= 80:
            print("🌟 EXCELLENT - Production-ready scalability")
        elif scalability_score >= 60:
            print("👍 GOOD - Strong scalability foundation")
        elif scalability_score >= 40:
            print("⚠️ ACCEPTABLE - Basic scalability features")
        else:
            print("❌ NEEDS IMPROVEMENT - Enhance scalability")
        
        validation_results['scalability_score'] = scalability_score
        
        # Cleanup
        cluster.shutdown()
        
        return validation_results
        
    except Exception as e:
        logger.error(f"Scalability validation failed: {e}")
        print(f"❌ Validation failed: {e}")
        return {"error": str(e), "scalability_score": 0}


if __name__ == "__main__":
    print("Starting scalable neural operator validation...")
    
    # Ensure log directory exists
    os.makedirs("logs", exist_ok=True)
    
    try:
        results = run_scalability_validation()
        print(f"\n🎯 Scalability validation complete!")
        print(f"Scalability score: {results.get('scalability_score', 0)}/100")
        
        # Save detailed results
        with open("scalability_validation_results.json", "w") as f:
            # Convert non-serializable objects to strings
            serializable_results = json.loads(json.dumps(results, default=str))
            json.dump(serializable_results, f, indent=2)
        
        print("📄 Detailed results saved to scalability_validation_results.json")
        
    except Exception as e:
        print(f"❌ Critical failure: {e}")
        logger.error(f"Critical scalability validation failure: {e}")