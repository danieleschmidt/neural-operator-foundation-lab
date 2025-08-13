"""Advanced performance optimization system for neural operators."""

import time
import threading
import functools
import pickle
import hashlib
from collections import OrderedDict, defaultdict
from typing import Any, Dict, List, Optional, Callable, Union, Tuple
from dataclasses import dataclass, field
from contextlib import contextmanager
import warnings
import os
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
import queue
import weakref
import gc

try:
    from .robust_framework import robust_logger, robust_execution, ResourceMonitor
    _HAS_ROBUST_FRAMEWORK = True
except ImportError:
    _HAS_ROBUST_FRAMEWORK = False


@dataclass
class PerformanceMetrics:
    """Performance metrics tracking."""
    operation_name: str
    execution_time: float
    memory_usage: Optional[float] = None
    cache_hit_rate: Optional[float] = None
    throughput: Optional[float] = None
    cpu_usage: Optional[float] = None
    timestamp: float = field(default_factory=time.time)
    
    def __post_init__(self):
        """Validate metrics."""
        if self.execution_time < 0:
            raise ValueError("Execution time cannot be negative")


class AdvancedCache:
    """High-performance caching system with LRU eviction and statistics."""
    
    def __init__(self, maxsize: int = 1000, ttl: Optional[float] = None):
        self.maxsize = maxsize
        self.ttl = ttl
        self.cache = OrderedDict()
        self.access_times = {}
        self.hit_count = 0
        self.miss_count = 0
        self.eviction_count = 0
        self._lock = threading.RLock()
        
        # Background cleanup for TTL
        if ttl:
            self._cleanup_thread = threading.Thread(target=self._cleanup_expired, daemon=True)
            self._cleanup_thread.start()
    
    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        key_data = (args, tuple(sorted(kwargs.items())))
        key_str = str(key_data)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def _cleanup_expired(self):
        """Background thread to clean up expired entries."""
        while True:
            time.sleep(60)  # Check every minute
            if self.ttl:
                current_time = time.time()
                with self._lock:
                    expired_keys = [
                        key for key, access_time in self.access_times.items()
                        if current_time - access_time > self.ttl
                    ]
                    for key in expired_keys:
                        self._evict_key(key)
    
    def _evict_key(self, key: str):
        """Evict a specific key from cache."""
        if key in self.cache:
            del self.cache[key]
            del self.access_times[key]
            self.eviction_count += 1
    
    def _evict_lru(self):
        """Evict least recently used item."""
        if self.cache:
            lru_key = next(iter(self.cache))
            self._evict_key(lru_key)
    
    def get(self, key: str) -> Tuple[bool, Any]:
        """Get item from cache."""
        with self._lock:
            if key in self.cache:
                # Move to end (most recently used)
                value = self.cache.pop(key)
                self.cache[key] = value
                self.access_times[key] = time.time()
                self.hit_count += 1
                return True, value
            else:
                self.miss_count += 1
                return False, None
    
    def put(self, key: str, value: Any):
        """Put item in cache."""
        with self._lock:
            # Check if we need to evict
            if len(self.cache) >= self.maxsize and key not in self.cache:
                self._evict_lru()
            
            self.cache[key] = value
            self.access_times[key] = time.time()
            
            # Move to end if already exists
            if key in self.cache:
                self.cache.move_to_end(key)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.cache.clear()
            self.access_times.clear()
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hit_count + self.miss_count
            hit_rate = self.hit_count / total_requests if total_requests > 0 else 0
            
            return {
                'size': len(self.cache),
                'maxsize': self.maxsize,
                'hits': self.hit_count,
                'misses': self.miss_count,
                'hit_rate': hit_rate,
                'evictions': self.eviction_count,
                'ttl': self.ttl
            }


def cached(maxsize: int = 128, ttl: Optional[float] = None, 
          key_func: Optional[Callable] = None):
    """Advanced caching decorator."""
    def decorator(func: Callable) -> Callable:
        cache = AdvancedCache(maxsize=maxsize, ttl=ttl)
        
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = cache._generate_key(*args, **kwargs)
            
            # Check cache
            hit, value = cache.get(cache_key)
            if hit:
                return value
            
            # Compute value
            result = func(*args, **kwargs)
            
            # Store in cache
            cache.put(cache_key, result)
            
            return result
        
        # Add cache management methods
        wrapper.cache_info = cache.get_stats
        wrapper.cache_clear = cache.clear
        
        return wrapper
    return decorator


class PerformanceProfiler:
    """Advanced performance profiling and optimization."""
    
    def __init__(self):
        self.metrics = []
        self.operation_stats = defaultdict(list)
        self.bottlenecks = []
        self._lock = threading.Lock()
    
    @contextmanager
    def profile(self, operation_name: str):
        """Profile an operation."""
        start_time = time.time()
        start_memory = self._get_memory_usage()
        
        try:
            yield
        finally:
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            execution_time = end_time - start_time
            memory_delta = end_memory - start_memory if end_memory and start_memory else None
            
            metrics = PerformanceMetrics(
                operation_name=operation_name,
                execution_time=execution_time,
                memory_usage=memory_delta
            )
            
            self.record_metrics(metrics)
    
    def _get_memory_usage(self) -> Optional[float]:
        """Get current memory usage in MB."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024  # MB
        except ImportError:
            return None
    
    def record_metrics(self, metrics: PerformanceMetrics):
        """Record performance metrics."""
        with self._lock:
            self.metrics.append(metrics)
            self.operation_stats[metrics.operation_name].append(metrics)
            
            # Keep only recent metrics (last 10000)
            if len(self.metrics) > 10000:
                self.metrics = self.metrics[-5000:]
    
    def get_operation_stats(self, operation_name: str) -> Dict[str, Any]:
        """Get statistics for a specific operation."""
        with self._lock:
            operation_metrics = self.operation_stats[operation_name]
            if not operation_metrics:
                return {}
            
            execution_times = [m.execution_time for m in operation_metrics]
            memory_usages = [m.memory_usage for m in operation_metrics if m.memory_usage is not None]
            
            stats = {
                'count': len(operation_metrics),
                'avg_execution_time': sum(execution_times) / len(execution_times),
                'min_execution_time': min(execution_times),
                'max_execution_time': max(execution_times),
                'total_execution_time': sum(execution_times)
            }
            
            if memory_usages:
                stats.update({
                    'avg_memory_usage': sum(memory_usages) / len(memory_usages),
                    'max_memory_usage': max(memory_usages),
                    'total_memory_usage': sum(memory_usages)
                })
            
            return stats
    
    def identify_bottlenecks(self, threshold_seconds: float = 1.0) -> List[Dict[str, Any]]:
        """Identify performance bottlenecks."""
        bottlenecks = []
        
        for operation_name, metrics_list in self.operation_stats.items():
            stats = self.get_operation_stats(operation_name)
            
            if stats.get('avg_execution_time', 0) > threshold_seconds:
                bottlenecks.append({
                    'operation': operation_name,
                    'avg_time': stats['avg_execution_time'],
                    'max_time': stats['max_execution_time'],
                    'call_count': stats['count'],
                    'total_time': stats['total_execution_time'],
                    'severity': 'high' if stats['avg_execution_time'] > threshold_seconds * 5 else 'medium'
                })
        
        return sorted(bottlenecks, key=lambda x: x['total_time'], reverse=True)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        with self._lock:
            if not self.metrics:
                return {'status': 'no_data'}
            
            total_operations = len(self.metrics)
            unique_operations = len(self.operation_stats)
            
            all_times = [m.execution_time for m in self.metrics]
            total_time = sum(all_times)
            avg_time = total_time / len(all_times)
            
            return {
                'total_operations': total_operations,
                'unique_operations': unique_operations,
                'total_execution_time': total_time,
                'average_execution_time': avg_time,
                'bottlenecks': self.identify_bottlenecks(),
                'top_operations': self._get_top_operations()
            }
    
    def _get_top_operations(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get top operations by total execution time."""
        operation_totals = []
        
        for operation_name in self.operation_stats:
            stats = self.get_operation_stats(operation_name)
            operation_totals.append({
                'operation': operation_name,
                'total_time': stats['total_execution_time'],
                'call_count': stats['count'],
                'avg_time': stats['avg_execution_time']
            })
        
        return sorted(operation_totals, key=lambda x: x['total_time'], reverse=True)[:limit]


class ParallelProcessor:
    """Advanced parallel processing with automatic scaling."""
    
    def __init__(self, max_workers: Optional[int] = None, 
                 use_processes: bool = False):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.use_processes = use_processes
        self.active_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0
        self._lock = threading.Lock()
        
        # Adaptive scaling
        self.performance_history = []
        self.optimal_workers = self.max_workers // 2
    
    def _get_executor(self, num_workers: int):
        """Get appropriate executor."""
        if self.use_processes:
            return ProcessPoolExecutor(max_workers=num_workers)
        else:
            return ThreadPoolExecutor(max_workers=num_workers)
    
    def parallel_map(self, func: Callable, items: List[Any], 
                    chunk_size: Optional[int] = None) -> List[Any]:
        """Parallel map with automatic optimization."""
        if not items:
            return []
        
        start_time = time.time()
        
        # Determine optimal chunk size
        if chunk_size is None:
            chunk_size = max(1, len(items) // (self.optimal_workers * 4))
        
        # Split items into chunks
        chunks = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
        
        def process_chunk(chunk):
            return [func(item) for item in chunk]
        
        # Process chunks in parallel
        results = []
        with self._get_executor(self.optimal_workers) as executor:
            future_to_chunk = {executor.submit(process_chunk, chunk): chunk 
                             for chunk in chunks}
            
            for future in as_completed(future_to_chunk):
                try:
                    chunk_results = future.result()
                    results.extend(chunk_results)
                    
                    with self._lock:
                        self.completed_tasks += len(chunk_results)
                        
                except Exception as e:
                    with self._lock:
                        self.failed_tasks += len(future_to_chunk[future])
                    
                    if _HAS_ROBUST_FRAMEWORK:
                        robust_logger.log_warning(f"Parallel processing chunk failed: {e}")
        
        # Record performance and adapt
        execution_time = time.time() - start_time
        throughput = len(items) / execution_time if execution_time > 0 else 0
        
        self.performance_history.append({
            'workers': self.optimal_workers,
            'throughput': throughput,
            'items': len(items),
            'time': execution_time
        })
        
        # Adapt worker count based on performance
        self._adapt_worker_count()
        
        return results
    
    def _adapt_worker_count(self):
        """Adapt worker count based on performance history."""
        if len(self.performance_history) < 3:
            return
        
        recent_performance = self.performance_history[-3:]
        avg_throughput = sum(p['throughput'] for p in recent_performance) / len(recent_performance)
        
        # If performance is declining, try different worker count
        if len(self.performance_history) >= 6:
            older_performance = self.performance_history[-6:-3]
            old_avg_throughput = sum(p['throughput'] for p in older_performance) / len(older_performance)
            
            if avg_throughput < old_avg_throughput * 0.95:  # 5% decline
                # Try adjusting worker count
                if self.optimal_workers < self.max_workers:
                    self.optimal_workers = min(self.max_workers, self.optimal_workers + 1)
                elif self.optimal_workers > 1:
                    self.optimal_workers = max(1, self.optimal_workers - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get parallel processing statistics."""
        with self._lock:
            total_tasks = self.completed_tasks + self.failed_tasks
            success_rate = self.completed_tasks / total_tasks if total_tasks > 0 else 0
            
            recent_throughput = 0
            if self.performance_history:
                recent_performance = self.performance_history[-5:]
                recent_throughput = sum(p['throughput'] for p in recent_performance) / len(recent_performance)
            
            return {
                'completed_tasks': self.completed_tasks,
                'failed_tasks': self.failed_tasks,
                'success_rate': success_rate,
                'optimal_workers': self.optimal_workers,
                'max_workers': self.max_workers,
                'recent_throughput': recent_throughput,
                'use_processes': self.use_processes
            }


class AutoScaler:
    """Automatic resource scaling based on load and performance."""
    
    def __init__(self, min_resources: int = 1, max_resources: int = 16):
        self.min_resources = min_resources
        self.max_resources = max_resources
        self.current_resources = min_resources
        self.load_history = []
        self.performance_history = []
        self.scaling_cooldown = 30  # seconds
        self.last_scale_time = 0
        self._lock = threading.Lock()
    
    def record_load(self, current_load: float, max_capacity: float):
        """Record current system load."""
        load_ratio = current_load / max_capacity if max_capacity > 0 else 0
        
        with self._lock:
            self.load_history.append({
                'timestamp': time.time(),
                'load_ratio': load_ratio,
                'current_load': current_load,
                'max_capacity': max_capacity
            })
            
            # Keep only recent history (last hour)
            cutoff_time = time.time() - 3600
            self.load_history = [
                entry for entry in self.load_history 
                if entry['timestamp'] > cutoff_time
            ]
    
    def should_scale_up(self) -> bool:
        """Determine if we should scale up resources."""
        if not self.load_history or self.current_resources >= self.max_resources:
            return False
        
        # Check if enough time has passed since last scaling
        if time.time() - self.last_scale_time < self.scaling_cooldown:
            return False
        
        # Check recent load average
        recent_loads = [entry['load_ratio'] for entry in self.load_history[-10:]]
        avg_load = sum(recent_loads) / len(recent_loads)
        
        # Scale up if consistently high load
        return avg_load > 0.75
    
    def should_scale_down(self) -> bool:
        """Determine if we should scale down resources."""
        if not self.load_history or self.current_resources <= self.min_resources:
            return False
        
        # Check if enough time has passed since last scaling
        if time.time() - self.last_scale_time < self.scaling_cooldown * 2:  # Longer cooldown for scaling down
            return False
        
        # Check recent load average
        recent_loads = [entry['load_ratio'] for entry in self.load_history[-20:]]
        avg_load = sum(recent_loads) / len(recent_loads)
        
        # Scale down if consistently low load
        return avg_load < 0.25
    
    def scale_up(self) -> int:
        """Scale up resources."""
        with self._lock:
            if self.should_scale_up():
                old_resources = self.current_resources
                self.current_resources = min(self.max_resources, self.current_resources * 2)
                self.last_scale_time = time.time()
                
                if _HAS_ROBUST_FRAMEWORK:
                    robust_logger.log_info(
                        f"Scaled up from {old_resources} to {self.current_resources} resources"
                    )
                
                return self.current_resources
        
        return self.current_resources
    
    def scale_down(self) -> int:
        """Scale down resources."""
        with self._lock:
            if self.should_scale_down():
                old_resources = self.current_resources
                self.current_resources = max(self.min_resources, self.current_resources // 2)
                self.last_scale_time = time.time()
                
                if _HAS_ROBUST_FRAMEWORK:
                    robust_logger.log_info(
                        f"Scaled down from {old_resources} to {self.current_resources} resources"
                    )
                
                return self.current_resources
        
        return self.current_resources
    
    def auto_scale(self) -> int:
        """Automatically scale based on current conditions."""
        if self.should_scale_up():
            return self.scale_up()
        elif self.should_scale_down():
            return self.scale_down()
        else:
            return self.current_resources
    
    def get_scaling_recommendation(self) -> Dict[str, Any]:
        """Get scaling recommendation without actually scaling."""
        if not self.load_history:
            return {'recommendation': 'no_data', 'current_resources': self.current_resources}
        
        recent_loads = [entry['load_ratio'] for entry in self.load_history[-10:]]
        avg_load = sum(recent_loads) / len(recent_loads)
        
        recommendation = 'maintain'
        if avg_load > 0.8:
            recommendation = 'scale_up'
        elif avg_load < 0.2:
            recommendation = 'scale_down'
        
        return {
            'recommendation': recommendation,
            'current_resources': self.current_resources,
            'avg_load': avg_load,
            'can_scale_up': self.current_resources < self.max_resources,
            'can_scale_down': self.current_resources > self.min_resources,
            'cooldown_remaining': max(0, self.scaling_cooldown - (time.time() - self.last_scale_time))
        }


# Global instances
_global_profiler = PerformanceProfiler()
_global_parallel_processor = ParallelProcessor()
_global_autoscaler = AutoScaler()


def profile(operation_name: str):
    """Decorator for profiling function performance."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            with _global_profiler.profile(operation_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def get_performance_stats() -> Dict[str, Any]:
    """Get global performance statistics."""
    return {
        'profiler': _global_profiler.get_summary(),
        'parallel_processor': _global_parallel_processor.get_stats(),
        'autoscaler': _global_autoscaler.get_scaling_recommendation()
    }


def optimize_parallel_execution(func: Callable, items: List[Any], **kwargs) -> List[Any]:
    """Optimized parallel execution with automatic scaling."""
    return _global_parallel_processor.parallel_map(func, items, **kwargs)


def record_system_load(current_load: float, max_capacity: float):
    """Record system load for autoscaling."""
    _global_autoscaler.record_load(current_load, max_capacity)


def auto_scale_resources() -> int:
    """Trigger automatic resource scaling."""
    return _global_autoscaler.auto_scale()