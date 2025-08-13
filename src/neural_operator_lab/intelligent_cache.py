"""Intelligent caching system with adaptive algorithms and ML-based optimization."""

import time
import threading
import pickle
import hashlib
import weakref
import gzip
import json
import functools
from collections import defaultdict, OrderedDict
from typing import Any, Dict, List, Optional, Callable, Union, Tuple, Set
from dataclasses import dataclass, field
from pathlib import Path
import tempfile
import os
import warnings
from abc import ABC, abstractmethod
import math
import heapq

try:
    from .robust_framework import robust_logger, robust_execution
    from .performance_optimizer import PerformanceMetrics
    _HAS_DEPENDENCIES = True
except ImportError:
    _HAS_DEPENDENCIES = False


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    key: str
    value: Any
    timestamp: float
    access_count: int = 0
    last_access: float = field(default_factory=time.time)
    size_bytes: Optional[int] = None
    ttl: Optional[float] = None
    priority: float = 1.0
    computation_cost: float = 1.0
    
    def __post_init__(self):
        """Calculate size if not provided."""
        if self.size_bytes is None:
            try:
                self.size_bytes = len(pickle.dumps(self.value))
            except Exception:
                self.size_bytes = 1000  # Default estimate
    
    def is_expired(self) -> bool:
        """Check if entry is expired."""
        if self.ttl is None:
            return False
        return time.time() - self.timestamp > self.ttl
    
    def access(self):
        """Record access to this entry."""
        self.access_count += 1
        self.last_access = time.time()
    
    def get_score(self, algorithm: str = "lru") -> float:
        """Get eviction score based on algorithm."""
        current_time = time.time()
        age = current_time - self.timestamp
        recency = current_time - self.last_access
        
        if algorithm == "lru":
            return recency
        elif algorithm == "lfu":
            return -self.access_count
        elif algorithm == "fifo":
            return age
        elif algorithm == "intelligent":
            # Intelligent scoring considers multiple factors
            frequency_score = math.log(self.access_count + 1)
            recency_score = 1.0 / (recency + 1)
            size_penalty = math.log(self.size_bytes + 1) / 1000
            cost_bonus = math.log(self.computation_cost + 1)
            
            return -(frequency_score * recency_score * cost_bonus / size_penalty)
        else:
            return recency


class EvictionStrategy(ABC):
    """Abstract base class for cache eviction strategies."""
    
    @abstractmethod
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select entry to evict."""
        pass
    
    @abstractmethod
    def update_on_access(self, key: str, entry: CacheEntry):
        """Update strategy state when entry is accessed."""
        pass
    
    @abstractmethod
    def update_on_insert(self, key: str, entry: CacheEntry):
        """Update strategy state when entry is inserted."""
        pass


class IntelligentEvictionStrategy(EvictionStrategy):
    """ML-inspired intelligent eviction strategy."""
    
    def __init__(self):
        self.access_patterns = defaultdict(list)
        self.prediction_weights = {
            'frequency': 0.3,
            'recency': 0.25,
            'size': 0.15,
            'cost': 0.2,
            'pattern': 0.1
        }
    
    def select_victim(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select victim using intelligent scoring."""
        if not entries:
            return None
        
        scores = {}
        for key, entry in entries.items():
            scores[key] = self._calculate_eviction_score(entry)
        
        # Return key with highest eviction score (most likely to evict)
        return max(scores.keys(), key=lambda k: scores[k])
    
    def _calculate_eviction_score(self, entry: CacheEntry) -> float:
        """Calculate intelligent eviction score."""
        current_time = time.time()
        
        # Frequency factor (lower is better for eviction)
        frequency_factor = 1.0 / (entry.access_count + 1)
        
        # Recency factor (higher recency = more likely to evict)
        recency_factor = current_time - entry.last_access
        
        # Size factor (larger = more likely to evict)
        size_factor = math.log(entry.size_bytes + 1) / 1000
        
        # Cost factor (higher cost = less likely to evict)
        cost_factor = 1.0 / (entry.computation_cost + 1)
        
        # Pattern factor (based on access patterns)
        pattern_factor = self._predict_future_access(entry.key)
        
        # Combine factors using weights
        score = (
            self.prediction_weights['frequency'] * frequency_factor +
            self.prediction_weights['recency'] * recency_factor +
            self.prediction_weights['size'] * size_factor +
            self.prediction_weights['cost'] * cost_factor +
            self.prediction_weights['pattern'] * pattern_factor
        )
        
        return score
    
    def _predict_future_access(self, key: str) -> float:
        """Predict likelihood of future access based on patterns."""
        access_times = self.access_patterns[key]
        if len(access_times) < 2:
            return 0.5  # Neutral score
        
        # Simple pattern analysis: check for periodic access
        intervals = [access_times[i] - access_times[i-1] for i in range(1, len(access_times))]
        
        if not intervals:
            return 0.5
        
        # Check for regular pattern
        avg_interval = sum(intervals) / len(intervals)
        variance = sum((interval - avg_interval) ** 2 for interval in intervals) / len(intervals)
        
        # Lower variance indicates more predictable pattern
        regularity = 1.0 / (variance + 1.0)
        
        # Predict next access time
        predicted_next_access = access_times[-1] + avg_interval
        time_to_predicted = predicted_next_access - time.time()
        
        # If predicted access is soon, return low eviction likelihood
        if time_to_predicted < avg_interval * 0.5:
            return 0.2 * regularity
        else:
            return 0.8 + 0.2 * (1 - regularity)
    
    def update_on_access(self, key: str, entry: CacheEntry):
        """Update access patterns."""
        self.access_patterns[key].append(time.time())
        
        # Keep only recent accesses (last 100)
        if len(self.access_patterns[key]) > 100:
            self.access_patterns[key] = self.access_patterns[key][-100:]
    
    def update_on_insert(self, key: str, entry: CacheEntry):
        """Update on insertion."""
        if key not in self.access_patterns:
            self.access_patterns[key] = [time.time()]


class AdaptiveCache:
    """Adaptive cache with intelligent eviction and compression."""
    
    def __init__(self, 
                 max_size_mb: float = 100.0,
                 max_entries: int = 10000,
                 compression_threshold: int = 1024,
                 persistence_path: Optional[Path] = None,
                 eviction_strategy: str = "intelligent"):
        
        self.max_size_bytes = int(max_size_mb * 1024 * 1024)
        self.max_entries = max_entries
        self.compression_threshold = compression_threshold
        self.persistence_path = persistence_path
        
        # Cache storage
        self.entries: Dict[str, CacheEntry] = {}
        self.compressed_entries: Set[str] = set()
        self.current_size_bytes = 0
        
        # Eviction strategy
        if eviction_strategy == "intelligent":
            self.eviction_strategy = IntelligentEvictionStrategy()
        else:
            # Fallback to simple LRU
            self.eviction_strategy = IntelligentEvictionStrategy()
        
        # Statistics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.compressions = 0
        self.decompressions = 0
        
        # Thread safety
        self._lock = threading.RLock()
        
        # Background tasks
        self._start_background_tasks()
        
        # Load persisted cache if available
        if self.persistence_path:
            self._load_persistent_cache()
    
    def _start_background_tasks(self):
        """Start background maintenance tasks."""
        # Cleanup thread
        cleanup_thread = threading.Thread(target=self._background_cleanup, daemon=True)
        cleanup_thread.start()
        
        # Statistics thread
        stats_thread = threading.Thread(target=self._background_stats, daemon=True)
        stats_thread.start()
    
    def _background_cleanup(self):
        """Background cleanup of expired entries."""
        while True:
            time.sleep(60)  # Run every minute
            self._cleanup_expired()
            self._adaptive_compression()
    
    def _background_stats(self):
        """Background statistics collection."""
        while True:
            time.sleep(300)  # Every 5 minutes
            if _HAS_DEPENDENCIES:
                stats = self.get_statistics()
                robust_logger.log_info(f"Cache stats: {stats}")
    
    def _cleanup_expired(self):
        """Clean up expired cache entries."""
        with self._lock:
            expired_keys = [
                key for key, entry in self.entries.items()
                if entry.is_expired()
            ]
            
            for key in expired_keys:
                self._remove_entry(key)
    
    def _adaptive_compression(self):
        """Adaptively compress large entries."""
        with self._lock:
            # Find large uncompressed entries
            candidates = [
                (key, entry) for key, entry in self.entries.items()
                if (entry.size_bytes > self.compression_threshold and 
                    key not in self.compressed_entries)
            ]
            
            # Sort by size (largest first)
            candidates.sort(key=lambda x: x[1].size_bytes, reverse=True)
            
            # Compress top candidates
            for key, entry in candidates[:10]:  # Limit to avoid blocking
                self._compress_entry(key)
    
    def _compress_entry(self, key: str):
        """Compress a cache entry."""
        if key not in self.entries or key in self.compressed_entries:
            return
        
        entry = self.entries[key]
        try:
            # Serialize and compress
            serialized = pickle.dumps(entry.value)
            compressed = gzip.compress(serialized)
            
            # Only compress if we save significant space
            if len(compressed) < len(serialized) * 0.8:
                entry.value = compressed
                old_size = entry.size_bytes
                entry.size_bytes = len(compressed)
                
                self.compressed_entries.add(key)
                self.current_size_bytes -= (old_size - entry.size_bytes)
                self.compressions += 1
                
        except Exception as e:
            if _HAS_DEPENDENCIES:
                robust_logger.log_warning(f"Failed to compress cache entry {key}: {e}")
    
    def _decompress_entry(self, key: str) -> Any:
        """Decompress a cache entry."""
        if key not in self.compressed_entries or key not in self.entries:
            return self.entries[key].value
        
        entry = self.entries[key]
        try:
            # Decompress and deserialize
            decompressed = gzip.decompress(entry.value)
            value = pickle.loads(decompressed)
            
            # Update entry
            entry.value = value
            entry.size_bytes = len(pickle.dumps(value))
            
            self.compressed_entries.remove(key)
            self.decompressions += 1
            
            return value
            
        except Exception as e:
            if _HAS_DEPENDENCIES:
                robust_logger.log_warning(f"Failed to decompress cache entry {key}: {e}")
            return entry.value
    
    def _generate_key(self, key_data: Any) -> str:
        """Generate cache key from data."""
        if isinstance(key_data, str):
            return key_data
        
        # Convert to string and hash
        key_str = str(key_data)
        return hashlib.sha256(key_str.encode()).hexdigest()[:16]
    
    def _should_evict(self) -> bool:
        """Check if we should evict entries."""
        return (len(self.entries) >= self.max_entries or 
                self.current_size_bytes >= self.max_size_bytes)
    
    def _evict_entries(self):
        """Evict entries to make space."""
        while self._should_evict() and self.entries:
            victim_key = self.eviction_strategy.select_victim(self.entries)
            if victim_key:
                self._remove_entry(victim_key)
                self.evictions += 1
    
    def _remove_entry(self, key: str):
        """Remove an entry from cache."""
        if key in self.entries:
            entry = self.entries[key]
            self.current_size_bytes -= entry.size_bytes
            del self.entries[key]
            
            if key in self.compressed_entries:
                self.compressed_entries.remove(key)
    
    def get(self, key_data: Any, default: Any = None) -> Any:
        """Get value from cache."""
        key = self._generate_key(key_data)
        
        with self._lock:
            if key in self.entries:
                entry = self.entries[key]
                
                # Check if expired
                if entry.is_expired():
                    self._remove_entry(key)
                    self.misses += 1
                    return default
                
                # Update access info
                entry.access()
                self.eviction_strategy.update_on_access(key, entry)
                self.hits += 1
                
                # Decompress if needed
                if key in self.compressed_entries:
                    return self._decompress_entry(key)
                else:
                    return entry.value
            else:
                self.misses += 1
                return default
    
    def put(self, key_data: Any, value: Any, 
           ttl: Optional[float] = None,
           computation_cost: float = 1.0):
        """Put value in cache."""
        key = self._generate_key(key_data)
        
        with self._lock:
            # Create cache entry
            entry = CacheEntry(
                key=key,
                value=value,
                timestamp=time.time(),
                ttl=ttl,
                computation_cost=computation_cost
            )
            
            # Remove existing entry if it exists
            if key in self.entries:
                self._remove_entry(key)
            
            # Evict if necessary
            self._evict_entries()
            
            # Add new entry
            self.entries[key] = entry
            self.current_size_bytes += entry.size_bytes
            self.eviction_strategy.update_on_insert(key, entry)
    
    def clear(self):
        """Clear all cache entries."""
        with self._lock:
            self.entries.clear()
            self.compressed_entries.clear()
            self.current_size_bytes = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                'entries': len(self.entries),
                'max_entries': self.max_entries,
                'size_mb': self.current_size_bytes / (1024 * 1024),
                'max_size_mb': self.max_size_bytes / (1024 * 1024),
                'hits': self.hits,
                'misses': self.misses,
                'hit_rate': hit_rate,
                'evictions': self.evictions,
                'compressed_entries': len(self.compressed_entries),
                'compressions': self.compressions,
                'decompressions': self.decompressions,
                'utilization': len(self.entries) / self.max_entries
            }
    
    def _load_persistent_cache(self):
        """Load cache from persistent storage."""
        if not self.persistence_path or not self.persistence_path.exists():
            return
        
        try:
            with open(self.persistence_path, 'rb') as f:
                cached_data = pickle.load(f)
                
            for key, entry_data in cached_data.items():
                if not entry_data.get('expired', False):
                    entry = CacheEntry(**entry_data)
                    self.entries[key] = entry
                    self.current_size_bytes += entry.size_bytes
                    
        except Exception as e:
            if _HAS_DEPENDENCIES:
                robust_logger.log_warning(f"Failed to load persistent cache: {e}")
    
    def _save_persistent_cache(self):
        """Save cache to persistent storage."""
        if not self.persistence_path:
            return
        
        try:
            # Prepare data for serialization
            cache_data = {}
            for key, entry in self.entries.items():
                if not entry.is_expired():
                    # Don't save the actual value if compressed
                    if key in self.compressed_entries:
                        # Save compressed data
                        cache_data[key] = {
                            'key': entry.key,
                            'value': entry.value,  # This is compressed
                            'timestamp': entry.timestamp,
                            'access_count': entry.access_count,
                            'last_access': entry.last_access,
                            'size_bytes': entry.size_bytes,
                            'ttl': entry.ttl,
                            'priority': entry.priority,
                            'computation_cost': entry.computation_cost,
                            'compressed': True
                        }
                    else:
                        cache_data[key] = {
                            'key': entry.key,
                            'value': entry.value,
                            'timestamp': entry.timestamp,
                            'access_count': entry.access_count,
                            'last_access': entry.last_access,
                            'size_bytes': entry.size_bytes,
                            'ttl': entry.ttl,
                            'priority': entry.priority,
                            'computation_cost': entry.computation_cost,
                            'compressed': False
                        }
            
            # Save to temporary file first, then move
            temp_path = self.persistence_path.with_suffix('.tmp')
            with open(temp_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            temp_path.rename(self.persistence_path)
            
        except Exception as e:
            if _HAS_DEPENDENCIES:
                robust_logger.log_warning(f"Failed to save persistent cache: {e}")
    
    def __del__(self):
        """Save cache on destruction."""
        if self.persistence_path:
            self._save_persistent_cache()


# Global intelligent cache instance
_global_cache = AdaptiveCache(max_size_mb=50.0, max_entries=5000)


def intelligent_cache(ttl: Optional[float] = None, 
                     computation_cost: float = 1.0,
                     key_func: Optional[Callable] = None):
    """Intelligent caching decorator."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            # Generate cache key
            if key_func:
                cache_key = key_func(*args, **kwargs)
            else:
                cache_key = (func.__name__, args, tuple(sorted(kwargs.items())))
            
            # Check cache
            cached_result = _global_cache.get(cache_key)
            if cached_result is not None:
                return cached_result
            
            # Compute result
            start_time = time.time()
            result = func(*args, **kwargs)
            computation_time = time.time() - start_time
            
            # Store in cache with actual computation cost
            _global_cache.put(
                cache_key, 
                result, 
                ttl=ttl, 
                computation_cost=computation_time
            )
            
            return result
        
        # Add cache management methods
        wrapper.cache_info = _global_cache.get_statistics
        wrapper.cache_clear = _global_cache.clear
        
        return wrapper
    return decorator


def get_cache_statistics() -> Dict[str, Any]:
    """Get global cache statistics."""
    return _global_cache.get_statistics()


def clear_global_cache():
    """Clear global cache."""
    _global_cache.clear()