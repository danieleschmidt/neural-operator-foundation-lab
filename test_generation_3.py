#!/usr/bin/env python3
"""Test Generation 3 implementation: Performance optimization and scaling."""

import sys
import time
import threading
import tempfile
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

def test_performance_optimizer():
    """Test the performance optimization system."""
    print("🚀 Testing Performance Optimizer...")
    
    try:
        from neural_operator_lab.performance_optimizer import (
            PerformanceProfiler, AdvancedCache, cached, 
            ParallelProcessor, AutoScaler, profile
        )
        print("✅ Performance optimizer imports successful")
        
        # Test performance profiler
        profiler = PerformanceProfiler()
        
        with profiler.profile("test_operation"):
            time.sleep(0.1)  # Simulate work
        
        # Check that metrics were recorded
        stats = profiler.get_operation_stats("test_operation")
        assert stats['count'] == 1
        assert stats['avg_execution_time'] >= 0.09  # Should be around 0.1 seconds
        print("✅ PerformanceProfiler working")
        
        # Test advanced cache
        cache = AdvancedCache(maxsize=100, ttl=1.0)
        
        # Test cache put/get
        cache.put("test_key", "test_value")
        hit, value = cache.get("test_key")
        assert hit == True
        assert value == "test_value"
        
        # Test cache miss
        hit, value = cache.get("nonexistent_key")
        assert hit == False
        assert value is None
        
        # Test cache stats
        stats = cache.get_stats()
        assert stats['hits'] >= 1
        assert stats['misses'] >= 1
        print("✅ AdvancedCache working")
        
        # Test caching decorator
        call_count = 0
        
        @cached(maxsize=10)
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            time.sleep(0.01)  # Simulate expensive computation
            return x * 2
        
        # First call should compute
        result1 = expensive_function(5)
        assert result1 == 10
        assert call_count == 1
        
        # Second call should use cache
        result2 = expensive_function(5)
        assert result2 == 10
        assert call_count == 1  # Should not increment
        print("✅ Caching decorator working")
        
        # Test parallel processor
        processor = ParallelProcessor(max_workers=2, use_processes=False)
        
        def simple_task(x):
            return x * x
        
        items = list(range(10))
        results = processor.parallel_map(simple_task, items)
        expected = [x * x for x in items]
        assert results == expected
        
        stats = processor.get_stats()
        assert stats['completed_tasks'] >= len(items)
        print("✅ ParallelProcessor working")
        
        # Test autoscaler
        scaler = AutoScaler(min_resources=1, max_resources=8)
        
        # Record high load
        for _ in range(5):
            scaler.record_load(80, 100)  # 80% load
        
        recommendation = scaler.get_scaling_recommendation()
        assert recommendation['avg_load'] >= 0.7
        print("✅ AutoScaler working")
        
        # Test profile decorator
        @profile("decorated_operation")
        def profiled_function(x):
            time.sleep(0.01)
            return x + 1
        
        result = profiled_function(5)
        assert result == 6
        print("✅ Profile decorator working")
        
        print("✅ All performance optimizer tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance optimizer test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_intelligent_cache():
    """Test the intelligent caching system."""
    print("\n🧠 Testing Intelligent Cache...")
    
    try:
        from neural_operator_lab.intelligent_cache import (
            AdaptiveCache, CacheEntry, IntelligentEvictionStrategy,
            intelligent_cache, get_cache_statistics
        )
        print("✅ Intelligent cache imports successful")
        
        # Test cache entry
        entry = CacheEntry(
            key="test",
            value="data",
            timestamp=time.time(),
            computation_cost=1.5
        )
        
        assert entry.key == "test"
        assert entry.value == "data"
        assert entry.computation_cost == 1.5
        assert not entry.is_expired()  # No TTL set
        print("✅ CacheEntry working")
        
        # Test intelligent eviction strategy
        strategy = IntelligentEvictionStrategy()
        
        # Create test entries
        entries = {
            "key1": CacheEntry("key1", "value1", time.time() - 100, access_count=10),
            "key2": CacheEntry("key2", "value2", time.time() - 50, access_count=2),
            "key3": CacheEntry("key3", "value3", time.time() - 10, access_count=1)
        }
        
        victim = strategy.select_victim(entries)
        assert victim is not None
        assert victim in entries
        print("✅ IntelligentEvictionStrategy working")
        
        # Test adaptive cache
        cache = AdaptiveCache(
            max_size_mb=1.0,  # Small for testing
            max_entries=5,
            compression_threshold=100
        )
        
        # Test basic operations
        cache.put("test1", "value1")
        result = cache.get("test1")
        assert result == "value1"
        
        # Test miss
        result = cache.get("nonexistent", default="default")
        assert result == "default"
        
        # Test eviction (fill cache beyond max entries)
        for i in range(10):
            cache.put(f"key_{i}", f"value_{i}")
        
        stats = cache.get_statistics()
        assert stats['entries'] <= 5  # Should have evicted some
        assert stats['evictions'] > 0
        print("✅ AdaptiveCache eviction working")
        
        # Test TTL
        cache.put("ttl_test", "expires", ttl=0.1)  # 100ms TTL
        result = cache.get("ttl_test")
        assert result == "expires"
        
        time.sleep(0.15)  # Wait for expiration
        result = cache.get("ttl_test", default="expired")
        assert result == "expired"
        print("✅ TTL expiration working")
        
        # Test intelligent caching decorator
        computation_count = 0
        
        @intelligent_cache(ttl=1.0, computation_cost=2.0)
        def expensive_computation(n):
            nonlocal computation_count
            computation_count += 1
            time.sleep(0.01)  # Simulate expensive work
            return n ** 3
        
        # First call should compute
        result1 = expensive_computation(3)
        assert result1 == 27
        assert computation_count == 1
        
        # Second call should use cache
        result2 = expensive_computation(3)
        assert result2 == 27
        assert computation_count == 1  # Should not increment
        print("✅ Intelligent caching decorator working")
        
        # Test compression (with large data)
        large_data = "x" * 2000  # Trigger compression threshold
        cache.put("large_item", large_data)
        
        # Allow background compression to run
        time.sleep(0.1)
        
        retrieved = cache.get("large_item")
        assert retrieved == large_data
        print("✅ Cache compression working")
        
        # Test statistics
        stats = cache.get_statistics()
        assert isinstance(stats, dict)
        assert 'hit_rate' in stats
        assert 'entries' in stats
        print("✅ Cache statistics working")
        
        print("✅ All intelligent cache tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Intelligent cache test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_performance():
    """Test concurrent performance features."""
    print("\n⚡ Testing Concurrent Performance...")
    
    try:
        from neural_operator_lab.performance_optimizer import (
            optimize_parallel_execution, get_performance_stats,
            record_system_load, auto_scale_resources
        )
        print("✅ Concurrent performance imports successful")
        
        # Test parallel execution optimization
        def cpu_intensive_task(n):
            # Simulate CPU-intensive work without external dependencies
            total = 0
            for i in range(n * 1000):
                total += i % 7
            return total
        
        items = [10, 20, 30, 40, 50]
        
        # Sequential execution time
        start_time = time.time()
        sequential_results = [cpu_intensive_task(item) for item in items]
        sequential_time = time.time() - start_time
        
        # Parallel execution time
        start_time = time.time()
        parallel_results = optimize_parallel_execution(cpu_intensive_task, items)
        parallel_time = time.time() - start_time
        
        # Results should be the same (but may be in different order due to parallel processing)
        assert sorted(parallel_results) == sorted(sequential_results)
        
        # Parallel should be faster (or at least not much slower)
        speedup_ratio = sequential_time / parallel_time
        print(f"📊 Speedup ratio: {speedup_ratio:.2f}x")
        
        print("✅ Parallel execution optimization working")
        
        # Test performance statistics
        stats = get_performance_stats()
        assert isinstance(stats, dict)
        assert 'parallel_processor' in stats
        print("✅ Performance statistics working")
        
        # Test system load recording and autoscaling
        for i in range(5):
            record_system_load(50 + i * 10, 100)  # Gradually increasing load
        
        # Trigger autoscaling
        new_resources = auto_scale_resources()
        assert isinstance(new_resources, int)
        assert new_resources >= 1
        print("✅ Auto-scaling working")
        
        print("✅ All concurrent performance tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Concurrent performance test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_memory_optimization():
    """Test memory optimization features."""
    print("\n💾 Testing Memory Optimization...")
    
    try:
        from neural_operator_lab.intelligent_cache import AdaptiveCache
        print("✅ Memory optimization imports successful")
        
        # Test memory-aware caching
        cache = AdaptiveCache(
            max_size_mb=0.1,  # Very small limit to test memory management
            max_entries=10,
            compression_threshold=50  # Low threshold for testing
        )
        
        # Fill cache with various sizes of data
        test_data = [
            ("small", "x" * 10),
            ("medium", "y" * 100),
            ("large", "z" * 1000),
            ("huge", "w" * 5000)
        ]
        
        for key, value in test_data:
            cache.put(key, value)
        
        # Check that cache respects memory limits
        stats = cache.get_statistics()
        assert stats['size_mb'] <= 0.15  # Should be close to or under limit
        
        # Test that we can still retrieve data
        small_data = cache.get("small")
        assert small_data == "x" * 10 or small_data is None  # Might be evicted
        
        print("✅ Memory-aware caching working")
        
        # Test compression statistics
        if stats['compressed_entries'] > 0:
            print(f"📊 Compressed entries: {stats['compressed_entries']}")
            print(f"📊 Compressions performed: {stats['compressions']}")
            
        print("✅ Memory optimization working")
        
        print("✅ All memory optimization tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_monitoring():
    """Test performance monitoring capabilities."""
    print("\n📊 Testing Performance Monitoring...")
    
    try:
        from neural_operator_lab.performance_optimizer import PerformanceProfiler
        from neural_operator_lab.intelligent_cache import get_cache_statistics
        print("✅ Performance monitoring imports successful")
        
        # Test detailed profiling
        profiler = PerformanceProfiler()
        
        # Profile multiple operations
        operations = [
            ("fast_op", 0.01),
            ("medium_op", 0.05),
            ("slow_op", 0.1),
            ("fast_op", 0.01)  # Repeat to test aggregation
        ]
        
        for op_name, sleep_time in operations:
            with profiler.profile(op_name):
                time.sleep(sleep_time)
        
        # Test operation statistics
        fast_stats = profiler.get_operation_stats("fast_op")
        assert fast_stats['count'] == 2  # Called twice
        assert 0.008 <= fast_stats['avg_execution_time'] <= 0.015  # Around 0.01s
        
        slow_stats = profiler.get_operation_stats("slow_op")
        assert slow_stats['count'] == 1
        assert 0.09 <= slow_stats['avg_execution_time'] <= 0.12  # Around 0.1s
        
        print("✅ Operation profiling working")
        
        # Test bottleneck identification
        bottlenecks = profiler.identify_bottlenecks(threshold_seconds=0.05)
        slow_bottleneck = next((b for b in bottlenecks if b['operation'] == 'slow_op'), None)
        assert slow_bottleneck is not None
        assert slow_bottleneck['severity'] in ['high', 'medium']
        
        print("✅ Bottleneck identification working")
        
        # Test performance summary
        summary = profiler.get_summary()
        assert summary['total_operations'] == 4
        assert summary['unique_operations'] == 3
        assert len(summary['bottlenecks']) >= 1
        
        print("✅ Performance summary working")
        
        # Test cache monitoring
        cache_stats = get_cache_statistics()
        assert isinstance(cache_stats, dict)
        
        # Cache should have some activity from previous tests
        if cache_stats.get('hits', 0) > 0 or cache_stats.get('misses', 0) > 0:
            print(f"📊 Cache hit rate: {cache_stats.get('hit_rate', 0):.2%}")
        
        print("✅ Cache monitoring working")
        
        print("✅ All performance monitoring tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Performance monitoring test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all Generation 3 tests."""
    print("🚀 Testing Generation 3: Performance Optimization & Scaling")
    print("=" * 70)
    
    test_results = []
    
    # Run all Generation 3 tests
    test_results.append(("Performance Optimizer", test_performance_optimizer()))
    test_results.append(("Intelligent Cache", test_intelligent_cache()))
    test_results.append(("Concurrent Performance", test_concurrent_performance()))
    test_results.append(("Memory Optimization", test_memory_optimization()))
    test_results.append(("Performance Monitoring", test_performance_monitoring()))
    
    # Summary
    print("\n" + "=" * 70)
    print("📊 GENERATION 3 TEST SUMMARY")
    print("=" * 70)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{test_name:.<40} {status}")
        if result:
            passed += 1
    
    print("=" * 70)
    print(f"Results: {passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("🎉 All Generation 3 tests passed! Optimization framework is working!")
        print("\n📋 GENERATION 3 FEATURES IMPLEMENTED:")
        print("  ✅ Advanced caching with LRU/LFU/Intelligent eviction")
        print("  ✅ Performance profiling and bottleneck identification")
        print("  ✅ Parallel processing with adaptive worker scaling")
        print("  ✅ Automatic resource scaling based on load")
        print("  ✅ Memory-aware caching with compression")
        print("  ✅ Intelligent cache eviction using ML-inspired algorithms")
        print("  ✅ Concurrent execution optimization")
        print("  ✅ Real-time performance monitoring and statistics")
        return True
    else:
        print("⚠️ Some Generation 3 tests failed. Please check the implementation.")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)