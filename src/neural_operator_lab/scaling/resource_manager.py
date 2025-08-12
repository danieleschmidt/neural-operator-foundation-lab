"""Resource management for neural operator scaling."""

import time
import threading
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass
from enum import Enum
import logging
import psutil

logger = logging.getLogger(__name__)


class ResourceType(Enum):
    """Types of resources that can be managed."""
    
    CPU = "cpu"
    MEMORY = "memory"
    GPU = "gpu"
    STORAGE = "storage"
    NETWORK = "network"


@dataclass
class ResourceAllocation:
    """Resource allocation specification."""
    
    cpu_cores: Optional[int] = None
    memory_gb: Optional[float] = None
    gpu_devices: Optional[List[int]] = None
    storage_gb: Optional[float] = None
    network_bandwidth_mbps: Optional[float] = None
    
    def __post_init__(self):
        """Post-initialization validation."""
        if self.gpu_devices is None:
            self.gpu_devices = []


class ResourceMetrics(NamedTuple):
    """Current resource usage metrics."""
    
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    gpu_memory_used_gb: Optional[float]
    gpu_utilization: Optional[float]
    disk_usage_percent: float
    network_bytes_sent: int
    network_bytes_recv: int
    timestamp: float


class ResourceManager:
    """Manages resource allocation and monitoring."""
    
    def __init__(
        self,
        monitoring_interval: float = 5.0,
        resource_limits: Optional[ResourceAllocation] = None
    ):
        """Initialize resource manager.
        
        Args:
            monitoring_interval: Resource monitoring interval in seconds
            resource_limits: Optional resource limits
        """
        self.monitoring_interval = monitoring_interval
        self.resource_limits = resource_limits or ResourceAllocation()
        
        # Resource tracking
        self._allocations: Dict[str, ResourceAllocation] = {}
        self._usage_history: List[ResourceMetrics] = []
        self._max_history = 1000
        
        # Monitoring
        self._monitoring = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        
        # System info
        self._system_cpu_count = psutil.cpu_count()
        self._system_memory_gb = psutil.virtual_memory().total / (1024**3)
        self._gpu_available = self._check_gpu_availability()
        
        logger.info(f"Resource manager initialized - "
                   f"CPU cores: {self._system_cpu_count}, "
                   f"Memory: {self._system_memory_gb:.1f}GB, "
                   f"GPU available: {self._gpu_available}")
    
    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available."""
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False
    
    def allocate_resources(
        self,
        allocation_id: str,
        allocation: ResourceAllocation
    ) -> bool:
        """Allocate resources for a task.
        
        Args:
            allocation_id: Unique identifier for allocation
            allocation: Resource allocation specification
            
        Returns:
            True if allocation successful, False otherwise
        """
        with self._lock:
            # Check if allocation already exists
            if allocation_id in self._allocations:
                logger.warning(f"Allocation {allocation_id} already exists")
                return False
            
            # Validate allocation against system limits
            if not self._validate_allocation(allocation):
                logger.error(f"Allocation {allocation_id} exceeds system limits")
                return False
            
            # Check resource availability
            if not self._check_resource_availability(allocation):
                logger.warning(f"Insufficient resources for allocation {allocation_id}")
                return False
            
            # Record allocation
            self._allocations[allocation_id] = allocation
            
            logger.info(f"Resources allocated for {allocation_id}: "
                       f"CPU={allocation.cpu_cores}, "
                       f"Memory={allocation.memory_gb}GB, "
                       f"GPU={allocation.gpu_devices}")
            
            return True
    
    def deallocate_resources(self, allocation_id: str) -> bool:
        """Deallocate resources for a task.
        
        Args:
            allocation_id: Allocation identifier to deallocate
            
        Returns:
            True if deallocation successful, False otherwise
        """
        with self._lock:
            if allocation_id not in self._allocations:
                logger.warning(f"Allocation {allocation_id} not found")
                return False
            
            del self._allocations[allocation_id]
            logger.info(f"Resources deallocated for {allocation_id}")
            return True
    
    def _validate_allocation(self, allocation: ResourceAllocation) -> bool:
        """Validate allocation against system limits.
        
        Args:
            allocation: Resource allocation to validate
            
        Returns:
            True if allocation is valid
        """
        # Check CPU
        if allocation.cpu_cores and allocation.cpu_cores > self._system_cpu_count:
            return False
        
        # Check memory
        if allocation.memory_gb and allocation.memory_gb > self._system_memory_gb:
            return False
        
        # Check GPU
        if allocation.gpu_devices and not self._gpu_available:
            return False
        
        if allocation.gpu_devices:
            try:
                import torch
                available_gpus = torch.cuda.device_count()
                for gpu_id in allocation.gpu_devices:
                    if gpu_id >= available_gpus:
                        return False
            except ImportError:
                return False
        
        return True
    
    def _check_resource_availability(self, allocation: ResourceAllocation) -> bool:
        """Check if resources are available for allocation.
        
        Args:
            allocation: Resource allocation to check
            
        Returns:
            True if resources are available
        """
        # Get current resource usage
        current_usage = self._get_current_usage()
        
        # Calculate total allocated resources
        total_cpu = sum(a.cpu_cores or 0 for a in self._allocations.values())
        total_memory = sum(a.memory_gb or 0 for a in self._allocations.values())
        
        # Check CPU availability
        if allocation.cpu_cores:
            available_cpu = self._system_cpu_count - total_cpu
            if allocation.cpu_cores > available_cpu:
                return False
        
        # Check memory availability
        if allocation.memory_gb:
            available_memory = self._system_memory_gb - total_memory
            if allocation.memory_gb > available_memory:
                return False
        
        # Check GPU availability
        if allocation.gpu_devices:
            allocated_gpus = set()
            for alloc in self._allocations.values():
                allocated_gpus.update(alloc.gpu_devices or [])
            
            for gpu_id in allocation.gpu_devices:
                if gpu_id in allocated_gpus:
                    return False
        
        return True
    
    def get_resource_usage(self) -> ResourceMetrics:
        """Get current resource usage.
        
        Returns:
            Current resource usage metrics
        """
        return self._get_current_usage()
    
    def _get_current_usage(self) -> ResourceMetrics:
        """Get current system resource usage."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        
        # GPU metrics
        gpu_memory_used = None
        gpu_utilization = None
        
        if self._gpu_available:
            try:
                import torch
                if torch.cuda.device_count() > 0:
                    gpu_memory_used = torch.cuda.memory_allocated(0) / (1024**3)
                    # GPU utilization would require nvidia-ml-py
            except Exception:
                pass
        
        # Disk usage
        disk = psutil.disk_usage('/')
        
        # Network stats
        network = psutil.net_io_counters()
        
        return ResourceMetrics(
            cpu_percent=cpu_percent,
            memory_percent=memory.percent,
            memory_used_gb=memory.used / (1024**3),
            gpu_memory_used_gb=gpu_memory_used,
            gpu_utilization=gpu_utilization,
            disk_usage_percent=(disk.used / disk.total) * 100,
            network_bytes_sent=network.bytes_sent,
            network_bytes_recv=network.bytes_recv,
            timestamp=time.time()
        )
    
    def start_monitoring(self) -> None:
        """Start resource monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            daemon=True
        )
        self._monitor_thread.start()
        logger.info("Resource monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop resource monitoring."""
        self._monitoring = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=5.0)
        logger.info("Resource monitoring stopped")
    
    def _monitoring_loop(self) -> None:
        """Resource monitoring loop."""
        while self._monitoring:
            try:
                usage = self._get_current_usage()
                
                with self._lock:
                    self._usage_history.append(usage)
                    
                    # Maintain history limit
                    if len(self._usage_history) > self._max_history:
                        self._usage_history.pop(0)
                
                # Check for resource alerts
                self._check_resource_alerts(usage)
                
            except Exception as e:
                logger.error(f"Resource monitoring error: {e}")
            
            time.sleep(self.monitoring_interval)
    
    def _check_resource_alerts(self, usage: ResourceMetrics) -> None:
        """Check for resource usage alerts.
        
        Args:
            usage: Current resource usage
        """
        # CPU alert
        if usage.cpu_percent > 90:
            logger.warning(f"High CPU usage: {usage.cpu_percent:.1f}%")
        
        # Memory alert
        if usage.memory_percent > 90:
            logger.warning(f"High memory usage: {usage.memory_percent:.1f}%")
        
        # GPU alert
        if usage.gpu_memory_used_gb and usage.gpu_memory_used_gb > 8:  # 8GB threshold
            logger.warning(f"High GPU memory usage: {usage.gpu_memory_used_gb:.1f}GB")
    
    def get_usage_history(self, last_n: Optional[int] = None) -> List[ResourceMetrics]:
        """Get resource usage history.
        
        Args:
            last_n: Number of recent entries to return
            
        Returns:
            List of resource usage metrics
        """
        with self._lock:
            history = self._usage_history.copy()
        
        if last_n:
            return history[-last_n:]
        return history
    
    def get_allocation_summary(self) -> Dict[str, Any]:
        """Get summary of current allocations.
        
        Returns:
            Allocation summary dictionary
        """
        with self._lock:
            total_cpu = sum(a.cpu_cores or 0 for a in self._allocations.values())
            total_memory = sum(a.memory_gb or 0 for a in self._allocations.values())
            allocated_gpus = set()
            
            for alloc in self._allocations.values():
                allocated_gpus.update(alloc.gpu_devices or [])
            
            summary = {
                'total_allocations': len(self._allocations),
                'allocated_cpu_cores': total_cpu,
                'allocated_memory_gb': total_memory,
                'allocated_gpu_devices': list(allocated_gpus),
                'available_cpu_cores': self._system_cpu_count - total_cpu,
                'available_memory_gb': self._system_memory_gb - total_memory,
                'system_cpu_cores': self._system_cpu_count,
                'system_memory_gb': self._system_memory_gb,
                'gpu_available': self._gpu_available
            }
        
        return summary
    
    def optimize_allocation(
        self,
        current_usage: Optional[ResourceMetrics] = None
    ) -> Dict[str, ResourceAllocation]:
        """Optimize resource allocations based on usage patterns.
        
        Args:
            current_usage: Current resource usage metrics
            
        Returns:
            Dictionary of optimized allocations
        """
        if current_usage is None:
            current_usage = self._get_current_usage()
        
        optimized_allocations = {}
        
        with self._lock:
            # Simple optimization: adjust based on current usage
            for alloc_id, allocation in self._allocations.items():
                optimized = ResourceAllocation(
                    cpu_cores=allocation.cpu_cores,
                    memory_gb=allocation.memory_gb,
                    gpu_devices=allocation.gpu_devices.copy() if allocation.gpu_devices else [],
                    storage_gb=allocation.storage_gb,
                    network_bandwidth_mbps=allocation.network_bandwidth_mbps
                )
                
                # Adjust CPU if underutilized
                if (allocation.cpu_cores and 
                    current_usage.cpu_percent < 50 and 
                    allocation.cpu_cores > 1):
                    optimized.cpu_cores = max(1, allocation.cpu_cores // 2)
                
                # Adjust memory if underutilized
                if (allocation.memory_gb and 
                    current_usage.memory_percent < 50 and 
                    allocation.memory_gb > 1):
                    optimized.memory_gb = max(1, allocation.memory_gb * 0.75)
                
                optimized_allocations[alloc_id] = optimized
        
        logger.info("Resource allocations optimized")
        return optimized_allocations
    
    def __enter__(self):
        """Context manager entry."""
        self.start_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_monitoring()