"""Utilities for distributed training setup and communication."""

import torch
import torch.distributed as dist
import os
import logging
from typing import Optional, Any, List
import subprocess
import socket


def setup_distributed(backend: str = 'nccl',
                     rank: Optional[int] = None,
                     world_size: Optional[int] = None,
                     master_addr: str = 'localhost',
                     master_port: str = '12355',
                     timeout_minutes: int = 30):
    """Setup distributed training environment."""
    
    # Auto-detect rank and world_size from environment
    if rank is None:
        rank = int(os.environ.get('RANK', 0))
    if world_size is None:
        world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Set environment variables
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['RANK'] = str(rank)
    os.environ['WORLD_SIZE'] = str(world_size)
    
    # Initialize process group
    timeout = torch.distributed.default_pg_timeout
    if timeout_minutes > 0:
        timeout = torch.distributed.timedelta(minutes=timeout_minutes)
    
    dist.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
        timeout=timeout
    )
    
    # Set CUDA device for current process
    if torch.cuda.is_available() and backend == 'nccl':
        local_rank = int(os.environ.get('LOCAL_RANK', rank))
        torch.cuda.set_device(local_rank)
    
    logger = logging.getLogger(__name__)
    logger.info(
        f"Initialized distributed training: "
        f"rank={rank}, world_size={world_size}, backend={backend}"
    )


def cleanup_distributed():
    """Clean up distributed training."""
    if dist.is_initialized():
        dist.destroy_process_group()


def reduce_tensor(tensor: torch.Tensor, op: str = 'mean') -> torch.Tensor:
    """Reduce tensor across all processes."""
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return tensor
    
    # Clone to avoid modifying original tensor
    reduced_tensor = tensor.clone()
    
    # All-reduce
    dist.all_reduce(reduced_tensor, op=dist.ReduceOp.SUM)
    
    if op == 'mean':
        reduced_tensor = reduced_tensor / dist.get_world_size()
    
    return reduced_tensor


def broadcast_tensor(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast tensor from source rank to all ranks."""
    if not dist.is_initialized():
        return tensor
    
    dist.broadcast(tensor, src=src)
    return tensor


def gather_tensor(tensor: torch.Tensor, dst: int = 0) -> List[torch.Tensor]:
    """Gather tensors from all ranks to destination rank."""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    
    if dist.get_rank() == dst:
        # Prepare list to receive tensors
        gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gather_list, dst=dst)
        return gather_list
    else:
        dist.gather(tensor, dst=dst)
        return []


def all_gather_tensor(tensor: torch.Tensor) -> List[torch.Tensor]:
    """All-gather tensor from all ranks."""
    if not dist.is_initialized():
        return [tensor]
    
    world_size = dist.get_world_size()
    gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gather_list, tensor)
    
    return gather_list


def get_world_size() -> int:
    """Get world size."""
    return dist.get_world_size() if dist.is_initialized() else 1


def get_rank() -> int:
    """Get current rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def is_main_process() -> bool:
    """Check if current process is main process (rank 0)."""
    return get_rank() == 0


def barrier():
    """Synchronize all processes."""
    if dist.is_initialized():
        dist.barrier()


def find_free_port() -> int:
    """Find a free port for distributed training."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        s.listen(1)
        port = s.getsockname()[1]
    return port


class DistributedMetrics:
    """Track distributed training metrics."""
    
    def __init__(self):
        self.communication_volume = 0  # Bytes
        self.communication_time = 0.0  # Seconds
        self.computation_time = 0.0  # Seconds
        self.synchronization_count = 0
    
    def record_communication(self, volume_bytes: int, time_seconds: float):
        """Record communication event."""
        self.communication_volume += volume_bytes
        self.communication_time += time_seconds
        self.synchronization_count += 1
    
    def record_computation(self, time_seconds: float):
        """Record computation time."""
        self.computation_time += time_seconds
    
    def get_efficiency_metrics(self) -> dict:
        """Get communication efficiency metrics."""
        total_time = self.communication_time + self.computation_time
        
        if total_time == 0:
            return {}
        
        return {
            'communication_efficiency': self.computation_time / total_time,
            'communication_overhead': self.communication_time / total_time,
            'avg_communication_time': self.communication_time / max(1, self.synchronization_count),
            'total_communication_mb': self.communication_volume / (1024 * 1024),
            'communication_bandwidth_mbps': (
                (self.communication_volume / (1024 * 1024)) / 
                max(0.001, self.communication_time)
            )
        }
    
    def reduce_metrics(self) -> dict:
        """Reduce metrics across all processes."""
        if not dist.is_initialized():
            return self.get_efficiency_metrics()
        
        # Create tensor with all metrics
        metrics_tensor = torch.tensor([
            self.communication_volume,
            self.communication_time,
            self.computation_time,
            self.synchronization_count
        ], dtype=torch.float32)
        
        # Reduce across all processes
        reduced_metrics = reduce_tensor(metrics_tensor, op='mean')
        
        # Update local metrics with reduced values
        self.communication_volume = int(reduced_metrics[0].item())
        self.communication_time = reduced_metrics[1].item()
        self.computation_time = reduced_metrics[2].item()
        self.synchronization_count = int(reduced_metrics[3].item())
        
        return self.get_efficiency_metrics()


def setup_slurm_distributed():
    """Setup distributed training for SLURM environments."""
    
    # SLURM environment variables
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id is None:
        raise RuntimeError("Not running under SLURM")
    
    # Get node and task information
    node_list = os.environ['SLURM_NODELIST']
    num_nodes = int(os.environ['SLURM_JOB_NUM_NODES'])
    num_tasks = int(os.environ['SLURM_NTASKS'])
    task_id = int(os.environ['SLURM_PROCID'])
    
    # Get master node
    result = subprocess.run(
        ['scontrol', 'show', 'hostnames', node_list],
        capture_output=True, text=True
    )
    if result.returncode != 0:
        raise RuntimeError("Failed to get hostnames from SLURM")
    
    hostnames = result.stdout.strip().split('\n')
    master_addr = hostnames[0]
    
    # Find free port on master node
    if task_id == 0:
        master_port = find_free_port()
    else:
        master_port = 12355  # Default port
    
    # Setup distributed environment
    setup_distributed(
        backend='nccl' if torch.cuda.is_available() else 'gloo',
        rank=task_id,
        world_size=num_tasks,
        master_addr=master_addr,
        master_port=str(master_port)
    )
    
    logger = logging.getLogger(__name__)
    logger.info(
        f"SLURM distributed setup: "
        f"job_id={job_id}, nodes={num_nodes}, tasks={num_tasks}, "
        f"rank={task_id}, master={master_addr}:{master_port}"
    )


def check_distributed_environment() -> dict:
    """Check and report distributed environment status."""
    
    status = {
        'distributed_available': dist.is_available(),
        'distributed_initialized': dist.is_initialized(),
        'nccl_available': dist.is_nccl_available(),
        'mpi_available': dist.is_mpi_available(),
        'gloo_available': dist.is_gloo_available(),
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if dist.is_initialized():
        status.update({
            'world_size': dist.get_world_size(),
            'rank': dist.get_rank(),
            'backend': dist.get_backend()
        })
    
    # Environment variables
    env_vars = [
        'RANK', 'WORLD_SIZE', 'LOCAL_RANK', 'MASTER_ADDR', 'MASTER_PORT',
        'SLURM_JOB_ID', 'SLURM_PROCID', 'SLURM_NTASKS', 'SLURM_NODELIST'
    ]
    
    status['environment_variables'] = {
        var: os.environ.get(var) for var in env_vars
    }
    
    return status


def optimize_distributed_communication(model: torch.nn.Module,
                                      bucket_size_mb: int = 25,
                                      compression_enabled: bool = False) -> torch.nn.Module:
    """Optimize model for distributed communication."""
    
    if not hasattr(model, '_ddp_params_and_buffers_to_ignore'):
        # Set parameters to ignore during DDP synchronization
        model._ddp_params_and_buffers_to_ignore = set()
    
    # Configure bucket size for gradient allreduce
    if hasattr(model, '_set_static_graph'):
        # Enable static graph optimization if available
        model._set_static_graph()
    
    # Setup gradient compression hooks if enabled
    if compression_enabled:
        _setup_gradient_compression_hooks(model)
    
    return model


def _setup_gradient_compression_hooks(model: torch.nn.Module):
    """Setup hooks for gradient compression."""
    
    def compression_hook(grad):
        """Compress gradients using top-k sparsification."""
        if grad is not None:
            # Simple top-k compression (10% of gradients)
            k = max(1, int(grad.numel() * 0.1))
            flat_grad = grad.flatten()
            
            # Get top-k absolute values
            _, indices = torch.topk(torch.abs(flat_grad), k)
            
            # Zero out non-top-k values
            mask = torch.zeros_like(flat_grad, dtype=torch.bool)
            mask[indices] = True
            flat_grad[~mask] = 0
            
            return flat_grad.reshape(grad.shape)
        return grad
    
    # Register hooks for all parameters
    for param in model.parameters():
        if param.requires_grad:
            param.register_hook(compression_hook)