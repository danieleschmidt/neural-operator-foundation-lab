"""Distributed optimization for neural operators."""

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from typing import Dict, List, Optional, Any, Tuple
import logging
import os

logger = logging.getLogger(__name__)


class DistributedOptimizer:
    """Optimizer for distributed neural operator training."""
    
    def __init__(
        self,
        strategy: str = "ddp",  # "ddp", "fsdp", "pipeline"
        mixed_precision: bool = True,
        gradient_accumulation_steps: int = 1
    ):
        """Initialize distributed optimizer.
        
        Args:
            strategy: Distributed training strategy
            mixed_precision: Whether to use mixed precision
            gradient_accumulation_steps: Number of gradient accumulation steps
        """
        self.strategy = strategy
        self.mixed_precision = mixed_precision
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Check if distributed is available
        self.is_distributed = dist.is_available() and dist.is_initialized()
        
        if self.is_distributed:
            self.rank = dist.get_rank()
            self.world_size = dist.get_world_size()
        else:
            self.rank = 0
            self.world_size = 1
        
        logger.info(f"Distributed optimizer initialized - strategy: {strategy}, "
                   f"rank: {self.rank}, world_size: {self.world_size}")
    
    def wrap_model(self, model: torch.nn.Module) -> torch.nn.Module:
        """Wrap model for distributed training.
        
        Args:
            model: Model to wrap
            
        Returns:
            Wrapped model
        """
        if not self.is_distributed:
            logger.info("Single process training, no model wrapping needed")
            return model
        
        device = next(model.parameters()).device
        
        if self.strategy == "ddp":
            wrapped_model = DDP(
                model,
                device_ids=[device.index] if device.type == 'cuda' else None,
                find_unused_parameters=True
            )
            logger.info("Model wrapped with DistributedDataParallel")
            
        elif self.strategy == "fsdp":
            # FSDP configuration
            from torch.distributed.fsdp.wrap import (
                size_based_auto_wrap_policy,
                transformer_auto_wrap_policy
            )
            
            auto_wrap_policy = size_based_auto_wrap_policy
            
            wrapped_model = FSDP(
                model,
                auto_wrap_policy=auto_wrap_policy,
                mixed_precision=self._get_fsdp_mixed_precision_policy(),
                device_id=device.index if device.type == 'cuda' else None
            )
            logger.info("Model wrapped with FullyShardedDataParallel")
            
        else:
            logger.warning(f"Unknown strategy {self.strategy}, using original model")
            wrapped_model = model
        
        return wrapped_model
    
    def _get_fsdp_mixed_precision_policy(self):
        """Get FSDP mixed precision policy."""
        if not self.mixed_precision:
            return None
        
        try:
            from torch.distributed.fsdp import MixedPrecision
            return MixedPrecision(
                param_dtype=torch.float16,
                reduce_dtype=torch.float16,
                buffer_dtype=torch.float16
            )
        except ImportError:
            logger.warning("FSDP MixedPrecision not available")
            return None
    
    def setup_distributed_sampler(self, dataset):
        """Setup distributed sampler for dataset.
        
        Args:
            dataset: Dataset to sample from
            
        Returns:
            Distributed sampler or None
        """
        if not self.is_distributed:
            return None
        
        from torch.utils.data.distributed import DistributedSampler
        
        sampler = DistributedSampler(
            dataset,
            num_replicas=self.world_size,
            rank=self.rank,
            shuffle=True
        )
        
        logger.info("Distributed sampler configured")
        return sampler
    
    def all_reduce_tensor(self, tensor: torch.Tensor) -> torch.Tensor:
        """All-reduce tensor across processes.
        
        Args:
            tensor: Tensor to reduce
            
        Returns:
            Reduced tensor
        """
        if not self.is_distributed:
            return tensor
        
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= self.world_size
        return tensor
    
    def gather_tensor(self, tensor: torch.Tensor) -> List[torch.Tensor]:
        """Gather tensor from all processes.
        
        Args:
            tensor: Tensor to gather
            
        Returns:
            List of tensors from all processes
        """
        if not self.is_distributed:
            return [tensor]
        
        gathered_tensors = [torch.zeros_like(tensor) for _ in range(self.world_size)]
        dist.all_gather(gathered_tensors, tensor)
        return gathered_tensors
    
    def broadcast_tensor(self, tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
        """Broadcast tensor from source process.
        
        Args:
            tensor: Tensor to broadcast
            src: Source process rank
            
        Returns:
            Broadcasted tensor
        """
        if not self.is_distributed:
            return tensor
        
        dist.broadcast(tensor, src=src)
        return tensor


class ModelSharding:
    """Model sharding utilities for large neural operators."""
    
    def __init__(self, num_shards: Optional[int] = None):
        """Initialize model sharding.
        
        Args:
            num_shards: Number of shards. Auto-detected if None.
        """
        if num_shards is None:
            if dist.is_available() and dist.is_initialized():
                num_shards = dist.get_world_size()
            else:
                num_shards = 1
        
        self.num_shards = num_shards
        self.current_shard = 0
        
        if dist.is_available() and dist.is_initialized():
            self.current_shard = dist.get_rank()
        
        logger.info(f"Model sharding initialized - {num_shards} shards, "
                   f"current shard: {self.current_shard}")
    
    def shard_parameters(
        self,
        model: torch.nn.Module,
        shard_axis: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Shard model parameters across processes.
        
        Args:
            model: Model to shard
            shard_axis: Axis along which to shard parameters
            
        Returns:
            Dictionary of sharded parameters
        """
        sharded_params = {}
        
        for name, param in model.named_parameters():
            if param.numel() < self.num_shards:
                # Don't shard very small parameters
                sharded_params[name] = param
                continue
            
            # Calculate shard size
            total_size = param.shape[shard_axis]
            shard_size = total_size // self.num_shards
            remainder = total_size % self.num_shards
            
            # Calculate start and end indices for current shard
            start_idx = self.current_shard * shard_size
            if self.current_shard < remainder:
                start_idx += self.current_shard
                shard_size += 1
            else:
                start_idx += remainder
            
            end_idx = start_idx + shard_size
            
            # Extract shard
            if shard_axis == 0:
                shard = param[start_idx:end_idx]
            elif shard_axis == 1:
                shard = param[:, start_idx:end_idx]
            else:
                # More complex slicing for higher dimensions
                slices = [slice(None)] * param.ndim
                slices[shard_axis] = slice(start_idx, end_idx)
                shard = param[tuple(slices)]
            
            sharded_params[name] = shard.clone()
        
        logger.info(f"Parameters sharded for shard {self.current_shard}")
        return sharded_params
    
    def gather_sharded_parameters(
        self,
        sharded_params: Dict[str, torch.Tensor],
        shard_axis: int = 0
    ) -> Dict[str, torch.Tensor]:
        """Gather sharded parameters from all processes.
        
        Args:
            sharded_params: Sharded parameters
            shard_axis: Axis along which parameters were sharded
            
        Returns:
            Complete parameters
        """
        if not (dist.is_available() and dist.is_initialized()):
            return sharded_params
        
        complete_params = {}
        
        for name, shard in sharded_params.items():
            # Gather all shards
            all_shards = [torch.zeros_like(shard) for _ in range(self.num_shards)]
            dist.all_gather(all_shards, shard)
            
            # Concatenate shards
            complete_param = torch.cat(all_shards, dim=shard_axis)
            complete_params[name] = complete_param
        
        logger.info("Sharded parameters gathered")
        return complete_params
    
    def balance_load(
        self,
        workload_sizes: List[int]
    ) -> List[List[int]]:
        """Balance workload across shards.
        
        Args:
            workload_sizes: List of workload sizes
            
        Returns:
            List of workload assignments per shard
        """
        # Simple greedy load balancing
        shard_loads = [[] for _ in range(self.num_shards)]
        shard_totals = [0] * self.num_shards
        
        # Sort workloads by size (largest first)
        sorted_workloads = sorted(enumerate(workload_sizes), 
                                key=lambda x: x[1], reverse=True)
        
        for workload_idx, workload_size in sorted_workloads:
            # Assign to shard with minimum current load
            min_shard = min(range(self.num_shards), key=lambda i: shard_totals[i])
            shard_loads[min_shard].append(workload_idx)
            shard_totals[min_shard] += workload_size
        
        logger.info(f"Load balanced across {self.num_shards} shards")
        logger.debug(f"Shard totals: {shard_totals}")
        
        return shard_loads


def setup_distributed_environment(
    backend: str = "nccl",
    init_method: str = "env://",
    timeout_minutes: int = 30
) -> bool:
    """Setup distributed training environment.
    
    Args:
        backend: Distributed backend ("nccl", "gloo", "mpi")
        init_method: Initialization method
        timeout_minutes: Timeout for initialization
        
    Returns:
        True if setup successful, False otherwise
    """
    try:
        if not dist.is_available():
            logger.warning("Distributed training not available")
            return False
        
        if dist.is_initialized():
            logger.info("Distributed training already initialized")
            return True
        
        # Get environment variables
        rank = int(os.environ.get("RANK", 0))
        world_size = int(os.environ.get("WORLD_SIZE", 1))
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        
        # Initialize distributed training
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.timedelta(minutes=timeout_minutes)
        )
        
        # Set device for current process
        if torch.cuda.is_available() and backend == "nccl":
            torch.cuda.set_device(local_rank)
        
        logger.info(f"Distributed training initialized - rank: {rank}, "
                   f"world_size: {world_size}, backend: {backend}")
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to setup distributed environment: {e}")
        return False


def cleanup_distributed_environment() -> None:
    """Cleanup distributed training environment."""
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Distributed training environment cleaned up")
    else:
        logger.debug("No distributed environment to cleanup")