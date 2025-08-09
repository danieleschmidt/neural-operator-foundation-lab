"""Comprehensive Logging System

Advanced logging infrastructure with structured logging, performance metrics,
distributed logging, and integration with popular monitoring systems.
"""

import logging
import json
import time
import threading
import queue
import sys
import os
from typing import Dict, Any, Optional, List, Callable, Union
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime
import traceback
from contextlib import contextmanager
import atexit

import torch
import numpy as np

try:
    from torch.profiler import profile, ProfilerActivity
    HAS_PROFILER = True
except ImportError:
    HAS_PROFILER = False

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

try:
    from tensorboardX import SummaryWriter
    HAS_TENSORBOARD = True
except ImportError:
    HAS_TENSORBOARD = False


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    logger_name: str
    message: str
    module: str
    function: str
    line_number: int
    thread_id: int
    process_id: int
    extra_data: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), default=str)


@dataclass
class PerformanceMetric:
    """Performance metric entry."""
    name: str
    value: float
    unit: str
    timestamp: float
    tags: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class StructuredFormatter(logging.Formatter):
    """Structured logging formatter for JSON output."""
    
    def format(self, record: logging.LogRecord) -> str:
        # Extract extra data
        extra_data = {}
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 
                          'pathname', 'filename', 'module', 'lineno', 
                          'funcName', 'created', 'msecs', 'relativeCreated',
                          'thread', 'threadName', 'processName', 'process',
                          'getMessage', 'exc_info', 'exc_text', 'stack_info']:
                try:
                    # Ensure JSON serializable
                    json.dumps(value)
                    extra_data[key] = value
                except (TypeError, ValueError):
                    extra_data[key] = str(value)
        
        # Create structured log entry
        log_entry = LogEntry(
            timestamp=record.created,
            level=record.levelname,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            function=record.funcName,
            line_number=record.lineno,
            thread_id=record.thread,
            process_id=record.process,
            extra_data=extra_data
        )
        
        return log_entry.to_json()


class PerformanceLogger:
    """Logger for performance metrics and profiling."""
    
    def __init__(self, name: str = "performance"):
        self.name = name
        self.metrics_queue = queue.Queue()
        self.start_times: Dict[str, float] = {}
        self.counters: Dict[str, int] = {}
        self._lock = threading.Lock()
        
    def start_timer(self, name: str):
        """Start a named timer."""
        with self._lock:
            self.start_times[name] = time.perf_counter()
    
    def end_timer(self, name: str, tags: Optional[Dict[str, str]] = None) -> float:
        """End a named timer and log duration."""
        with self._lock:
            if name not in self.start_times:
                raise ValueError(f"Timer '{name}' not started")
            
            duration = time.perf_counter() - self.start_times[name]
            del self.start_times[name]
            
            metric = PerformanceMetric(
                name=f"{name}_duration",
                value=duration,
                unit="seconds",
                timestamp=time.time(),
                tags=tags or {}
            )
            
            self.metrics_queue.put(metric)
            return duration
    
    @contextmanager
    def timer(self, name: str, tags: Optional[Dict[str, str]] = None):
        """Context manager for timing operations."""
        self.start_timer(name)
        try:
            yield
        finally:
            self.end_timer(name, tags)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """Increment a counter metric."""
        with self._lock:
            self.counters[name] = self.counters.get(name, 0) + value
            
            metric = PerformanceMetric(
                name=name,
                value=self.counters[name],
                unit="count",
                timestamp=time.time(),
                tags=tags or {}
            )
            
            self.metrics_queue.put(metric)
    
    def log_gauge(self, name: str, value: float, unit: str = "", tags: Optional[Dict[str, str]] = None):
        """Log a gauge metric (point-in-time value)."""
        metric = PerformanceMetric(
            name=name,
            value=value,
            unit=unit,
            timestamp=time.time(),
            tags=tags or {}
        )
        
        self.metrics_queue.put(metric)
    
    def log_memory_usage(self, tags: Optional[Dict[str, str]] = None):
        """Log current memory usage."""
        try:
            import psutil
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            self.log_gauge("memory_usage_mb", memory_mb, "MB", tags)
        except ImportError:
            pass
        
        # GPU memory if available
        if torch.cuda.is_available():
            allocated_mb = torch.cuda.memory_allocated() / 1024 / 1024
            cached_mb = torch.cuda.memory_reserved() / 1024 / 1024
            
            self.log_gauge("gpu_memory_allocated_mb", allocated_mb, "MB", tags)
            self.log_gauge("gpu_memory_cached_mb", cached_mb, "MB", tags)
    
    def get_metrics(self) -> List[PerformanceMetric]:
        """Get all queued metrics."""
        metrics = []
        while not self.metrics_queue.empty():
            try:
                metrics.append(self.metrics_queue.get_nowait())
            except queue.Empty:
                break
        return metrics


class AsyncLogHandler(logging.Handler):
    """Asynchronous log handler to avoid blocking main thread."""
    
    def __init__(self, target_handler: logging.Handler):
        super().__init__()
        self.target_handler = target_handler
        self.log_queue = queue.Queue()
        self.worker_thread = threading.Thread(target=self._worker, daemon=True)
        self.worker_thread.start()
        self._shutdown = False
        
        # Register shutdown handler
        atexit.register(self.shutdown)
    
    def emit(self, record: logging.LogRecord):
        """Queue log record for asynchronous processing."""
        if not self._shutdown:
            try:
                self.log_queue.put(record, block=False)
            except queue.Full:
                # Drop log if queue is full to prevent memory issues
                pass
    
    def _worker(self):
        """Worker thread to process log records."""
        while not self._shutdown:
            try:
                record = self.log_queue.get(timeout=1.0)
                if record is None:  # Shutdown signal
                    break
                self.target_handler.emit(record)
                self.log_queue.task_done()
            except queue.Empty:
                continue
            except Exception:
                # Avoid infinite recursion in logging
                pass
    
    def shutdown(self):
        """Gracefully shutdown the handler."""
        self._shutdown = True
        self.log_queue.put(None)  # Shutdown signal
        self.worker_thread.join(timeout=5.0)


class ComprehensiveLogger:
    """Comprehensive logging system with multiple backends."""
    
    def __init__(
        self,
        name: str = "neural_operator_lab",
        log_level: str = "INFO",
        log_dir: Optional[str] = None,
        structured_logging: bool = True,
        async_logging: bool = True,
        enable_performance_logging: bool = True,
        enable_wandb: bool = False,
        enable_tensorboard: bool = False,
        wandb_project: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None
    ):
        self.name = name
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir) if log_dir else Path.cwd() / "logs"
        self.log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create main logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Performance logger
        self.perf_logger = PerformanceLogger() if enable_performance_logging else None
        
        # Setup handlers
        self._setup_console_handler(structured_logging, async_logging)
        self._setup_file_handlers(structured_logging, async_logging)
        
        # External integrations
        self.wandb_enabled = enable_wandb and HAS_WANDB
        self.tensorboard_enabled = enable_tensorboard and HAS_TENSORBOARD
        self.tensorboard_writer = None
        
        if self.wandb_enabled and wandb_project:
            self._setup_wandb(wandb_project)
        
        if self.tensorboard_enabled:
            tb_dir = tensorboard_log_dir or str(self.log_dir / "tensorboard")
            self._setup_tensorboard(tb_dir)
        
        # Profiling context
        self._profiler = None
        
        self.logger.info(f"Comprehensive logger initialized: {name}")
    
    def _setup_console_handler(self, structured: bool, async_handler: bool):
        """Setup console logging handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(self.log_level)
        
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        console_handler.setFormatter(formatter)
        
        if async_handler:
            async_console_handler = AsyncLogHandler(console_handler)
            self.logger.addHandler(async_console_handler)
        else:
            self.logger.addHandler(console_handler)
    
    def _setup_file_handlers(self, structured: bool, async_handler: bool):
        """Setup file logging handlers."""
        # Main log file
        main_log_file = self.log_dir / f"{self.name}.log"
        file_handler = logging.FileHandler(main_log_file)
        file_handler.setLevel(self.log_level)
        
        # Error log file
        error_log_file = self.log_dir / f"{self.name}_errors.log"
        error_handler = logging.FileHandler(error_log_file)
        error_handler.setLevel(logging.ERROR)
        
        # Formatters
        if structured:
            formatter = StructuredFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
        
        file_handler.setFormatter(formatter)
        error_handler.setFormatter(formatter)
        
        # Add handlers (with async wrapper if enabled)
        if async_handler:
            self.logger.addHandler(AsyncLogHandler(file_handler))
            self.logger.addHandler(AsyncLogHandler(error_handler))
        else:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(error_handler)
    
    def _setup_wandb(self, project: str):
        """Setup Weights & Biases integration."""
        try:
            wandb.init(project=project, name=f"{self.name}_{int(time.time())}")
            self.logger.info("Weights & Biases integration enabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize Weights & Biases: {e}")
            self.wandb_enabled = False
    
    def _setup_tensorboard(self, log_dir: str):
        """Setup TensorBoard integration."""
        try:
            self.tensorboard_writer = SummaryWriter(log_dir)
            self.logger.info(f"TensorBoard integration enabled: {log_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to initialize TensorBoard: {e}")
            self.tensorboard_enabled = False
    
    # Convenience methods for different log levels
    def debug(self, message: str, **kwargs):
        self.logger.debug(message, extra=kwargs)
    
    def info(self, message: str, **kwargs):
        self.logger.info(message, extra=kwargs)
    
    def warning(self, message: str, **kwargs):
        self.logger.warning(message, extra=kwargs)
    
    def error(self, message: str, **kwargs):
        self.logger.error(message, extra=kwargs)
    
    def critical(self, message: str, **kwargs):
        self.logger.critical(message, extra=kwargs)
    
    def log_training_metrics(
        self,
        epoch: int,
        metrics: Dict[str, float],
        step: Optional[int] = None
    ):
        """Log training metrics to all configured backends."""
        
        # Log to main logger
        metrics_str = ", ".join(f"{k}={v:.6f}" for k, v in metrics.items())
        self.info(f"Epoch {epoch} metrics: {metrics_str}", 
                 epoch=epoch, step=step, **metrics)
        
        # Log to performance logger
        if self.perf_logger:
            for name, value in metrics.items():
                self.perf_logger.log_gauge(
                    f"training_{name}", 
                    value, 
                    tags={"epoch": str(epoch)}
                )
        
        # Log to W&B
        if self.wandb_enabled:
            wandb_metrics = {"epoch": epoch}
            wandb_metrics.update(metrics)
            if step is not None:
                wandb_metrics["step"] = step
            wandb.log(wandb_metrics)
        
        # Log to TensorBoard
        if self.tensorboard_enabled and self.tensorboard_writer:
            global_step = step if step is not None else epoch
            for name, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"training/{name}", value, global_step)
    
    def log_model_info(self, model: torch.nn.Module, input_shape: Optional[tuple] = None):
        """Log comprehensive model information."""
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        model_info = {
            "model_class": model.__class__.__name__,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "non_trainable_parameters": total_params - trainable_params,
        }
        
        if input_shape:
            model_info["input_shape"] = input_shape
        
        self.info("Model information", **model_info)
        
        # Log to W&B
        if self.wandb_enabled:
            wandb.log(model_info)
    
    @contextmanager
    def profiling_context(
        self, 
        name: str = "training",
        activities: Optional[List] = None,
        record_shapes: bool = True
    ):
        """Context manager for PyTorch profiling."""
        
        if not HAS_PROFILER:
            self.warning("PyTorch profiler not available")
            yield
            return
        
        if activities is None:
            activities = [ProfilerActivity.CPU]
            if torch.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
        
        profile_dir = self.log_dir / "profiles"
        profile_dir.mkdir(exist_ok=True)
        
        with profile(
            activities=activities,
            record_shapes=record_shapes,
            on_trace_ready=torch.profiler.tensorboard_trace_handler(str(profile_dir))
        ) as prof:
            self._profiler = prof
            self.info(f"Started profiling: {name}")
            
            try:
                yield prof
            finally:
                self.info(f"Finished profiling: {name}")
                
                # Log profiler statistics
                if prof:
                    stats = prof.key_averages()
                    cpu_time = sum(item.cpu_time_total for item in stats) / 1000  # Convert to ms
                    cuda_time = sum(item.cuda_time_total for item in stats) / 1000 if torch.cuda.is_available() else 0
                    
                    self.info(
                        f"Profiling results: CPU time={cpu_time:.2f}ms, CUDA time={cuda_time:.2f}ms",
                        profiling_name=name,
                        cpu_time_ms=cpu_time,
                        cuda_time_ms=cuda_time
                    )
                
                self._profiler = None
    
    def log_exception(self, exc: Exception, context: str = ""):
        """Log exception with full traceback and context."""
        
        exc_info = {
            "exception_type": type(exc).__name__,
            "exception_message": str(exc),
            "traceback": traceback.format_exc(),
            "context": context
        }
        
        self.error(f"Exception occurred: {context}", **exc_info)
    
    def log_system_info(self):
        """Log comprehensive system information."""
        
        system_info = {
            "python_version": sys.version,
            "pytorch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
        }
        
        if torch.cuda.is_available():
            system_info.update({
                "cuda_version": torch.version.cuda,
                "gpu_count": torch.cuda.device_count(),
                "gpu_name": torch.cuda.get_device_name(0)
            })
        
        try:
            import psutil
            system_info.update({
                "cpu_count": psutil.cpu_count(),
                "memory_gb": psutil.virtual_memory().total / (1024**3),
            })
        except ImportError:
            pass
        
        self.info("System information", **system_info)
        
        # Log to W&B as system metadata
        if self.wandb_enabled:
            wandb.config.update(system_info)
    
    def create_child_logger(self, name: str) -> logging.Logger:
        """Create a child logger with the same configuration."""
        child_name = f"{self.name}.{name}"
        child_logger = logging.getLogger(child_name)
        
        # Child loggers inherit parent handlers by default
        return child_logger
    
    def flush_metrics(self):
        """Flush all pending metrics to backends."""
        
        if self.perf_logger:
            metrics = self.perf_logger.get_metrics()
            for metric in metrics:
                self.debug(f"Performance metric: {metric.name}={metric.value}{metric.unit}")
                
                # Send to external backends
                if self.wandb_enabled:
                    wandb.log({metric.name: metric.value}, step=int(metric.timestamp))
                
                if self.tensorboard_enabled and self.tensorboard_writer:
                    self.tensorboard_writer.add_scalar(
                        f"performance/{metric.name}", 
                        metric.value, 
                        int(metric.timestamp)
                    )
        
        # Flush TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.flush()
    
    def shutdown(self):
        """Gracefully shutdown the logging system."""
        
        self.info("Shutting down comprehensive logger")
        
        # Flush any remaining metrics
        self.flush_metrics()
        
        # Close TensorBoard writer
        if self.tensorboard_writer:
            self.tensorboard_writer.close()
        
        # Finish W&B run
        if self.wandb_enabled:
            wandb.finish()
        
        # Shutdown async handlers
        for handler in self.logger.handlers[:]:
            if isinstance(handler, AsyncLogHandler):
                handler.shutdown()
            handler.close()
            self.logger.removeHandler(handler)


# Global logger instance
_global_logger: Optional[ComprehensiveLogger] = None


def get_logger(name: str = "neural_operator_lab") -> ComprehensiveLogger:
    """Get or create global logger instance."""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = ComprehensiveLogger(name=name)
    
    return _global_logger


def setup_logging(
    log_level: str = "INFO",
    log_dir: Optional[str] = None,
    enable_wandb: bool = False,
    enable_tensorboard: bool = False,
    wandb_project: Optional[str] = None,
    **kwargs
) -> ComprehensiveLogger:
    """Setup comprehensive logging system."""
    global _global_logger
    
    _global_logger = ComprehensiveLogger(
        log_level=log_level,
        log_dir=log_dir,
        enable_wandb=enable_wandb,
        enable_tensorboard=enable_tensorboard,
        wandb_project=wandb_project,
        **kwargs
    )
    
    return _global_logger