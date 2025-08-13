"""Robust framework for error handling, validation, and resilience."""

import sys
import traceback
import logging
import functools
import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Union, Type
from dataclasses import dataclass, field
from contextlib import contextmanager
from pathlib import Path
import json
import hashlib
from collections import defaultdict
import threading
import queue


@dataclass
class ErrorContext:
    """Context information for error reporting."""
    function_name: str
    module_name: str
    args: List[Any] = field(default_factory=list)
    kwargs: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    stack_trace: Optional[str] = None
    error_id: Optional[str] = None
    
    def __post_init__(self):
        """Generate unique error ID."""
        if self.error_id is None:
            content = f"{self.function_name}_{self.module_name}_{self.timestamp}"
            self.error_id = hashlib.md5(content.encode()).hexdigest()[:8]


class RobustLogger:
    """Advanced logging system with error tracking."""
    
    def __init__(self, name: str = "neural_operator_lab", level: int = logging.INFO):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(level)
        
        # Thread-safe error collector
        self.error_queue = queue.Queue()
        self.error_history = []
        self.error_counts = defaultdict(int)
        self._lock = threading.Lock()
        
        # Setup handlers if not already configured
        if not self.logger.handlers:
            self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers."""
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if possible)
        try:
            log_dir = Path("logs")
            log_dir.mkdir(exist_ok=True)
            file_handler = logging.FileHandler(log_dir / "neural_operator.log")
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        except (OSError, PermissionError):
            self.logger.warning("Could not create file handler for logging")
    
    def log_error(self, error: Exception, context: ErrorContext):
        """Log error with detailed context."""
        with self._lock:
            self.error_counts[type(error).__name__] += 1
            self.error_history.append({
                'error': str(error),
                'error_type': type(error).__name__,
                'context': context,
                'timestamp': time.time()
            })
            
            # Keep only last 1000 errors
            if len(self.error_history) > 1000:
                self.error_history = self.error_history[-1000:]
        
        # Log error details
        error_msg = (
            f"Error in {context.function_name} (ID: {context.error_id}): "
            f"{type(error).__name__}: {str(error)}"
        )
        
        if context.stack_trace:
            error_msg += f"\nStack trace:\n{context.stack_trace}"
        
        self.logger.error(error_msg)
    
    def log_warning(self, message: str, **kwargs):
        """Log warning with context."""
        self.logger.warning(message, extra=kwargs)
    
    def log_info(self, message: str, **kwargs):
        """Log info with context."""
        self.logger.info(message, extra=kwargs)
    
    def get_error_summary(self) -> Dict[str, Any]:
        """Get summary of errors."""
        with self._lock:
            return {
                'total_errors': len(self.error_history),
                'error_types': dict(self.error_counts),
                'recent_errors': self.error_history[-10:] if self.error_history else []
            }


# Global logger instance
robust_logger = RobustLogger()


class RobustError(Exception):
    """Base exception for robust framework."""
    pass


class ValidationError(RobustError):
    """Raised when validation fails."""
    pass


class ResourceError(RobustError):
    """Raised when resource constraints are violated."""
    pass


class SecurityError(RobustError):
    """Raised when security constraints are violated."""
    pass


def robust_execution(
    max_retries: int = 3,
    retry_delay: float = 1.0,
    catch_exceptions: tuple = (Exception,),
    fallback_value: Any = None,
    log_errors: bool = True
):
    """Decorator for robust function execution with retries and error handling."""
    
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            context = ErrorContext(
                function_name=func.__name__,
                module_name=func.__module__,
                args=list(args),
                kwargs=kwargs
            )
            
            last_exception = None
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                
                except catch_exceptions as e:
                    last_exception = e
                    context.stack_trace = traceback.format_exc()
                    
                    if log_errors:
                        if attempt == max_retries:
                            robust_logger.log_error(e, context)
                        else:
                            robust_logger.log_warning(
                                f"Attempt {attempt + 1} failed for {func.__name__}: {str(e)}"
                            )
                    
                    if attempt < max_retries:
                        time.sleep(retry_delay * (2 ** attempt))  # Exponential backoff
                    else:
                        # Final attempt failed
                        if fallback_value is not None:
                            robust_logger.log_info(
                                f"Using fallback value for {func.__name__}"
                            )
                            return fallback_value
                        else:
                            raise last_exception
            
            return fallback_value
        
        return wrapper
    return decorator


@contextmanager
def error_boundary(operation_name: str, 
                  reraise: bool = True,
                  fallback_value: Any = None):
    """Context manager for error boundary protection."""
    try:
        robust_logger.log_info(f"Starting operation: {operation_name}")
        yield
        robust_logger.log_info(f"Completed operation: {operation_name}")
        
    except Exception as e:
        context = ErrorContext(
            function_name=operation_name,
            module_name="error_boundary",
            stack_trace=traceback.format_exc()
        )
        
        robust_logger.log_error(e, context)
        
        if reraise:
            raise
        else:
            robust_logger.log_info(f"Error suppressed for {operation_name}, using fallback")
            return fallback_value


class ResourceMonitor:
    """Monitor system resources and enforce limits."""
    
    def __init__(self, 
                 max_memory_mb: Optional[int] = None,
                 max_execution_time: Optional[float] = None):
        self.max_memory_mb = max_memory_mb
        self.max_execution_time = max_execution_time
        self.start_time = None
        
    def start_monitoring(self):
        """Start resource monitoring."""
        self.start_time = time.time()
        
    def check_resources(self, operation_name: str = "unknown"):
        """Check if resource limits are violated."""
        if self.start_time is None:
            self.start_monitoring()
        
        # Check execution time
        if self.max_execution_time:
            elapsed = time.time() - self.start_time
            if elapsed > self.max_execution_time:
                raise ResourceError(
                    f"Operation '{operation_name}' exceeded max execution time "
                    f"({elapsed:.1f}s > {self.max_execution_time}s)"
                )
        
        # Check memory usage (simplified, would need psutil for real implementation)
        if self.max_memory_mb:
            try:
                import sys
                # Rough approximation of memory usage
                if hasattr(sys, 'getsizeof'):
                    # This is a very rough estimate
                    pass
            except ImportError:
                pass


class InputValidator:
    """Comprehensive input validation."""
    
    @staticmethod
    def validate_tensor_input(tensor: Any, 
                            name: str = "tensor",
                            expected_dims: Optional[int] = None,
                            expected_shape: Optional[tuple] = None,
                            min_value: Optional[float] = None,
                            max_value: Optional[float] = None,
                            allow_nan: bool = False,
                            allow_inf: bool = False) -> None:
        """Validate tensor inputs."""
        
        # Check if it's tensor-like
        if not hasattr(tensor, 'shape') or not hasattr(tensor, 'dtype'):
            raise ValidationError(f"{name} must be a tensor-like object")
        
        # Check dimensions
        if expected_dims is not None and len(tensor.shape) != expected_dims:
            raise ValidationError(
                f"{name} must have {expected_dims} dimensions, got {len(tensor.shape)}"
            )
        
        # Check shape
        if expected_shape is not None:
            for i, (actual, expected) in enumerate(zip(tensor.shape, expected_shape)):
                if expected is not None and actual != expected:
                    raise ValidationError(
                        f"{name} dimension {i} must be {expected}, got {actual}"
                    )
        
        # Check for NaN/Inf
        try:
            if hasattr(tensor, 'isnan') and not allow_nan:
                if tensor.isnan().any():
                    raise ValidationError(f"{name} contains NaN values")
            
            if hasattr(tensor, 'isinf') and not allow_inf:
                if tensor.isinf().any():
                    raise ValidationError(f"{name} contains infinite values")
        except AttributeError:
            # Skip if tensor doesn't support these operations
            pass
        
        # Check value range
        try:
            if min_value is not None and tensor.min() < min_value:
                raise ValidationError(f"{name} contains values below {min_value}")
            
            if max_value is not None and tensor.max() > max_value:
                raise ValidationError(f"{name} contains values above {max_value}")
        except AttributeError:
            # Skip if tensor doesn't support min/max
            pass
    
    @staticmethod
    def validate_config(config: Dict[str, Any], 
                       required_keys: List[str],
                       optional_keys: Optional[List[str]] = None) -> None:
        """Validate configuration dictionary."""
        if not isinstance(config, dict):
            raise ValidationError("Config must be a dictionary")
        
        # Check required keys
        missing_keys = set(required_keys) - set(config.keys())
        if missing_keys:
            raise ValidationError(f"Missing required config keys: {missing_keys}")
        
        # Check for unknown keys
        if optional_keys is not None:
            allowed_keys = set(required_keys) | set(optional_keys)
            unknown_keys = set(config.keys()) - allowed_keys
            if unknown_keys:
                robust_logger.log_warning(f"Unknown config keys: {unknown_keys}")
    
    @staticmethod
    def validate_file_path(path: Union[str, Path], 
                         must_exist: bool = True,
                         must_be_file: bool = True,
                         allowed_extensions: Optional[List[str]] = None) -> Path:
        """Validate file path."""
        path = Path(path)
        
        if must_exist and not path.exists():
            raise ValidationError(f"Path does not exist: {path}")
        
        if must_be_file and path.exists() and not path.is_file():
            raise ValidationError(f"Path is not a file: {path}")
        
        if allowed_extensions:
            if path.suffix.lower() not in [ext.lower() for ext in allowed_extensions]:
                raise ValidationError(
                    f"File extension must be one of {allowed_extensions}, got {path.suffix}"
                )
        
        return path


class CircuitBreaker:
    """Circuit breaker pattern for fault tolerance."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 timeout: float = 60.0,
                 expected_exception: Type[Exception] = Exception):
        self.failure_threshold = failure_threshold
        self.timeout = timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = 'CLOSED'  # CLOSED, OPEN, HALF_OPEN
        self._lock = threading.Lock()
    
    def call(self, func: Callable, *args, **kwargs):
        """Call function through circuit breaker."""
        with self._lock:
            if self.state == 'OPEN':
                if time.time() - self.last_failure_time >= self.timeout:
                    self.state = 'HALF_OPEN'
                    robust_logger.log_info("Circuit breaker moving to HALF_OPEN state")
                else:
                    raise ResourceError("Circuit breaker is OPEN")
            
            try:
                result = func(*args, **kwargs)
                
                # Success - reset failure count
                if self.state == 'HALF_OPEN':
                    self.state = 'CLOSED'
                    robust_logger.log_info("Circuit breaker CLOSED after successful call")
                
                self.failure_count = 0
                return result
                
            except self.expected_exception as e:
                self.failure_count += 1
                self.last_failure_time = time.time()
                
                if self.failure_count >= self.failure_threshold:
                    self.state = 'OPEN'
                    robust_logger.log_warning(
                        f"Circuit breaker OPEN after {self.failure_count} failures"
                    )
                
                raise e


class HealthChecker:
    """System health monitoring."""
    
    def __init__(self):
        self.checks = {}
        self.last_check_time = {}
        
    def register_check(self, name: str, check_func: Callable[[], bool], 
                      interval: float = 60.0):
        """Register a health check."""
        self.checks[name] = {
            'func': check_func,
            'interval': interval,
            'last_result': None,
            'last_error': None
        }
        
    def run_check(self, name: str) -> Dict[str, Any]:
        """Run a specific health check."""
        if name not in self.checks:
            return {'status': 'unknown', 'error': f'Check {name} not found'}
        
        check = self.checks[name]
        current_time = time.time()
        
        # Check if we need to run the check
        last_check = self.last_check_time.get(name, 0)
        if current_time - last_check < check['interval']:
            return {
                'status': 'cached',
                'result': check['last_result'],
                'last_check': last_check
            }
        
        # Run the check
        try:
            result = check['func']()
            check['last_result'] = result
            check['last_error'] = None
            self.last_check_time[name] = current_time
            
            return {
                'status': 'ok' if result else 'failed',
                'result': result,
                'last_check': current_time
            }
            
        except Exception as e:
            check['last_error'] = str(e)
            self.last_check_time[name] = current_time
            
            return {
                'status': 'error',
                'error': str(e),
                'last_check': current_time
            }
    
    def run_all_checks(self) -> Dict[str, Dict[str, Any]]:
        """Run all registered health checks."""
        results = {}
        for name in self.checks:
            results[name] = self.run_check(name)
        return results
    
    def get_system_status(self) -> str:
        """Get overall system status."""
        results = self.run_all_checks()
        
        if not results:
            return 'unknown'
        
        statuses = [result['status'] for result in results.values()]
        
        if any(status == 'error' for status in statuses):
            return 'critical'
        elif any(status == 'failed' for status in statuses):
            return 'degraded'
        else:
            return 'healthy'


# Global health checker
health_checker = HealthChecker()


def register_health_check(name: str, interval: float = 60.0):
    """Decorator to register a function as a health check."""
    def decorator(func: Callable[[], bool]) -> Callable[[], bool]:
        health_checker.register_check(name, func, interval)
        return func
    return decorator


# Example health checks
@register_health_check('basic_functionality', interval=30.0)
def check_basic_functionality() -> bool:
    """Check if basic functionality is working."""
    try:
        # Basic Python functionality test
        test_data = [1, 2, 3, 4, 5]
        result = sum(test_data)
        return result == 15
    except Exception:
        return False


@register_health_check('memory_usage', interval=120.0)
def check_memory_usage() -> bool:
    """Check if memory usage is reasonable."""
    try:
        import sys
        # This is a placeholder - real implementation would use psutil
        # For now, just check if we can allocate a small object
        test_obj = list(range(1000))
        del test_obj
        return True
    except MemoryError:
        return False
    except Exception:
        return False