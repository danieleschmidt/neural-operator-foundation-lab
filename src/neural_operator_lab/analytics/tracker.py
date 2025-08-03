"""Advanced metrics tracking and performance analysis."""

import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import time
import psutil
import threading
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
from pathlib import Path


@dataclass
class MetricPoint:
    """Single metric data point."""
    timestamp: float
    value: float
    metadata: Dict[str, Any] = field(default_factory=dict)


class MetricsTracker:
    """Advanced metrics tracking with real-time aggregation."""
    
    def __init__(self, window_size: int = 1000, save_interval: float = 30.0):
        """Initialize metrics tracker.
        
        Args:
            window_size: Maximum number of points to keep in memory
            save_interval: Seconds between automatic saves
        """
        self.window_size = window_size
        self.save_interval = save_interval
        
        # Metric storage
        self.metrics: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.aggregated_metrics: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        # System metrics
        self.system_metrics = {
            'cpu_percent': deque(maxlen=window_size),
            'memory_percent': deque(maxlen=window_size),
            'gpu_memory_used': deque(maxlen=window_size),
            'gpu_utilization': deque(maxlen=window_size)
        }
        
        # Auto-save setup
        self._save_path: Optional[Path] = None
        self._save_thread: Optional[threading.Thread] = None
        self._stop_saving = threading.Event()
        
        # Start system monitoring
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start background system metrics collection."""
        def monitor():
            while not self._stop_saving.is_set():
                timestamp = time.time()
                
                # CPU and Memory
                cpu_percent = psutil.cpu_percent()
                memory_percent = psutil.virtual_memory().percent
                
                self.system_metrics['cpu_percent'].append(
                    MetricPoint(timestamp, cpu_percent)
                )
                self.system_metrics['memory_percent'].append(
                    MetricPoint(timestamp, memory_percent)
                )
                
                # GPU metrics
                if torch.cuda.is_available():
                    gpu_memory = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
                    gpu_util = torch.cuda.utilization() if hasattr(torch.cuda, 'utilization') else 0
                    
                    self.system_metrics['gpu_memory_used'].append(
                        MetricPoint(timestamp, gpu_memory * 100)
                    )
                    self.system_metrics['gpu_utilization'].append(
                        MetricPoint(timestamp, gpu_util)
                    )
                
                time.sleep(1.0)  # Update every second
        
        self._monitor_thread = threading.Thread(target=monitor, daemon=True)
        self._monitor_thread.start()
    
    def add_metric(self, name: str, value: float, metadata: Optional[Dict[str, Any]] = None):
        """Add a metric point."""
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            metadata=metadata or {}
        )
        self.metrics[name].append(point)
        self._update_aggregation(name)
    
    def add_batch_metrics(self, metrics: Dict[str, float], metadata: Optional[Dict[str, Any]] = None):
        """Add multiple metrics at once."""
        timestamp = time.time()
        for name, value in metrics.items():
            point = MetricPoint(
                timestamp=timestamp,
                value=value,
                metadata=metadata or {}
            )
            self.metrics[name].append(point)
            self._update_aggregation(name)
    
    def _update_aggregation(self, metric_name: str):
        """Update aggregated statistics for a metric."""
        points = self.metrics[metric_name]
        if not points:
            return
        
        values = [p.value for p in points]
        
        self.aggregated_metrics[metric_name] = {
            'current': values[-1],
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'count': len(values)
        }
    
    def get_metric_summary(self, metric_name: str) -> Dict[str, Any]:
        """Get summary statistics for a metric."""
        if metric_name not in self.aggregated_metrics:
            return {}
        return self.aggregated_metrics[metric_name].copy()
    
    def get_metric_history(self, metric_name: str, 
                          last_n: Optional[int] = None) -> List[Tuple[float, float]]:
        """Get metric history as (timestamp, value) pairs."""
        if metric_name not in self.metrics:
            return []
        
        points = list(self.metrics[metric_name])
        if last_n:
            points = points[-last_n:]
        
        return [(p.timestamp, p.value) for p in points]
    
    def get_system_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary of system metrics."""
        summary = {}
        for metric_name, points in self.system_metrics.items():
            if points:
                values = [p.value for p in points]
                summary[metric_name] = {
                    'current': values[-1],
                    'mean': np.mean(values),
                    'max': np.max(values),
                    'min': np.min(values)
                }
        return summary
    
    def enable_auto_save(self, save_path: str):
        """Enable automatic saving of metrics."""
        self._save_path = Path(save_path)
        self._save_path.parent.mkdir(parents=True, exist_ok=True)
        
        def auto_save():
            while not self._stop_saving.wait(self.save_interval):
                self.save_metrics()
        
        self._save_thread = threading.Thread(target=auto_save, daemon=True)
        self._save_thread.start()
    
    def save_metrics(self, path: Optional[str] = None):
        """Save metrics to file."""
        save_path = Path(path) if path else self._save_path
        if not save_path:
            return
        
        data = {
            'metrics': {
                name: {
                    'points': [(p.timestamp, p.value, p.metadata) for p in points],
                    'aggregated': self.aggregated_metrics.get(name, {})
                }
                for name, points in self.metrics.items()
            },
            'system_metrics': {
                name: [(p.timestamp, p.value) for p in points]
                for name, points in self.system_metrics.items()
            },
            'metadata': {
                'window_size': self.window_size,
                'save_timestamp': time.time()
            }
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_metrics(self, path: str):
        """Load metrics from file."""
        with open(path, 'r') as f:
            data = json.load(f)
        
        # Restore metrics
        for name, metric_data in data['metrics'].items():
            self.metrics[name].clear()
            for timestamp, value, metadata in metric_data['points']:
                point = MetricPoint(timestamp, value, metadata)
                self.metrics[name].append(point)
            
            self.aggregated_metrics[name] = metric_data.get('aggregated', {})
        
        # Restore system metrics
        for name, points in data['system_metrics'].items():
            self.system_metrics[name].clear()
            for timestamp, value in points:
                point = MetricPoint(timestamp, value)
                self.system_metrics[name].append(point)
    
    def cleanup(self):
        """Clean up resources."""
        self._stop_saving.set()
        if self._save_thread:
            self._save_thread.join(timeout=1.0)
        if hasattr(self, '_monitor_thread'):
            self._monitor_thread.join(timeout=1.0)


class PerformanceAnalyzer:
    """Analyze model and training performance."""
    
    def __init__(self, tracker: MetricsTracker):
        self.tracker = tracker
    
    def analyze_training_efficiency(self) -> Dict[str, Any]:
        """Analyze training efficiency metrics."""
        # Get training metrics
        train_loss_history = self.tracker.get_metric_history('train_loss')
        epoch_time_history = self.tracker.get_metric_history('epoch_time')
        
        if not train_loss_history or not epoch_time_history:
            return {}
        
        # Convergence analysis
        train_losses = [loss for _, loss in train_loss_history]
        convergence_rate = self._calculate_convergence_rate(train_losses)
        
        # Time efficiency
        epoch_times = [time for _, time in epoch_time_history]
        time_stability = np.std(epoch_times) / np.mean(epoch_times) if epoch_times else 0
        
        # System utilization
        system_summary = self.tracker.get_system_summary()
        
        return {
            'convergence_rate': convergence_rate,
            'time_stability': time_stability,
            'avg_epoch_time': np.mean(epoch_times) if epoch_times else 0,
            'system_utilization': system_summary,
            'training_stability': self._assess_training_stability(train_losses)
        }
    
    def _calculate_convergence_rate(self, losses: List[float]) -> float:
        """Calculate convergence rate from loss history."""
        if len(losses) < 5:
            return 0.0
        
        # Fit exponential decay to recent losses
        recent_losses = losses[-min(50, len(losses)):]
        x = np.arange(len(recent_losses))
        
        try:
            # Log transform for exponential fit
            log_losses = np.log(np.maximum(recent_losses, 1e-10))
            # Linear fit to log(loss) vs epoch
            slope, _ = np.polyfit(x, log_losses, 1)
            return -slope  # Negative slope means convergence
        except:
            return 0.0
    
    def _assess_training_stability(self, losses: List[float]) -> Dict[str, float]:
        """Assess training stability metrics."""
        if len(losses) < 10:
            return {}
        
        # Moving window analysis
        window_size = min(10, len(losses) // 4)
        windows = [losses[i:i+window_size] for i in range(len(losses) - window_size + 1)]
        
        window_means = [np.mean(w) for w in windows]
        window_stds = [np.std(w) for w in windows]
        
        # Stability metrics
        mean_stability = np.std(window_means) / np.mean(window_means) if window_means else 0
        variance_stability = np.std(window_stds) / np.mean(window_stds) if window_stds else 0
        
        # Oscillation detection
        oscillations = self._detect_oscillations(losses)
        
        return {
            'mean_stability': mean_stability,
            'variance_stability': variance_stability,
            'oscillation_frequency': oscillations
        }
    
    def _detect_oscillations(self, losses: List[float]) -> float:
        """Detect oscillation frequency in loss."""
        if len(losses) < 20:
            return 0.0
        
        # Calculate relative changes
        changes = np.diff(losses)
        sign_changes = np.diff(np.sign(changes))
        
        # Count sign changes (peaks and valleys)
        oscillations = np.sum(np.abs(sign_changes) == 2)
        
        return oscillations / len(changes) if changes.size > 0 else 0.0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report."""
        training_analysis = self.analyze_training_efficiency()
        
        # Get all metric summaries
        metric_summaries = {}
        for metric_name in self.tracker.metrics.keys():
            metric_summaries[metric_name] = self.tracker.get_metric_summary(metric_name)
        
        return {
            'timestamp': time.time(),
            'training_efficiency': training_analysis,
            'metric_summaries': metric_summaries,
            'system_performance': self.tracker.get_system_summary(),
            'recommendations': self._generate_recommendations(training_analysis)
        }
    
    def _generate_recommendations(self, analysis: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations."""
        recommendations = []
        
        if not analysis:
            return recommendations
        
        # Training efficiency recommendations
        if analysis.get('convergence_rate', 0) < 0.01:
            recommendations.append("Consider increasing learning rate or using learning rate scheduling")
        
        if analysis.get('time_stability', 1.0) > 0.3:
            recommendations.append("High variation in epoch times - check for data loading bottlenecks")
        
        # System utilization recommendations
        system_util = analysis.get('system_utilization', {})
        if system_util.get('gpu_utilization', {}).get('mean', 0) < 50:
            recommendations.append("Low GPU utilization - consider increasing batch size or model complexity")
        
        if system_util.get('memory_percent', {}).get('max', 0) > 90:
            recommendations.append("High memory usage - consider reducing batch size or enabling gradient checkpointing")
        
        # Training stability recommendations
        stability = analysis.get('training_stability', {})
        if stability.get('oscillation_frequency', 0) > 0.3:
            recommendations.append("High loss oscillations detected - consider reducing learning rate")
        
        return recommendations