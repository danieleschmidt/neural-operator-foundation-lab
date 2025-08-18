"""Global telemetry and monitoring for neural operator deployments."""

import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from collections import defaultdict, deque
import logging
import uuid


class TelemetryLevel(Enum):
    """Telemetry collection levels."""
    MINIMAL = "minimal"
    STANDARD = "standard"
    DETAILED = "detailed"
    DEBUG = "debug"


class MetricType(Enum):
    """Types of metrics collected."""
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"
    SUMMARY = "summary"


@dataclass
class MetricPoint:
    """Individual metric data point."""
    timestamp: float
    value: float
    tags: Dict[str, str]
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


@dataclass
class TelemetryConfig:
    """Configuration for telemetry collection."""
    enabled: bool = True
    level: TelemetryLevel = TelemetryLevel.STANDARD
    collection_interval_sec: int = 30
    batch_size: int = 100
    retention_days: int = 30
    enable_real_time_alerts: bool = True
    enable_compliance_tracking: bool = True
    enable_performance_profiling: bool = True
    export_endpoints: List[str] = None
    privacy_mode: bool = True
    
    def __post_init__(self):
        if self.export_endpoints is None:
            self.export_endpoints = []


class ComplianceTracker:
    """Track compliance-related metrics and events."""
    
    def __init__(self):
        self.events = deque(maxlen=10000)
        self.violations = deque(maxlen=1000)
        self.access_logs = deque(maxlen=5000)
        
    def log_data_access(self, user_id: str, data_type: str, purpose: str, region: str):
        """Log data access for compliance auditing."""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'event_type': 'data_access',
            'user_id': user_id,
            'data_type': data_type,
            'purpose': purpose,
            'region': region,
            'lawful_basis': self._determine_lawful_basis(purpose)
        }
        self.access_logs.append(event)
    
    def log_consent_change(self, user_id: str, consent_type: str, granted: bool, region: str):
        """Log consent changes."""
        event = {
            'event_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'event_type': 'consent_change',
            'user_id': user_id,
            'consent_type': consent_type,
            'granted': granted,
            'region': region
        }
        self.events.append(event)
    
    def log_violation(self, violation_type: str, description: str, severity: str, region: str):
        """Log compliance violations."""
        violation = {
            'violation_id': str(uuid.uuid4()),
            'timestamp': time.time(),
            'type': violation_type,
            'description': description,
            'severity': severity,
            'region': region,
            'status': 'open'
        }
        self.violations.append(violation)
    
    def _determine_lawful_basis(self, purpose: str) -> str:
        """Determine lawful basis for data processing."""
        purpose_mapping = {
            'model_training': 'legitimate_interests',
            'performance_optimization': 'legitimate_interests',
            'security_monitoring': 'legitimate_interests',
            'user_personalization': 'consent',
            'marketing': 'consent',
            'legal_compliance': 'legal_obligation',
            'safety_monitoring': 'vital_interests'
        }
        return purpose_mapping.get(purpose, 'legitimate_interests')
    
    def get_compliance_report(self, region: str = None) -> Dict[str, Any]:
        """Generate compliance report."""
        events = list(self.events)
        violations = list(self.violations)
        access_logs = list(self.access_logs)
        
        if region:
            events = [e for e in events if e.get('region') == region]
            violations = [v for v in violations if v.get('region') == region]
            access_logs = [a for a in access_logs if a.get('region') == region]
        
        return {
            'report_generated_at': time.time(),
            'region': region or 'global',
            'total_events': len(events),
            'total_violations': len(violations),
            'total_access_logs': len(access_logs),
            'open_violations': len([v for v in violations if v.get('status') == 'open']),
            'consent_events': len([e for e in events if e.get('event_type') == 'consent_change']),
            'data_access_events': len([e for e in events if e.get('event_type') == 'data_access']),
            'lawful_basis_distribution': self._get_lawful_basis_distribution(access_logs),
            'recent_violations': violations[-10:] if violations else []
        }
    
    def _get_lawful_basis_distribution(self, access_logs: List[Dict]) -> Dict[str, int]:
        """Get distribution of lawful bases used."""
        distribution = defaultdict(int)
        for log in access_logs:
            basis = log.get('lawful_basis', 'unknown')
            distribution[basis] += 1
        return dict(distribution)


class PerformanceProfiler:
    """Advanced performance profiling for neural operators."""
    
    def __init__(self):
        self.profiles = deque(maxlen=1000)
        self.active_traces = {}
        
    def start_trace(self, trace_id: str, operation: str, region: str) -> str:
        """Start a performance trace."""
        if not trace_id:
            trace_id = str(uuid.uuid4())
        
        self.active_traces[trace_id] = {
            'trace_id': trace_id,
            'operation': operation,
            'region': region,
            'start_time': time.time(),
            'spans': []
        }
        
        return trace_id
    
    def add_span(self, trace_id: str, span_name: str, duration_ms: float, metadata: Dict[str, Any] = None):
        """Add a span to an active trace."""
        if trace_id in self.active_traces:
            span = {
                'span_id': str(uuid.uuid4()),
                'name': span_name,
                'duration_ms': duration_ms,
                'timestamp': time.time(),
                'metadata': metadata or {}
            }
            self.active_traces[trace_id]['spans'].append(span)
    
    def finish_trace(self, trace_id: str, success: bool = True, error: str = None):
        """Finish a performance trace."""
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        trace['end_time'] = time.time()
        trace['duration_ms'] = (trace['end_time'] - trace['start_time']) * 1000
        trace['success'] = success
        
        if error:
            trace['error'] = error
        
        self.profiles.append(trace)
        del self.active_traces[trace_id]
    
    def get_performance_summary(self, region: str = None, operation: str = None) -> Dict[str, Any]:
        """Get performance summary statistics."""
        profiles = list(self.profiles)
        
        # Filter by region and operation if specified
        if region:
            profiles = [p for p in profiles if p.get('region') == region]
        if operation:
            profiles = [p for p in profiles if p.get('operation') == operation]
        
        if not profiles:
            return {'no_data': True}
        
        durations = [p['duration_ms'] for p in profiles]
        success_rate = len([p for p in profiles if p.get('success', True)]) / len(profiles)
        
        return {
            'total_traces': len(profiles),
            'avg_duration_ms': sum(durations) / len(durations),
            'min_duration_ms': min(durations),
            'max_duration_ms': max(durations),
            'p95_duration_ms': self._percentile(durations, 95),
            'p99_duration_ms': self._percentile(durations, 99),
            'success_rate': success_rate,
            'error_rate': 1 - success_rate,
            'recent_errors': [p.get('error') for p in profiles[-10:] if p.get('error')],
            'operation_breakdown': self._get_operation_breakdown(profiles)
        }
    
    def _percentile(self, data: List[float], percentile: int) -> float:
        """Calculate percentile of data."""
        if not data:
            return 0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile / 100)
        return sorted_data[min(index, len(sorted_data) - 1)]
    
    def _get_operation_breakdown(self, profiles: List[Dict]) -> Dict[str, Dict[str, float]]:
        """Get breakdown of performance by operation type."""
        breakdown = defaultdict(list)
        
        for profile in profiles:
            operation = profile.get('operation', 'unknown')
            breakdown[operation].append(profile['duration_ms'])
        
        result = {}
        for operation, durations in breakdown.items():
            result[operation] = {
                'count': len(durations),
                'avg_duration_ms': sum(durations) / len(durations),
                'max_duration_ms': max(durations)
            }
        
        return result


class GlobalTelemetryCollector:
    """Global telemetry collection and aggregation."""
    
    def __init__(self, config: TelemetryConfig):
        self.config = config
        self.metrics = defaultdict(lambda: deque(maxlen=10000))
        self.compliance_tracker = ComplianceTracker()
        self.performance_profiler = PerformanceProfiler()
        self.alert_callbacks = []
        self.logger = logging.getLogger(__name__)
        
        # Collection state
        self.collection_active = False
        self.collection_thread = None
        self.last_export = 0
        
        # Privacy controls
        self.anonymization_enabled = config.privacy_mode
        self.data_retention_sec = config.retention_days * 24 * 3600
    
    def start_collection(self):
        """Start telemetry collection."""
        if self.collection_active:
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        self.logger.info("Started global telemetry collection")
    
    def stop_collection(self):
        """Stop telemetry collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        self.logger.info("Stopped global telemetry collection")
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                # Collect system metrics
                if self.config.level in [TelemetryLevel.STANDARD, TelemetryLevel.DETAILED, TelemetryLevel.DEBUG]:
                    self._collect_system_metrics()
                
                # Collect performance metrics
                if self.config.enable_performance_profiling:
                    self._collect_performance_metrics()
                
                # Clean up old data
                self._cleanup_old_data()
                
                # Export metrics if needed
                current_time = time.time()
                if current_time - self.last_export > self.config.collection_interval_sec:
                    await self._export_metrics()
                    self.last_export = current_time
                
                time.sleep(self.config.collection_interval_sec)
                
            except Exception as e:
                self.logger.error(f"Telemetry collection error: {e}")
                time.sleep(self.config.collection_interval_sec)
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            import psutil
            
            # CPU and memory metrics
            cpu_percent = psutil.cpu_percent()
            memory = psutil.virtual_memory()
            
            timestamp = time.time()
            
            self.record_metric('system.cpu.usage', cpu_percent, {
                'unit': 'percent',
                'region': 'global'
            })
            
            self.record_metric('system.memory.usage', memory.percent, {
                'unit': 'percent',
                'region': 'global'
            })
            
            self.record_metric('system.memory.available', memory.available / (1024**3), {
                'unit': 'gb',
                'region': 'global'
            })
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric('system.disk.usage', disk.percent, {
                'unit': 'percent',
                'region': 'global'
            })
            
        except ImportError:
            # psutil not available, skip system metrics
            pass
        except Exception as e:
            self.logger.warning(f"System metrics collection failed: {e}")
    
    def _collect_performance_metrics(self):
        """Collect performance-related metrics."""
        # Get recent performance data
        summary = self.performance_profiler.get_performance_summary()
        
        if not summary.get('no_data'):
            timestamp = time.time()
            
            self.record_metric('performance.avg_latency', summary.get('avg_duration_ms', 0), {
                'unit': 'ms',
                'region': 'global'
            })
            
            self.record_metric('performance.p95_latency', summary.get('p95_duration_ms', 0), {
                'unit': 'ms',
                'region': 'global'
            })
            
            self.record_metric('performance.success_rate', summary.get('success_rate', 1.0), {
                'unit': 'ratio',
                'region': 'global'
            })
    
    def _cleanup_old_data(self):
        """Clean up old telemetry data."""
        current_time = time.time()
        cutoff_time = current_time - self.data_retention_sec
        
        # Clean up metrics
        for metric_name, points in self.metrics.items():
            # Remove old points
            while points and points[0].timestamp < cutoff_time:
                points.popleft()
    
    async def _export_metrics(self):
        """Export metrics to configured endpoints."""
        if not self.config.export_endpoints:
            return
        
        # Prepare metrics for export
        export_data = {
            'timestamp': time.time(),
            'metrics': self._serialize_metrics(),
            'compliance_report': self.compliance_tracker.get_compliance_report(),
            'performance_summary': self.performance_profiler.get_performance_summary()
        }
        
        # Apply privacy controls
        if self.anonymization_enabled:
            export_data = self._anonymize_data(export_data)
        
        # Export to each endpoint
        for endpoint in self.config.export_endpoints:
            try:
                await self._send_to_endpoint(endpoint, export_data)
            except Exception as e:
                self.logger.error(f"Failed to export to {endpoint}: {e}")
    
    def _serialize_metrics(self) -> Dict[str, List[Dict]]:
        """Serialize metrics for export."""
        serialized = {}
        
        for metric_name, points in self.metrics.items():
            serialized[metric_name] = [
                {
                    'timestamp': point.timestamp,
                    'value': point.value,
                    'tags': point.tags,
                    'metadata': point.metadata
                }
                for point in list(points)[-self.config.batch_size:]  # Last N points
            ]
        
        return serialized
    
    def _anonymize_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply anonymization to exported data."""
        # Remove or hash personally identifiable information
        anonymized = data.copy()
        
        # Anonymize compliance data
        if 'compliance_report' in anonymized:
            compliance = anonymized['compliance_report']
            # Remove specific user identifiers while keeping aggregate stats
            compliance.pop('recent_violations', None)
        
        return anonymized
    
    async def _send_to_endpoint(self, endpoint: str, data: Dict[str, Any]):
        """Send data to telemetry endpoint."""
        # In production, this would send via HTTP/gRPC to monitoring systems
        # For now, just log the export
        self.logger.debug(f"Exporting telemetry to {endpoint}: {len(data.get('metrics', {}))} metrics")
    
    def record_metric(self, name: str, value: float, tags: Dict[str, str] = None, metadata: Dict[str, Any] = None):
        """Record a metric point."""
        if not self.config.enabled:
            return
        
        point = MetricPoint(
            timestamp=time.time(),
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics[name].append(point)
        
        # Check for alerts
        self._check_alerts(name, value, tags or {})
    
    def _check_alerts(self, metric_name: str, value: float, tags: Dict[str, str]):
        """Check if metric triggers any alerts."""
        if not self.config.enable_real_time_alerts:
            return
        
        # Define alert thresholds
        alert_thresholds = {
            'system.cpu.usage': 90.0,
            'system.memory.usage': 90.0,
            'performance.avg_latency': 1000.0,  # 1 second
            'performance.error_rate': 0.1  # 10%
        }
        
        threshold = alert_thresholds.get(metric_name)
        if threshold and value > threshold:
            alert = {
                'alert_id': str(uuid.uuid4()),
                'timestamp': time.time(),
                'metric': metric_name,
                'value': value,
                'threshold': threshold,
                'tags': tags,
                'severity': 'warning'
            }
            
            # Trigger alert callbacks
            for callback in self.alert_callbacks:
                try:
                    callback(alert)
                except Exception as e:
                    self.logger.error(f"Alert callback failed: {e}")
    
    def add_alert_callback(self, callback: Callable[[Dict[str, Any]], None]):
        """Add callback for real-time alerts."""
        self.alert_callbacks.append(callback)
    
    def get_metrics_summary(self, metric_name: str = None, region: str = None, duration_sec: int = 3600) -> Dict[str, Any]:
        """Get summary of collected metrics."""
        current_time = time.time()
        start_time = current_time - duration_sec
        
        summary = {}
        
        metrics_to_analyze = [metric_name] if metric_name else list(self.metrics.keys())
        
        for name in metrics_to_analyze:
            points = self.metrics.get(name, deque())
            
            # Filter by time and region
            filtered_points = [
                p for p in points
                if p.timestamp >= start_time and (not region or p.tags.get('region') == region)
            ]
            
            if filtered_points:
                values = [p.value for p in filtered_points]
                summary[name] = {
                    'count': len(values),
                    'avg': sum(values) / len(values),
                    'min': min(values),
                    'max': max(values),
                    'latest': values[-1],
                    'trend': 'increasing' if len(values) > 1 and values[-1] > values[0] else 'stable'
                }
        
        return summary
    
    def get_global_dashboard_data(self) -> Dict[str, Any]:
        """Get comprehensive data for global monitoring dashboard."""
        return {
            'overview': {
                'collection_active': self.collection_active,
                'total_metrics': len(self.metrics),
                'total_data_points': sum(len(points) for points in self.metrics.values()),
                'last_export': self.last_export,
                'config': asdict(self.config)
            },
            'metrics_summary': self.get_metrics_summary(),
            'compliance_report': self.compliance_tracker.get_compliance_report(),
            'performance_summary': self.performance_profiler.get_performance_summary(),
            'alerts': {
                'enabled': self.config.enable_real_time_alerts,
                'callback_count': len(self.alert_callbacks)
            }
        }