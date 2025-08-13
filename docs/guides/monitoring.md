# Monitoring & Observability Guide

Complete guide to monitoring and observability for Neural Operator Foundation Lab.

## 🔍 Overview

The lab includes comprehensive monitoring capabilities:
- Performance metrics collection
- Health checking and alerting
- Audit logging for compliance
- Real-time system metrics
- Error tracking and analysis

## 📊 Metrics Collection

### Performance Profiler

```python
from neural_operator_lab.performance_optimizer import PerformanceProfiler

# Initialize profiler
profiler = PerformanceProfiler()

# Profile operations
with profiler.profile('training_step'):
    model.train_step(batch)

# Get statistics
stats = profiler.get_operation_stats('training_step')
print(f"Average execution time: {stats['avg_execution_time']:.3f}s")
```

### System Metrics

```python
from neural_operator_lab.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()
collector.start()  # Start background collection

# Get current metrics
metrics = collector.get_current_metrics()
print(f"CPU Usage: {metrics['cpu_percent']}%")
print(f"Memory Usage: {metrics['memory_usage']}MB")
```

### Custom Metrics

```python
from neural_operator_lab.monitoring.metrics_collector import MetricsCollector

collector = MetricsCollector()

# Record custom metrics
collector.record_metric('model_accuracy', 0.95)
collector.record_metric('inference_latency', 0.032)

# Get metric history
accuracy_history = collector.get_metric_history('model_accuracy')
```

## 🏥 Health Checking

### Automatic Health Monitoring

```python
from neural_operator_lab.monitoring.health_checker import HealthChecker

health_checker = HealthChecker()

# Add custom health checks
def model_health_check():
    """Check if model is responding correctly."""
    try:
        test_input = generate_test_input()
        prediction = model.predict(test_input)
        return len(prediction) > 0
    except Exception as e:
        return False

health_checker.add_check('model_responsiveness', model_health_check)

# Get overall health status
health_status = health_checker.get_health_status()
print(f"System Health: {health_status['status']}")
```

### Health Check Endpoints

The system exposes health check endpoints:

```bash
# Basic health check
curl http://localhost:8000/health

# Detailed health information
curl http://localhost:8000/health/detailed

# Component-specific health
curl http://localhost:8000/health/model
```

## 📝 Comprehensive Logging

### Structured Logging

```python
from neural_operator_lab.monitoring.comprehensive_logging import StructuredLogger

logger = StructuredLogger('neural_operator')

# Log with structured data
logger.log_info(
    "Training completed",
    extra={
        'epoch': 100,
        'accuracy': 0.95,
        'loss': 0.032,
        'duration_seconds': 3600
    }
)

# Log errors with context
logger.log_error(
    "Model prediction failed",
    error=exception,
    extra={'input_shape': input.shape}
)
```

### Audit Logging

```python
from neural_operator_lab.security.audit import AuditLogger

audit = AuditLogger()

# Log security events
audit.log_event(
    event_type='model_access',
    user_id='user123',
    action='predict',
    resource='physics_model_v1',
    result='success'
)

# Log data access
audit.log_data_access(
    user_id='researcher01',
    data_type='training_data',
    action='read',
    record_count=1000
)
```

## 📈 Performance Monitoring

### Real-time Performance Dashboard

```python
from neural_operator_lab.performance_optimizer import get_performance_stats

# Get comprehensive performance stats
stats = get_performance_stats()

print("=== PERFORMANCE DASHBOARD ===")
print(f"Profiler Summary:")
for operation, data in stats['profiler']['top_operations']:
    print(f"  {operation}: {data['total_time']:.2f}s total")

print(f"\\nParallel Processing:")
print(f"  Success Rate: {stats['parallel_processor']['success_rate']:.1%}")
print(f"  Optimal Workers: {stats['parallel_processor']['optimal_workers']}")

print(f"\\nAuto-scaler:")
print(f"  Current Resources: {stats['autoscaler']['current_resources']}")
print(f"  Recommendation: {stats['autoscaler']['recommendation']}")
```

### Cache Performance

```python
from neural_operator_lab.intelligent_cache import AdaptiveCache

cache = AdaptiveCache()

# Monitor cache performance
cache_stats = cache.get_statistics()
print(f"Cache Hit Rate: {cache_stats['hit_rate']:.1%}")
print(f"Memory Usage: {cache_stats['memory_usage']}MB")
print(f"Eviction Rate: {cache_stats['eviction_rate']:.2f}/sec")
```

## 🚨 Alerting & Notifications

### Error Rate Monitoring

```python
from neural_operator_lab.monitoring.advanced_error_handler import ErrorRateMonitor

error_monitor = ErrorRateMonitor()

# Configure thresholds
error_monitor.set_threshold('error_rate', 0.05)  # 5% error rate
error_monitor.set_threshold('critical_errors', 10)  # 10 critical errors

# Check for alerts
alerts = error_monitor.check_alerts()
if alerts:
    for alert in alerts:
        print(f"ALERT: {alert['type']} - {alert['message']}")
```

### Performance Degradation Alerts

```python
from neural_operator_lab.performance_optimizer import PerformanceProfiler

profiler = PerformanceProfiler()

# Identify bottlenecks
bottlenecks = profiler.identify_bottlenecks(threshold_seconds=2.0)

if bottlenecks:
    print("🚨 PERFORMANCE ALERTS:")
    for bottleneck in bottlenecks[:5]:
        print(f"  {bottleneck['operation']}: {bottleneck['avg_time']:.2f}s avg")
```

## 📊 Monitoring Dashboards

### Grafana Integration

Create custom Grafana dashboards using metrics from the system:

```yaml
# grafana-dashboard.json
{
  "dashboard": {
    "title": "Neural Operator Lab - Performance",
    "panels": [
      {
        "title": "Model Inference Time",
        "targets": [{
          "expr": "neural_operator_inference_duration_seconds"
        }]
      },
      {
        "title": "Cache Hit Rate",
        "targets": [{
          "expr": "neural_operator_cache_hit_rate"
        }]
      }
    ]
  }
}
```

### Prometheus Metrics Export

```python
from neural_operator_lab.monitoring.metrics_collector import PrometheusExporter

exporter = PrometheusExporter()
exporter.start_server(port=8001)

# Metrics available at http://localhost:8001/metrics
```

## 🔧 Configuration

### Monitoring Configuration

```python
# config/monitoring.yaml
monitoring:
  metrics:
    enabled: true
    collection_interval: 30  # seconds
    retention_days: 30
  
  health_checks:
    enabled: true
    check_interval: 60  # seconds
    
  alerting:
    error_rate_threshold: 0.05
    response_time_threshold: 2.0
    
  logging:
    level: INFO
    structured: true
    audit_enabled: true
```

### Load Configuration

```python
from neural_operator_lab.config_manager import ConfigManager

config = ConfigManager()
monitoring_config = config.get_monitoring_config()

# Apply monitoring settings
if monitoring_config['metrics']['enabled']:
    start_metrics_collection()
```

## 🚀 Production Monitoring

### Container Monitoring

```bash
# Docker health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \\
  CMD curl -f http://localhost:8000/health || exit 1

# Kubernetes liveness probe
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 30
  periodSeconds: 10
```

### Log Aggregation

```yaml
# logging-config.yaml
logging:
  handlers:
    - type: file
      filename: /var/log/neural-operator.log
      rotation: daily
    - type: elasticsearch
      host: elasticsearch.monitoring.svc.cluster.local
      index: neural-operator-logs
```

## 📋 Monitoring Checklist

### Daily Monitoring Tasks
- [ ] Check system health status
- [ ] Review error rates and patterns
- [ ] Monitor resource utilization
- [ ] Validate cache performance
- [ ] Check security audit logs

### Weekly Monitoring Tasks
- [ ] Analyze performance trends
- [ ] Review capacity planning metrics
- [ ] Update alerting thresholds
- [ ] Validate backup and recovery
- [ ] Security compliance review

### Monthly Monitoring Tasks
- [ ] Comprehensive performance analysis
- [ ] Cost optimization review
- [ ] Monitoring system updates
- [ ] Documentation updates
- [ ] Disaster recovery testing

## 📚 References

- [Metrics Collector API](../api/monitoring.md)
- [Health Checker Configuration](../api/health.md)
- [Audit Logging Guide](security.md#audit-logging)
- [Performance Optimization](performance.md)

---

**Monitoring is essential for maintaining system reliability and performance.** 🔍