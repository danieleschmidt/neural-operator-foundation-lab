# Deployment Guide

This document provides comprehensive deployment instructions for Neural Operator Foundation Lab across different environments.

## 🐳 Container Deployment

### Local Development with Docker

```bash
# Build development image
docker build -t neural-operator-lab:dev .

# Run with mounted source code
docker run -it --rm \
  -v $(pwd):/app \
  -p 8888:8888 \
  -p 6006:6006 \
  neural-operator-lab:dev

# Or use docker-compose for full development stack
docker-compose up -d
```

### Production Container Deployment

```bash
# Build production image
docker build -t neural-operator-lab:prod --target production .

# Run production container
docker run -d \
  --name neural-operator-prod \
  -p 8080:8080 \
  -v /data:/data \
  -v /models:/models \
  -e NEURAL_OP_SECURE_MODE=true \
  neural-operator-lab:prod
```

### Multi-stage Docker Setup

The Dockerfile includes optimized multi-stage builds:
- `base`: Development environment with all tools
- `production`: Optimized runtime without dev dependencies  
- `final`: Minimal production image with security hardening

## ☸️ Kubernetes Deployment

### Basic Kubernetes Deployment

```yaml
# deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-operator-lab
spec:
  replicas: 3
  selector:
    matchLabels:
      app: neural-operator-lab
  template:
    metadata:
      labels:
        app: neural-operator-lab
    spec:
      containers:
      - name: neural-operator-lab
        image: ghcr.io/terragon-labs/neural-operator-foundation-lab:latest
        ports:
        - containerPort: 8080
        env:
        - name: NEURAL_OP_SECURE_MODE
          value: "true"
        resources:
          limits:
            memory: "2Gi"
            cpu: "1000m"
          requests:
            memory: "1Gi" 
            cpu: "500m"
        livenessProbe:
          httpGet:
            path: /health
            port: 8080
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /ready
            port: 8080
          initialDelaySeconds: 5
          periodSeconds: 5
```

### Service and Ingress

```yaml
# service.yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-operator-service
spec:
  selector:
    app: neural-operator-lab
  ports:
  - port: 80
    targetPort: 8080
  type: ClusterIP

---
# ingress.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-operator-ingress
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  tls:
  - hosts:
    - neural-operator.example.com
    secretName: neural-operator-tls
  rules:
  - host: neural-operator.example.com
    http:
      paths:
      - path: /
        pathType: Prefix
        backend:
          service:
            name: neural-operator-service
            port:
              number: 80
```

### GPU-enabled Deployment

```yaml
# gpu-deployment.yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-operator-gpu
spec:
  replicas: 1
  selector:
    matchLabels:
      app: neural-operator-gpu
  template:
    metadata:
      labels:
        app: neural-operator-gpu
    spec:
      containers:
      - name: neural-operator-lab
        image: ghcr.io/terragon-labs/neural-operator-foundation-lab:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: "8Gi"
            cpu: "4000m"
        env:
        - name: CUDA_VISIBLE_DEVICES
          value: "0"
        - name: NEURAL_OP_SECURE_MODE
          value: "true"
      nodeSelector:
        accelerator: nvidia-tesla-v100
```

## 🌩️ Cloud Deployment

### AWS ECS/Fargate

```json
{
  "family": "neural-operator-lab",
  "networkMode": "awsvpc",
  "requiresCompatibilities": ["FARGATE"],
  "cpu": "1024",
  "memory": "2048",
  "executionRoleArn": "arn:aws:iam::account:role/ecsTaskExecutionRole",
  "containerDefinitions": [
    {
      "name": "neural-operator-lab",
      "image": "ghcr.io/terragon-labs/neural-operator-foundation-lab:latest",
      "portMappings": [
        {
          "containerPort": 8080,
          "protocol": "tcp"
        }
      ],
      "environment": [
        {
          "name": "NEURAL_OP_SECURE_MODE",
          "value": "true"
        }
      ],
      "logConfiguration": {
        "logDriver": "awslogs",
        "options": {
          "awslogs-group": "/ecs/neural-operator-lab",
          "awslogs-region": "us-west-2",
          "awslogs-stream-prefix": "ecs"
        }
      }
    }
  ]
}
```

### Google Cloud Run

```yaml
# cloudrun.yaml
apiVersion: serving.knative.dev/v1
kind: Service
metadata:
  name: neural-operator-lab
  annotations:
    run.googleapis.com/ingress: all
spec:
  template:
    metadata:
      annotations:
        run.googleapis.com/cpu-throttling: "false"
        run.googleapis.com/memory: "2Gi"
        run.googleapis.com/cpu: "2"
    spec:
      containers:
      - image: ghcr.io/terragon-labs/neural-operator-foundation-lab:latest
        ports:
        - containerPort: 8080
        env:
        - name: NEURAL_OP_SECURE_MODE
          value: "true"
        resources:
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

### Azure Container Instances

```yaml
# azure-container.yaml
apiVersion: 2021-09-01
location: eastus
name: neural-operator-lab
properties:
  containers:
  - name: neural-operator-lab
    properties:
      image: ghcr.io/terragon-labs/neural-operator-foundation-lab:latest
      ports:
      - port: 8080
        protocol: TCP
      environmentVariables:
      - name: NEURAL_OP_SECURE_MODE
        value: true
      resources:
        requests:
          cpu: 1.0
          memoryInGb: 2.0
  osType: Linux
  restartPolicy: Always
  ipAddress:
    type: Public
    ports:
    - port: 8080
      protocol: TCP
```

## 🔧 Configuration Management

### Environment Variables

```bash
# Security Configuration
export NEURAL_OP_SECURE_MODE=true
export NEURAL_OP_VALIDATE_MODELS=true
export NEURAL_OP_SANDBOX_EXECUTION=true

# Performance Configuration
export CUDA_VISIBLE_DEVICES=0,1
export OMP_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Monitoring Configuration
export OTEL_EXPORTER_OTLP_ENDPOINT=http://jaeger:14268/api/traces
export OTEL_SERVICE_NAME=neural-operator-lab
```

### ConfigMap for Kubernetes

```yaml
# configmap.yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: neural-operator-config
data:
  NEURAL_OP_SECURE_MODE: "true"
  NEURAL_OP_VALIDATE_MODELS: "true"
  OMP_NUM_THREADS: "4"
  
---
# Apply to deployment
spec:
  template:
    spec:
      containers:
      - name: neural-operator-lab
        envFrom:
        - configMapRef:
            name: neural-operator-config
```

## 📊 Monitoring & Observability

### Prometheus Metrics

```yaml
# monitoring.yaml
apiVersion: v1
kind: Service
metadata:
  name: neural-operator-metrics
  labels:
    app: neural-operator-lab
spec:
  ports:
  - port: 9090
    name: metrics
  selector:
    app: neural-operator-lab

---
apiVersion: monitoring.coreos.com/v1
kind: ServiceMonitor
metadata:
  name: neural-operator-monitor
spec:
  selector:
    matchLabels:
      app: neural-operator-lab
  endpoints:
  - port: metrics
    interval: 30s
    path: /metrics
```

### Grafana Dashboard

```json
{
  "dashboard": {
    "title": "Neural Operator Lab Metrics",
    "panels": [
      {
        "title": "Training Loss",
        "type": "graph",
        "targets": [
          {
            "expr": "training_loss{job=\"neural-operator-lab\"}"
          }
        ]
      },
      {
        "title": "GPU Utilization",
        "type": "graph", 
        "targets": [
          {
            "expr": "gpu_utilization{job=\"neural-operator-lab\"}"
          }
        ]
      }
    ]
  }
}
```

## 🔒 Security Considerations

### Container Security

```dockerfile
# Security hardening in Dockerfile
FROM python:3.9-slim

# Update packages and remove package manager
RUN apt-get update && apt-get upgrade -y && \
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    rm -rf /usr/bin/apt*

# Create non-root user
RUN groupadd -r neural && useradd -r -g neural neural
USER neural

# Use read-only filesystem
VOLUME ["/tmp"]
```

### Network Policies

```yaml
# network-policy.yaml
apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: neural-operator-network-policy
spec:
  podSelector:
    matchLabels:
      app: neural-operator-lab
  policyTypes:
  - Ingress
  - Egress
  ingress:
  - from:
    - podSelector:
        matchLabels:
          app: nginx-ingress
    ports:
    - protocol: TCP
      port: 8080
  egress:
  - to: []
    ports:
    - protocol: TCP
      port: 443  # HTTPS only
```

## 🚀 CI/CD Integration

### GitHub Actions Deployment

```yaml
# deploy.yml
name: Deploy
on:
  push:
    branches: [main]
    
jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Deploy to Kubernetes
      uses: azure/k8s-deploy@v1
      with:
        manifests: |
          k8s/deployment.yaml
          k8s/service.yaml 
          k8s/ingress.yaml
        kubectl-version: 'latest'
```

### Helm Chart Deployment

```yaml
# values.yaml
replicaCount: 3
image:
  repository: ghcr.io/terragon-labs/neural-operator-foundation-lab
  tag: latest
  pullPolicy: Always

service:
  type: ClusterIP
  port: 80

ingress:
  enabled: true
  annotations:
    kubernetes.io/ingress.class: nginx
  hosts:
    - host: neural-operator.example.com
      paths: ["/"]

resources:
  limits:
    cpu: 2000m
    memory: 4Gi
  requests:
    cpu: 1000m
    memory: 2Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
```

## 📋 Deployment Checklist

### Pre-deployment

- [ ] Security scan completed (no high/critical vulnerabilities)
- [ ] All tests passing
- [ ] Configuration validated
- [ ] Resource requirements defined
- [ ] Monitoring configured
- [ ] Backup strategy in place

### Post-deployment

- [ ] Health checks responding
- [ ] Metrics being collected
- [ ] Logs being aggregated
- [ ] SSL/TLS certificates valid
- [ ] Performance baseline established
- [ ] Documentation updated

## 🆘 Troubleshooting

### Common Issues

**Container won't start:**
```bash
# Check logs
docker logs neural-operator-lab

# Check resource usage
docker stats neural-operator-lab

# Inspect container
docker inspect neural-operator-lab
```

**Kubernetes pod failing:**
```bash
# Check pod status
kubectl describe pod neural-operator-lab-xxx

# Check logs
kubectl logs neural-operator-lab-xxx

# Check events
kubectl get events --sort-by=.metadata.creationTimestamp
```

**Performance issues:**
```bash
# Check resource utilization
kubectl top pods
kubectl top nodes

# Check metrics
curl http://neural-operator-service/metrics
```

This deployment guide ensures secure, scalable, and maintainable deployments across various environments while maintaining observability and security best practices.