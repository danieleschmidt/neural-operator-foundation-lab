# Neural Operator Foundation Lab - Production Deployment Guide

## 🚀 Quick Start Production Deployment

### Prerequisites

```bash
# Required
- Python 3.9+
- Docker 20.10+
- Kubernetes 1.20+

# Optional but recommended  
- NVIDIA GPU with CUDA 11.0+
- 16GB+ RAM
- 100GB+ storage
```

## 📦 Installation Options

### Option 1: Standard Installation
```bash
git clone https://github.com/terragon-labs/neural-operator-foundation-lab.git
cd neural-operator-foundation-lab

# Create environment
python -m venv neural-ops-env
source neural-ops-env/bin/activate  # Linux/Mac
# or: neural-ops-env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Install package
pip install -e .
```

### Option 2: Docker Deployment
```bash
# Build container
docker build -t neural-operator-lab:latest .

# Run with GPU support (recommended)
docker run --gpus all -p 8080:8080 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  neural-operator-lab:latest

# Run CPU-only
docker run -p 8080:8080 \
  -v $(pwd)/data:/workspace/data \
  -v $(pwd)/outputs:/workspace/outputs \
  neural-operator-lab:latest
```

### Option 3: Kubernetes Production Deployment
```bash
# Deploy namespace and resources
kubectl apply -f deployment/kubernetes/

# Check deployment status
kubectl get pods -n neural-operator-lab

# Access service
kubectl port-forward svc/neural-operator-service 8080:80 -n neural-operator-lab
```

## 🌍 Cloud Deployment

### AWS Deployment
```bash
# Deploy infrastructure
aws cloudformation create-stack \
  --stack-name neural-operator-infra \
  --template-body file://deployment/aws/cloudformation/neural-operator-infrastructure.yaml \
  --capabilities CAPABILITY_NAMED_IAM

# Deploy application
kubectl apply -f deployment/kubernetes/
```

### Azure Deployment
```bash
# Create resource group
az group create --name neural-operator-rg --location eastus

# Deploy AKS cluster
az aks create \
  --resource-group neural-operator-rg \
  --name neural-operator-cluster \
  --node-count 3 \
  --enable-addons gpu

# Deploy application
kubectl apply -f deployment/kubernetes/
```

### Google Cloud Deployment
```bash
# Create GKE cluster
gcloud container clusters create neural-operator-cluster \
  --zone=us-central1-a \
  --machine-type=n1-standard-4 \
  --num-nodes=3 \
  --enable-gpu

# Deploy application  
kubectl apply -f deployment/kubernetes/
```

## ⚙️ Configuration

### Environment Variables
```bash
# Core settings
export NEURAL_OP_EXPERIMENT_NAME="production_experiment"
export NEURAL_OP_SEED=42
export NEURAL_OP_BATCH_SIZE=32
export NEURAL_OP_LEARNING_RATE=1e-4

# Distributed settings
export NEURAL_OP_DISTRIBUTED=true
export NEURAL_OP_ZERO_OPTIMIZATION=true

# Security settings
export NEURAL_OP_SECURE_MODE=true
export NEURAL_OP_VALIDATE_MODELS=true
export NEURAL_OP_SANDBOX_EXECUTION=true
```

### Configuration File
```yaml
# config/production.yaml
experiment:
  name: "production_neural_operators"
  description: "Production deployment of neural operator training"
  log_level: "INFO"

training:
  epochs: 1000
  batch_size: 64
  learning_rate: 1e-4
  mixed_precision: true
  gradient_clipping: 1.0

model:
  model_type: "transformer_neural_operator"  
  hidden_dim: 512
  num_layers: 12
  num_heads: 16

distributed:
  enabled: true
  backend: "nccl"
  zero_optimization: true
  zero_stage: 2
```

## 🔧 Advanced Configuration

### Auto-Scaling Setup
```python
from neural_operator_lab.scaling import create_training_auto_scaler

# Create auto-scaler for training
scaler = create_training_auto_scaler(trainer)
scaler.start()

# Configure scaling policy
scaler.update_policy(
    check_interval=30.0,
    max_batch_size=256,
    aggressive_scaling=True
)
```

### Performance Optimization
```python
from neural_operator_lab.optimization import auto_optimize_training

# Apply automatic optimizations
trainer, dataloader = auto_optimize_training(
    trainer, 
    dataloader,
    PerformanceConfig(
        mixed_precision=True,
        gradient_checkpointing=True,
        num_workers=8,
        enable_cache=True
    )
)
```

### Health Monitoring
```python
from neural_operator_lab.monitoring import create_training_health_monitor

# Start health monitoring
health_monitor = create_training_health_monitor(trainer, check_interval=30.0)

# Export health report
health_monitor.export_health_report("health_report.json", hours=24)
```

## 📊 Monitoring & Observability

### Health Checks
```bash
# Check application health
curl http://localhost:8080/health

# Check detailed metrics
curl http://localhost:8080/metrics

# Check system status
curl http://localhost:8080/status
```

### Prometheus Integration
```yaml
# monitoring/prometheus-config.yaml
global:
  scrape_interval: 15s

scrape_configs:
  - job_name: 'neural-operator-lab'
    static_configs:
      - targets: ['neural-operator-service:8080']
```

### Grafana Dashboard
```bash
# Import dashboard
kubectl apply -f monitoring/grafana-dashboard.json
```

## 🔒 Security Configuration

### Production Security Settings
```python
from neural_operator_lab import configure_security

# Enable all security features
configure_security(
    validate_models=True,
    max_model_size="2GB", 
    secure_mode=True,
    sandbox_execution=True,
    allowed_file_types=[".pt", ".pth", ".h5"]
)
```

### SSL/TLS Configuration
```yaml
# ingress-tls.yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: neural-operator-ingress
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
spec:
  tls:
    - hosts:
        - neural-operator.yourdomain.com
      secretName: neural-operator-tls
  rules:
    - host: neural-operator.yourdomain.com
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

## 🎯 Production Use Cases

### Research Institution Deployment
```python
# research_config.py
from neural_operator_lab.research import ExperimentManager

# Set up research campaign
experiment_manager = ExperimentManager(output_dir="research_results")

# Configure hypothesis testing
experiments = [
    create_fno_experiment(),
    create_tno_experiment(), 
    create_pno_experiment()
]

# Run comparative study
results = experiment_manager.run_research_campaign(experiments)
```

### Industrial Application Deployment
```python
# production_inference.py
from neural_operator_lab.optimization import ConcurrentInference

# Set up concurrent inference
with ConcurrentInference(model, num_workers=8) as inference:
    # Process multiple batches concurrently
    results = inference.predict_concurrent(input_batches)
```

### High-Performance Computing Deployment
```bash
# Submit SLURM job
sbatch --nodes=4 --gpus=16 \
       --job-name=neural_operator_training \
       scripts/distributed_training.sh
```

## 🔄 CI/CD Pipeline

### GitHub Actions
```yaml
# .github/workflows/deploy.yml
name: Deploy to Production

on:
  push:
    branches: [main]

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Build and test
        run: |
          python3 simple_test.py
          
      - name: Build Docker image
        run: |
          docker build -t neural-operator-lab:${{ github.sha }} .
          
      - name: Deploy to Kubernetes
        run: |
          kubectl set image deployment/neural-operator-deployment \
            app=neural-operator-lab:${{ github.sha }}
```

### Jenkins Pipeline
```groovy
pipeline {
    agent any
    
    stages {
        stage('Test') {
            steps {
                sh 'python3 simple_test.py'
            }
        }
        
        stage('Build') {
            steps {
                sh 'docker build -t neural-operator-lab:${BUILD_NUMBER} .'
            }
        }
        
        stage('Deploy') {
            steps {
                sh 'kubectl apply -f deployment/kubernetes/'
            }
        }
    }
}
```

## 🚨 Troubleshooting

### Common Issues

**Issue: CUDA out of memory**
```python
# Solution: Enable auto-scaling
from neural_operator_lab.scaling import create_training_auto_scaler
scaler = create_training_auto_scaler(trainer)
scaler.start()
```

**Issue: Training instability**
```python
# Solution: Enable robust trainer
from neural_operator_lab.monitoring import RobustTrainer
robust_trainer = RobustTrainer(model, config)
```

**Issue: Slow data loading**
```python
# Solution: Optimize data loader
from neural_operator_lab.optimization import auto_optimize_training
trainer, dataloader = auto_optimize_training(trainer, dataloader)
```

### Performance Tuning
```python
# Get performance report
optimizer = PerformanceOptimizer()
report = optimizer.get_optimization_report()
print(f"Cache hit rate: {report['model_cache_stats']['hit_rate']}")

# Profile training step
profile_results = optimizer.profile_training_step(trainer, dataloader, num_steps=10)
print(f"Throughput: {profile_results['throughput_samples_per_sec']} samples/sec")
```

## 📋 Production Checklist

### Pre-Deployment
- [ ] All tests passing (>85% coverage)
- [ ] Security scan completed  
- [ ] Performance benchmarks verified
- [ ] Configuration validated
- [ ] Dependencies verified
- [ ] Documentation updated

### Deployment
- [ ] Infrastructure provisioned
- [ ] Application deployed
- [ ] Health checks passing
- [ ] Monitoring configured
- [ ] Alerts configured
- [ ] Backup strategy implemented

### Post-Deployment  
- [ ] Smoke tests completed
- [ ] Performance monitoring active
- [ ] Log aggregation working
- [ ] Security monitoring enabled
- [ ] Documentation accessible
- [ ] Support team notified

## 📞 Support

### Community Support
- GitHub Issues: https://github.com/terragon-labs/neural-operator-foundation-lab/issues
- Discussion Forum: https://github.com/terragon-labs/neural-operator-foundation-lab/discussions
- Documentation: https://neural-operator-lab.readthedocs.io

### Enterprise Support
- Email: support@terragon.ai
- Documentation: https://docs.terragon.ai/neural-operator-lab
- Training: https://training.terragon.ai

---

**🚀 Neural Operator Foundation Lab - Production Deployment Guide**  
*From Research to Production in Minutes*

*Terragon Labs - Advancing Neural Operator Technology*