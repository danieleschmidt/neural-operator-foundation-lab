#!/usr/bin/env python3
"""Global-first deployment manager for neural operator lab."""

import sys
import time
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

# Add src to path
sys.path.insert(0, 'src')


@dataclass
class DeploymentRegion:
    """Deployment region configuration."""
    name: str
    code: str  # e.g., 'us-east-1', 'eu-west-1'
    compliance_requirements: List[str]
    data_residency: bool = True
    languages: List[str] = field(default_factory=lambda: ['en'])
    
    def __post_init__(self):
        """Validate region configuration."""
        if not self.code:
            raise ValueError("Region code is required")


@dataclass
class ComplianceConfig:
    """Compliance configuration for regions."""
    gdpr_required: bool = False
    ccpa_required: bool = False
    pdpa_required: bool = False
    hipaa_required: bool = False
    soc2_required: bool = False
    iso27001_required: bool = False
    data_encryption: bool = True
    audit_logging: bool = True
    
    def get_requirements(self) -> List[str]:
        """Get list of compliance requirements."""
        requirements = []
        if self.gdpr_required:
            requirements.append("GDPR")
        if self.ccpa_required:
            requirements.append("CCPA")
        if self.pdpa_required:
            requirements.append("PDPA")
        if self.hipaa_required:
            requirements.append("HIPAA")
        if self.soc2_required:
            requirements.append("SOC2")
        if self.iso27001_required:
            requirements.append("ISO27001")
        return requirements


class GlobalDeploymentManager:
    """Manages global deployment across multiple regions."""
    
    def __init__(self):
        self.regions = self._setup_default_regions()
        self.deployment_configs = {}
        self.compliance_configs = {}
        
    def _setup_default_regions(self) -> Dict[str, DeploymentRegion]:
        """Setup default deployment regions."""
        regions = {
            'us-east-1': DeploymentRegion(
                name="US East (Virginia)",
                code="us-east-1",
                compliance_requirements=["CCPA", "SOC2"],
                languages=['en', 'es']
            ),
            'eu-west-1': DeploymentRegion(
                name="Europe (Ireland)",
                code="eu-west-1", 
                compliance_requirements=["GDPR", "ISO27001"],
                languages=['en', 'de', 'fr', 'es']
            ),
            'ap-southeast-1': DeploymentRegion(
                name="Asia Pacific (Singapore)",
                code="ap-southeast-1",
                compliance_requirements=["PDPA"],
                languages=['en', 'zh', 'ja']
            ),
            'ap-northeast-1': DeploymentRegion(
                name="Asia Pacific (Tokyo)",
                code="ap-northeast-1",
                compliance_requirements=["PDPA"],
                languages=['en', 'ja']
            ),
            'ca-central-1': DeploymentRegion(
                name="Canada (Central)",
                code="ca-central-1",
                compliance_requirements=["GDPR"],  # Similar to GDPR
                languages=['en', 'fr']
            )
        }
        
        return regions
    
    def prepare_global_deployment(self) -> Dict[str, Any]:
        """Prepare comprehensive global deployment."""
        print("🌍 PREPARING GLOBAL-FIRST DEPLOYMENT")
        print("=" * 50)
        
        results = {
            'timestamp': time.time(),
            'regions': {},
            'overall_status': 'success',
            'deployment_artifacts': [],
            'compliance_summary': {},
            'errors': []
        }
        
        try:
            # 1. Generate region-specific configurations
            print("📋 Generating region-specific configurations...")
            self._generate_region_configs(results)
            
            # 2. Prepare compliance configurations
            print("🛡️ Preparing compliance configurations...")
            self._prepare_compliance_configs(results)
            
            # 3. Generate deployment artifacts
            print("📦 Generating deployment artifacts...")
            self._generate_deployment_artifacts(results)
            
            # 4. Create infrastructure configurations
            print("🏗️ Creating infrastructure configurations...")
            self._create_infrastructure_configs(results)
            
            # 5. Setup monitoring and observability
            print("📊 Setting up monitoring...")
            self._setup_monitoring_configs(results)
            
            # 6. Generate security configurations
            print("🔒 Configuring security...")
            self._setup_security_configs(results)
            
            # 7. Create deployment scripts
            print("🚀 Creating deployment scripts...")
            self._create_deployment_scripts(results)
            
            print("\n✅ Global deployment preparation completed!")
            
        except Exception as e:
            results['overall_status'] = 'failed'
            results['errors'].append(str(e))
            print(f"\n❌ Global deployment preparation failed: {e}")
        
        return results
    
    def _generate_region_configs(self, results: Dict[str, Any]):
        """Generate region-specific configurations."""
        for region_code, region in self.regions.items():
            region_config = {
                'region_info': {
                    'name': region.name,
                    'code': region.code,
                    'compliance_requirements': region.compliance_requirements,
                    'data_residency': region.data_residency,
                    'languages': region.languages
                },
                'environment': self._generate_environment_config(region),
                'application': self._generate_app_config(region),
                'database': self._generate_db_config(region),
                'networking': self._generate_network_config(region)
            }
            
            results['regions'][region_code] = region_config
            
            # Save region-specific config file
            config_file = Path(f"deployment/regions/{region_code}_config.json")
            config_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(config_file, 'w') as f:
                json.dump(region_config, f, indent=2)
            
            results['deployment_artifacts'].append(str(config_file))
    
    def _generate_environment_config(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate environment configuration for region."""
        return {
            'REGION': region.code,
            'DATA_RESIDENCY': str(region.data_residency).lower(),
            'COMPLIANCE_REQUIREMENTS': ','.join(region.compliance_requirements),
            'DEFAULT_LANGUAGE': region.languages[0],
            'SUPPORTED_LANGUAGES': ','.join(region.languages),
            'LOG_LEVEL': 'INFO',
            'METRICS_ENABLED': 'true',
            'ENCRYPTION_ENABLED': 'true',
            'AUDIT_LOGGING': 'true'
        }
    
    def _generate_app_config(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate application configuration for region."""
        return {
            'scaling': {
                'min_instances': 2,
                'max_instances': 10,
                'target_cpu_utilization': 70
            },
            'resources': {
                'memory_limit': '2Gi',
                'cpu_limit': '1000m',
                'memory_request': '1Gi',
                'cpu_request': '500m'
            },
            'health_checks': {
                'liveness_probe': '/health/live',
                'readiness_probe': '/health/ready',
                'startup_probe': '/health/startup'
            }
        }
    
    def _generate_db_config(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate database configuration for region."""
        return {
            'type': 'postgresql',
            'version': '13',
            'encryption_at_rest': True,
            'encryption_in_transit': True,
            'backup_retention_days': 30,
            'multi_az': True,
            'performance_insights': True,
            'deletion_protection': True
        }
    
    def _generate_network_config(self, region: DeploymentRegion) -> Dict[str, Any]:
        """Generate networking configuration for region."""
        return {
            'vpc_cidr': '10.0.0.0/16',
            'public_subnets': ['10.0.1.0/24', '10.0.2.0/24'],
            'private_subnets': ['10.0.10.0/24', '10.0.20.0/24'],
            'enable_nat_gateway': True,
            'enable_vpn_gateway': False,
            'dns_hostnames': True,
            'dns_support': True
        }
    
    def _prepare_compliance_configs(self, results: Dict[str, Any]):
        """Prepare compliance configurations for each region."""
        compliance_summary = {}
        
        for region_code, region in self.regions.items():
            compliance_config = ComplianceConfig()
            
            # Set compliance requirements based on region
            if 'GDPR' in region.compliance_requirements:
                compliance_config.gdpr_required = True
            if 'CCPA' in region.compliance_requirements:
                compliance_config.ccpa_required = True
            if 'PDPA' in region.compliance_requirements:
                compliance_config.pdpa_required = True
            if 'SOC2' in region.compliance_requirements:
                compliance_config.soc2_required = True
            if 'ISO27001' in region.compliance_requirements:
                compliance_config.iso27001_required = True
            
            self.compliance_configs[region_code] = compliance_config
            compliance_summary[region_code] = compliance_config.get_requirements()
            
            # Generate compliance configuration file
            compliance_file = Path(f"deployment/compliance/{region_code}_compliance.json")
            compliance_file.parent.mkdir(parents=True, exist_ok=True)
            
            compliance_data = {
                'region': region_code,
                'requirements': compliance_config.get_requirements(),
                'data_encryption': compliance_config.data_encryption,
                'audit_logging': compliance_config.audit_logging,
                'policies': self._generate_compliance_policies(compliance_config)
            }
            
            with open(compliance_file, 'w') as f:
                json.dump(compliance_data, f, indent=2)
            
            results['deployment_artifacts'].append(str(compliance_file))
        
        results['compliance_summary'] = compliance_summary
    
    def _generate_compliance_policies(self, config: ComplianceConfig) -> Dict[str, Any]:
        """Generate compliance policies."""
        policies = {
            'data_retention': {
                'default_retention_days': 365,
                'user_data_retention_days': 1095,  # 3 years
                'log_retention_days': 90
            },
            'encryption': {
                'encryption_at_rest': config.data_encryption,
                'encryption_in_transit': True,
                'key_rotation_days': 90
            },
            'access_control': {
                'mfa_required': True,
                'role_based_access': True,
                'audit_access_logs': config.audit_logging
            }
        }
        
        if config.gdpr_required:
            policies['gdpr'] = {
                'right_to_be_forgotten': True,
                'data_portability': True,
                'consent_management': True,
                'breach_notification_hours': 72
            }
        
        if config.ccpa_required:
            policies['ccpa'] = {
                'opt_out_rights': True,
                'data_sale_disclosure': True,
                'consumer_rights': True
            }
        
        return policies
    
    def _generate_deployment_artifacts(self, results: Dict[str, Any]):
        """Generate deployment artifacts."""
        artifacts = []
        
        # 1. Docker configurations
        self._create_docker_configs(artifacts)
        
        # 2. Kubernetes manifests
        self._create_k8s_manifests(artifacts)
        
        # 3. Helm charts
        self._create_helm_charts(artifacts)
        
        # 4. Terraform configurations
        self._create_terraform_configs(artifacts)
        
        results['deployment_artifacts'].extend(artifacts)
    
    def _create_docker_configs(self, artifacts: List[str]):
        """Create Docker configurations."""
        # Multi-stage Dockerfile for production
        dockerfile_content = '''# Multi-stage Docker build for Neural Operator Lab
FROM python:3.9-slim as builder

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    gcc \\
    g++ \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements*.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.9-slim as production

# Security: Create non-root user
RUN groupadd -r neuralop && useradd -r -g neuralop neuralop

WORKDIR /app

# Copy Python dependencies from builder
COPY --from=builder /root/.local /home/neuralop/.local

# Copy application code
COPY src/ ./src/
COPY *.py ./

# Set ownership
RUN chown -R neuralop:neuralop /app

# Switch to non-root user
USER neuralop

# Set PATH for user-installed packages
ENV PATH=/home/neuralop/.local/bin:$PATH

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Expose port
EXPOSE 8000

# Run application
CMD ["python", "-m", "neural_operator_lab.cli.main", "serve"]
'''
        
        dockerfile_path = Path("deployment/docker/Dockerfile.production")
        dockerfile_path.parent.mkdir(parents=True, exist_ok=True)
        dockerfile_path.write_text(dockerfile_content)
        artifacts.append(str(dockerfile_path))
        
        # Docker Compose for development
        compose_content = '''version: '3.8'

services:
  neural-operator-lab:
    build:
      context: ../..
      dockerfile: deployment/docker/Dockerfile.production
    ports:
      - "8000:8000"
    environment:
      - LOG_LEVEL=INFO
      - METRICS_ENABLED=true
    volumes:
      - neural_op_data:/app/data
    depends_on:
      - postgres
      - redis
    restart: unless-stopped

  postgres:
    image: postgres:13-alpine
    environment:
      POSTGRES_DB: neuralop
      POSTGRES_USER: neuralop
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD:-neuralop123}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    ports:
      - "5432:5432"
    restart: unless-stopped

  redis:
    image: redis:6-alpine
    ports:
      - "6379:6379"
    volumes:
      - redis_data:/data
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
      - prometheus_data:/prometheus
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=${GRAFANA_PASSWORD:-admin123}
    volumes:
      - grafana_data:/var/lib/grafana
    restart: unless-stopped

volumes:
  neural_op_data:
  postgres_data:
  redis_data:
  prometheus_data:
  grafana_data:
'''
        
        compose_path = Path("deployment/docker/docker-compose.production.yml")
        compose_path.write_text(compose_content)
        artifacts.append(str(compose_path))
    
    def _create_k8s_manifests(self, artifacts: List[str]):
        """Create Kubernetes manifests."""
        k8s_dir = Path("deployment/kubernetes")
        k8s_dir.mkdir(parents=True, exist_ok=True)
        
        # Deployment manifest
        deployment_yaml = '''apiVersion: apps/v1
kind: Deployment
metadata:
  name: neural-operator-lab
  labels:
    app: neural-operator-lab
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
      securityContext:
        runAsNonRoot: true
        runAsUser: 1000
      containers:
      - name: neural-operator-lab
        image: neural-operator-lab:latest
        ports:
        - containerPort: 8000
        env:
        - name: LOG_LEVEL
          value: "INFO"
        - name: METRICS_ENABLED
          value: "true"
        resources:
          requests:
            memory: "1Gi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "1000m"
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
          initialDelaySeconds: 10
          periodSeconds: 5
        securityContext:
          allowPrivilegeEscalation: false
          readOnlyRootFilesystem: true
          capabilities:
            drop:
            - ALL
---
apiVersion: v1
kind: Service
metadata:
  name: neural-operator-lab-service
spec:
  selector:
    app: neural-operator-lab
  ports:
  - protocol: TCP
    port: 80
    targetPort: 8000
  type: ClusterIP
'''
        
        deployment_path = k8s_dir / "deployment.yaml"
        deployment_path.write_text(deployment_yaml)
        artifacts.append(str(deployment_path))
        
        # HPA manifest
        hpa_yaml = '''apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: neural-operator-lab-hpa
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: neural-operator-lab
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  - type: Resource
    resource:
      name: memory
      target:
        type: Utilization
        averageUtilization: 80
'''
        
        hpa_path = k8s_dir / "hpa.yaml"
        hpa_path.write_text(hpa_yaml)
        artifacts.append(str(hpa_path))
    
    def _create_helm_charts(self, artifacts: List[str]):
        """Create Helm charts."""
        helm_dir = Path("deployment/helm/neural-operator-lab")
        helm_dir.mkdir(parents=True, exist_ok=True)
        
        # Chart.yaml
        chart_yaml = '''apiVersion: v2
name: neural-operator-lab
description: A Helm chart for Neural Operator Foundation Lab
type: application
version: 0.1.0
appVersion: "0.1.0"
keywords:
  - neural-operators
  - machine-learning
  - pde
  - physics
home: https://github.com/terragon-labs/neural-operator-foundation-lab
sources:
  - https://github.com/terragon-labs/neural-operator-foundation-lab
maintainers:
  - name: Daniel Schmidt
    email: daniel@terragon.ai
'''
        
        chart_path = helm_dir / "Chart.yaml"
        chart_path.write_text(chart_yaml)
        artifacts.append(str(chart_path))
        
        # Values.yaml
        values_yaml = '''# Default values for neural-operator-lab
replicaCount: 3

image:
  repository: neural-operator-lab
  pullPolicy: IfNotPresent
  tag: "latest"

nameOverride: ""
fullnameOverride: ""

service:
  type: ClusterIP
  port: 80
  targetPort: 8000

ingress:
  enabled: true
  className: "nginx"
  annotations:
    cert-manager.io/cluster-issuer: "letsencrypt-prod"
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
  hosts:
    - host: neural-operator-lab.example.com
      paths:
        - path: /
          pathType: Prefix
  tls:
    - secretName: neural-operator-lab-tls
      hosts:
        - neural-operator-lab.example.com

resources:
  limits:
    cpu: 1000m
    memory: 2Gi
  requests:
    cpu: 500m
    memory: 1Gi

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilizationPercentage: 70
  targetMemoryUtilizationPercentage: 80

nodeSelector: {}
tolerations: []
affinity: {}

securityContext:
  runAsNonRoot: true
  runAsUser: 1000

podSecurityContext:
  fsGroup: 1000

env:
  LOG_LEVEL: INFO
  METRICS_ENABLED: "true"
  ENCRYPTION_ENABLED: "true"

monitoring:
  enabled: true
  serviceMonitor:
    enabled: true
    interval: 30s

persistence:
  enabled: true
  storageClass: "fast-ssd"
  size: 10Gi
'''
        
        values_path = helm_dir / "values.yaml"
        values_path.write_text(values_yaml)
        artifacts.append(str(values_path))
    
    def _create_terraform_configs(self, artifacts: List[str]):
        """Create Terraform configurations."""
        tf_dir = Path("deployment/terraform")
        tf_dir.mkdir(parents=True, exist_ok=True)
        
        # Main Terraform configuration
        main_tf = '''terraform {
  required_version = ">= 1.0"
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "aws" {
  region = var.aws_region
}

# VPC Configuration
module "vpc" {
  source = "terraform-aws-modules/vpc/aws"
  
  name = "${var.project_name}-vpc"
  cidr = var.vpc_cidr
  
  azs             = data.aws_availability_zones.available.names
  private_subnets = var.private_subnets
  public_subnets  = var.public_subnets
  
  enable_nat_gateway = true
  enable_vpn_gateway = false
  enable_dns_hostnames = true
  enable_dns_support = true
  
  tags = var.common_tags
}

# EKS Cluster
module "eks" {
  source = "terraform-aws-modules/eks/aws"
  
  cluster_name    = "${var.project_name}-cluster"
  cluster_version = "1.27"
  
  vpc_id     = module.vpc.vpc_id
  subnet_ids = module.vpc.private_subnets
  
  cluster_endpoint_private_access = true
  cluster_endpoint_public_access  = true
  
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
  }
  
  eks_managed_node_groups = {
    main = {
      min_size     = 2
      max_size     = 10
      desired_size = 3
      
      instance_types = ["t3.medium"]
      capacity_type  = "ON_DEMAND"
      
      k8s_labels = {
        Environment = var.environment
        Project     = var.project_name
      }
      
      tags = var.common_tags
    }
  }
  
  tags = var.common_tags
}

# RDS Database
resource "aws_db_instance" "main" {
  identifier = "${var.project_name}-db"
  
  engine         = "postgres"
  engine_version = "13.13"
  instance_class = "db.t3.micro"
  
  allocated_storage     = 20
  max_allocated_storage = 100
  
  db_name  = var.db_name
  username = var.db_username
  password = var.db_password
  
  vpc_security_group_ids = [aws_security_group.rds.id]
  db_subnet_group_name   = aws_db_subnet_group.main.name
  
  backup_retention_period = 30
  backup_window          = "03:00-04:00"
  maintenance_window     = "sun:04:00-sun:05:00"
  
  storage_encrypted = true
  deletion_protection = true
  
  skip_final_snapshot = false
  final_snapshot_identifier = "${var.project_name}-final-snapshot"
  
  tags = var.common_tags
}

data "aws_availability_zones" "available" {
  state = "available"
}
'''
        
        main_tf_path = tf_dir / "main.tf"
        main_tf_path.write_text(main_tf)
        artifacts.append(str(main_tf_path))
        
        # Variables
        variables_tf = '''variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "project_name" {
  description = "Project name"
  type        = string
  default     = "neural-operator-lab"
}

variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "vpc_cidr" {
  description = "VPC CIDR block"
  type        = string
  default     = "10.0.0.0/16"
}

variable "private_subnets" {
  description = "Private subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.1.0/24", "10.0.2.0/24", "10.0.3.0/24"]
}

variable "public_subnets" {
  description = "Public subnet CIDR blocks"
  type        = list(string)
  default     = ["10.0.101.0/24", "10.0.102.0/24", "10.0.103.0/24"]
}

variable "db_name" {
  description = "Database name"
  type        = string
  default     = "neuralop"
}

variable "db_username" {
  description = "Database username"
  type        = string
  default     = "neuralop"
}

variable "db_password" {
  description = "Database password"
  type        = string
  sensitive   = true
}

variable "common_tags" {
  description = "Common tags for all resources"
  type        = map(string)
  default = {
    Project     = "neural-operator-lab"
    Environment = "production"
    ManagedBy   = "terraform"
  }
}
'''
        
        variables_tf_path = tf_dir / "variables.tf"
        variables_tf_path.write_text(variables_tf)
        artifacts.append(str(variables_tf_path))
    
    def _create_infrastructure_configs(self, results: Dict[str, Any]):
        """Create infrastructure configurations."""
        # CloudFormation template is already in deployment/aws/
        # Add additional infrastructure configs here
        pass
    
    def _setup_monitoring_configs(self, results: Dict[str, Any]):
        """Setup monitoring configurations."""
        monitoring_dir = Path("deployment/monitoring")
        monitoring_dir.mkdir(parents=True, exist_ok=True)
        
        # Prometheus configuration
        prometheus_config = '''global:
  scrape_interval: 15s
  evaluation_interval: 15s

rule_files:
  - "alert_rules.yml"

alerting:
  alertmanagers:
    - static_configs:
        - targets:
          - alertmanager:9093

scrape_configs:
  - job_name: 'neural-operator-lab'
    static_configs:
      - targets: ['neural-operator-lab:8000']
    metrics_path: '/metrics'
    scrape_interval: 30s

  - job_name: 'kubernetes-pods'
    kubernetes_sd_configs:
      - role: pod
    relabel_configs:
      - source_labels: [__meta_kubernetes_pod_annotation_prometheus_io_scrape]
        action: keep
        regex: true
'''
        
        prometheus_path = monitoring_dir / "prometheus.yml"
        prometheus_path.write_text(prometheus_config)
        results['deployment_artifacts'].append(str(prometheus_path))
        
        # Grafana dashboard
        grafana_dashboard = '''{
  "dashboard": {
    "id": null,
    "title": "Neural Operator Lab Dashboard",
    "tags": ["neural-operators", "ml"],
    "timezone": "browser",
    "panels": [
      {
        "id": 1,
        "title": "Request Rate",
        "type": "graph",
        "targets": [
          {
            "expr": "rate(http_requests_total[5m])",
            "legendFormat": "{{method}} {{status}}"
          }
        ]
      },
      {
        "id": 2,
        "title": "Response Time",
        "type": "graph",
        "targets": [
          {
            "expr": "histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m]))",
            "legendFormat": "95th percentile"
          }
        ]
      },
      {
        "id": 3,
        "title": "Memory Usage",
        "type": "graph",
        "targets": [
          {
            "expr": "process_resident_memory_bytes / 1024 / 1024",
            "legendFormat": "Memory (MB)"
          }
        ]
      }
    ],
    "time": {
      "from": "now-1h",
      "to": "now"
    },
    "refresh": "30s"
  }
}'''
        
        dashboard_path = monitoring_dir / "grafana_dashboard.json"
        dashboard_path.write_text(grafana_dashboard)
        results['deployment_artifacts'].append(str(dashboard_path))
    
    def _setup_security_configs(self, results: Dict[str, Any]):
        """Setup security configurations."""
        security_dir = Path("deployment/security")
        security_dir.mkdir(parents=True, exist_ok=True)
        
        # Network policies
        network_policy = '''apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: neural-operator-lab-netpol
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
      port: 8000
  egress:
  - to:
    - namespaceSelector:
        matchLabels:
          name: kube-system
  - to: []
    ports:
    - protocol: TCP
      port: 5432  # PostgreSQL
    - protocol: TCP
      port: 6379  # Redis
'''
        
        netpol_path = security_dir / "network_policy.yaml"
        netpol_path.write_text(network_policy)
        results['deployment_artifacts'].append(str(netpol_path))
        
        # Pod Security Policy
        psp = '''apiVersion: policy/v1beta1
kind: PodSecurityPolicy
metadata:
  name: neural-operator-lab-psp
spec:
  privileged: false
  allowPrivilegeEscalation: false
  requiredDropCapabilities:
    - ALL
  volumes:
    - 'configMap'
    - 'emptyDir'
    - 'projected'
    - 'secret'
    - 'downwardAPI'
    - 'persistentVolumeClaim'
  hostNetwork: false
  hostIPC: false
  hostPID: false
  runAsUser:
    rule: 'MustRunAsNonRoot'
  seLinux:
    rule: 'RunAsAny'
  fsGroup:
    rule: 'RunAsAny'
'''
        
        psp_path = security_dir / "pod_security_policy.yaml"
        psp_path.write_text(psp)
        results['deployment_artifacts'].append(str(psp_path))
    
    def _create_deployment_scripts(self, results: Dict[str, Any]):
        """Create deployment scripts."""
        scripts_dir = Path("deployment/scripts")
        scripts_dir.mkdir(parents=True, exist_ok=True)
        
        # Main deployment script
        deploy_script = '''#!/bin/bash
set -euo pipefail

# Neural Operator Lab Global Deployment Script
echo "🌍 Starting Neural Operator Lab Global Deployment"

# Configuration
REGIONS=("us-east-1" "eu-west-1" "ap-southeast-1")
PROJECT_NAME="neural-operator-lab"
ENVIRONMENT="${ENVIRONMENT:-production}"

# Colors for output
RED='\\033[0;31m'
GREEN='\\033[0;32m'
YELLOW='\\033[1;33m'
NC='\\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking deployment dependencies..."
    
    if ! command -v docker &> /dev/null; then
        log_error "Docker is required but not installed"
        exit 1
    fi
    
    if ! command -v kubectl &> /dev/null; then
        log_error "kubectl is required but not installed"
        exit 1
    fi
    
    if ! command -v terraform &> /dev/null; then
        log_error "terraform is required but not installed"
        exit 1
    fi
    
    log_info "All dependencies are available"
}

# Build application
build_application() {
    log_info "Building application..."
    docker build -t $PROJECT_NAME:latest -f deployment/docker/Dockerfile.production .
    log_info "Application build completed"
}

# Deploy to region
deploy_to_region() {
    local region=$1
    log_info "Deploying to region: $region"
    
    # Set AWS region
    export AWS_DEFAULT_REGION=$region
    
    # Initialize Terraform
    cd deployment/terraform
    terraform init -backend-config="region=$region"
    
    # Plan and apply Terraform
    terraform plan -var="aws_region=$region" -var="environment=$ENVIRONMENT"
    terraform apply -var="aws_region=$region" -var="environment=$ENVIRONMENT" -auto-approve
    
    # Deploy to Kubernetes
    kubectl apply -f ../kubernetes/
    
    cd ../..
    log_info "Deployment to $region completed"
}

# Main deployment function
main() {
    log_info "Starting global deployment process"
    
    check_dependencies
    build_application
    
    for region in "${REGIONS[@]}"; do
        deploy_to_region $region
    done
    
    log_info "🎉 Global deployment completed successfully!"
}

# Run main function
main "$@"
'''
        
        deploy_script_path = scripts_dir / "deploy.sh"
        deploy_script_path.write_text(deploy_script)
        deploy_script_path.chmod(0o755)  # Make executable
        results['deployment_artifacts'].append(str(deploy_script_path))
        
        # Health check script
        health_check_script = '''#!/bin/bash
set -euo pipefail

# Health check script for Neural Operator Lab
ENDPOINTS=(
    "https://us-east-1.neural-operator-lab.com/health"
    "https://eu-west-1.neural-operator-lab.com/health"
    "https://ap-southeast-1.neural-operator-lab.com/health"
)

echo "🔍 Checking health of all regions..."

for endpoint in "${ENDPOINTS[@]}"; do
    echo "Checking $endpoint..."
    
    if curl -f -s "$endpoint" > /dev/null; then
        echo "✅ $endpoint is healthy"
    else
        echo "❌ $endpoint is unhealthy"
        exit 1
    fi
done

echo "🎉 All regions are healthy!"
'''
        
        health_script_path = scripts_dir / "health_check.sh"
        health_script_path.write_text(health_check_script)
        health_script_path.chmod(0o755)
        results['deployment_artifacts'].append(str(health_script_path))


def main():
    """Run global deployment preparation."""
    manager = GlobalDeploymentManager()
    results = manager.prepare_global_deployment()
    
    print(f"\n📊 GLOBAL DEPLOYMENT SUMMARY")
    print("=" * 50)
    print(f"Status: {results['overall_status'].upper()}")
    print(f"Regions configured: {len(results['regions'])}")
    print(f"Artifacts generated: {len(results['deployment_artifacts'])}")
    
    if results['errors']:
        print(f"Errors: {len(results['errors'])}")
        for error in results['errors']:
            print(f"  ❌ {error}")
    
    print(f"\n🌍 Regional Configuration:")
    for region_code, region_config in results['regions'].items():
        region_info = region_config['region_info']
        print(f"  📍 {region_info['name']} ({region_code})")
        print(f"     Languages: {', '.join(region_info['languages'])}")
        print(f"     Compliance: {', '.join(region_info['compliance_requirements'])}")
    
    print(f"\n🛡️ Compliance Summary:")
    for region, requirements in results['compliance_summary'].items():
        print(f"  {region}: {', '.join(requirements) if requirements else 'No specific requirements'}")
    
    print(f"\n📦 Key Artifacts Generated:")
    key_artifacts = [
        "deployment/docker/Dockerfile.production",
        "deployment/kubernetes/deployment.yaml", 
        "deployment/terraform/main.tf",
        "deployment/scripts/deploy.sh"
    ]
    
    for artifact in key_artifacts:
        if any(artifact in path for path in results['deployment_artifacts']):
            print(f"  ✅ {artifact}")
        else:
            print(f"  ⏭️ {artifact}")
    
    # Save deployment summary
    summary_file = Path("global_deployment_summary.json")
    with open(summary_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Deployment summary saved to: {summary_file}")
    
    if results['overall_status'] == 'success':
        print("\n✅ Global deployment preparation completed successfully!")
        print("🚀 Ready for multi-region deployment!")
        return 0
    else:
        print("\n❌ Global deployment preparation failed!")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)