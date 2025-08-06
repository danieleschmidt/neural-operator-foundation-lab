#!/bin/bash

# Neural Operator Foundation Lab - Production Deployment Script
# Terragon Labs - Autonomous SDLC Execution

set -euo pipefail

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"
DEPLOYMENT_MODE="${1:-docker}"
ENVIRONMENT="${2:-production}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check dependencies
check_dependencies() {
    log_info "Checking dependencies..."
    
    local deps=()
    case "$DEPLOYMENT_MODE" in
        docker)
            deps=("docker" "docker-compose")
            ;;
        kubernetes)
            deps=("kubectl" "helm")
            ;;
        aws)
            deps=("aws" "docker")
            ;;
    esac
    
    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            log_error "$dep is required but not installed."
            exit 1
        fi
    done
    
    log_success "All dependencies are available."
}

# Validate configuration
validate_config() {
    log_info "Validating configuration..."
    
    local config_file="$PROJECT_ROOT/deployment/configs/$ENVIRONMENT.yaml"
    if [[ ! -f "$config_file" ]]; then
        log_error "Configuration file not found: $config_file"
        exit 1
    fi
    
    # Python validation
    python3 -c "
import yaml
import sys

try:
    with open('$config_file', 'r') as f:
        config = yaml.safe_load(f)
    
    # Basic validation
    required_sections = ['experiment', 'training', 'model', 'data', 'distributed']
    for section in required_sections:
        if section not in config:
            print(f'Missing required section: {section}')
            sys.exit(1)
    
    print('Configuration validation passed')
except Exception as e:
    print(f'Configuration validation failed: {e}')
    sys.exit(1)
    " || exit 1
    
    log_success "Configuration validated successfully."
}

# Build Docker images
build_images() {
    log_info "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    
    # Build GPU image
    docker build -f deployment/docker/Dockerfile -t neural-operator-lab:latest -t neural-operator-lab:gpu .
    
    # Build CPU image
    docker build -f deployment/docker/Dockerfile.cpu -t neural-operator-lab:cpu .
    
    log_success "Docker images built successfully."
}

# Deploy with Docker Compose
deploy_docker() {
    log_info "Deploying with Docker Compose..."
    
    cd "$PROJECT_ROOT/deployment/docker"
    
    # Set environment variables
    export COMPOSE_PROJECT_NAME="neural-operator-${ENVIRONMENT}"
    export POSTGRES_PASSWORD="${POSTGRES_PASSWORD:-$(openssl rand -base64 32)}"
    export GRAFANA_PASSWORD="${GRAFANA_PASSWORD:-$(openssl rand -base64 16)}"
    
    # Create required directories
    mkdir -p ../../data ../../configs
    
    # Start services
    docker-compose -f docker-compose.yml up -d
    
    # Wait for services to be ready
    log_info "Waiting for services to be ready..."
    sleep 30
    
    # Check service health
    if docker-compose ps | grep -q "healthy\|Up"; then
        log_success "Docker deployment completed successfully."
        docker-compose ps
    else
        log_error "Some services failed to start properly."
        docker-compose logs
        exit 1
    fi
}

# Deploy to Kubernetes
deploy_kubernetes() {
    log_info "Deploying to Kubernetes..."
    
    local k8s_dir="$PROJECT_ROOT/deployment/kubernetes"
    
    # Apply namespace first
    kubectl apply -f "$k8s_dir/namespace.yaml"
    
    # Apply RBAC
    kubectl apply -f "$k8s_dir/rbac.yaml"
    
    # Apply storage
    kubectl apply -f "$k8s_dir/pvc.yaml"
    
    # Apply configuration
    kubectl apply -f "$k8s_dir/configmap.yaml"
    
    # Apply deployment
    kubectl apply -f "$k8s_dir/deployment.yaml"
    
    # Apply services
    kubectl apply -f "$k8s_dir/service.yaml"
    
    # Apply HPA (optional)
    kubectl apply -f "$k8s_dir/hpa.yaml"
    
    # Wait for deployment
    log_info "Waiting for deployment to be ready..."
    kubectl rollout status deployment/neural-operator-training -n neural-operator --timeout=300s
    
    log_success "Kubernetes deployment completed successfully."
    kubectl get pods -n neural-operator
}

# Deploy to AWS
deploy_aws() {
    log_info "Deploying to AWS..."
    
    local cf_template="$PROJECT_ROOT/deployment/aws/cloudformation/neural-operator-infrastructure.yaml"
    local stack_name="neural-operator-${ENVIRONMENT}"
    
    # Build and push images to ECR
    log_info "Setting up ECR repository..."
    
    local account_id=$(aws sts get-caller-identity --query Account --output text)
    local region=$(aws configure get region)
    local repository_uri="${account_id}.dkr.ecr.${region}.amazonaws.com/neural-operator-lab"
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names neural-operator-lab 2>/dev/null || \
        aws ecr create-repository --repository-name neural-operator-lab
    
    # Login to ECR
    aws ecr get-login-password --region "$region" | docker login --username AWS --password-stdin "$repository_uri"
    
    # Tag and push images
    docker tag neural-operator-lab:latest "$repository_uri:latest"
    docker tag neural-operator-lab:latest "$repository_uri:$ENVIRONMENT"
    docker push "$repository_uri:latest"
    docker push "$repository_uri:$ENVIRONMENT"
    
    # Deploy CloudFormation stack
    log_info "Deploying CloudFormation stack..."
    
    aws cloudformation deploy \
        --template-file "$cf_template" \
        --stack-name "$stack_name" \
        --parameter-overrides \
            Environment="$ENVIRONMENT" \
            KeyPairName="${AWS_KEY_PAIR_NAME:-default}" \
            DataBucketName="neural-operator-data-${ENVIRONMENT}-$(date +%s)" \
            ModelBucketName="neural-operator-models-${ENVIRONMENT}-$(date +%s)" \
        --capabilities CAPABILITY_NAMED_IAM
    
    # Get stack outputs
    local alb_dns=$(aws cloudformation describe-stacks \
        --stack-name "$stack_name" \
        --query 'Stacks[0].Outputs[?OutputKey==`LoadBalancerDNS`].OutputValue' \
        --output text)
    
    log_success "AWS deployment completed successfully."
    log_info "Application URL: http://$alb_dns"
}

# Run tests
run_tests() {
    log_info "Running deployment tests..."
    
    case "$DEPLOYMENT_MODE" in
        docker)
            # Test Docker services
            if curl -f http://localhost:8000/health >/dev/null 2>&1; then
                log_success "Health check passed"
            else
                log_warning "Health check failed - service may still be starting"
            fi
            ;;
        kubernetes)
            # Test Kubernetes deployment
            kubectl exec -n neural-operator \
                deployment/neural-operator-training \
                -- python3 -c "import neural_operator_lab; print('Import test passed')"
            ;;
        aws)
            # Test AWS deployment
            log_info "AWS deployment tests require manual verification via the ALB DNS"
            ;;
    esac
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    
    case "$DEPLOYMENT_MODE" in
        docker)
            cd "$PROJECT_ROOT/deployment/docker"
            docker-compose down
            ;;
        kubernetes)
            kubectl delete namespace neural-operator --ignore-not-found=true
            ;;
        aws)
            aws cloudformation delete-stack --stack-name "neural-operator-${ENVIRONMENT}"
            ;;
    esac
    
    log_success "Cleanup completed."
}

# Main deployment function
main() {
    log_info "Starting Neural Operator Foundation Lab deployment..."
    log_info "Deployment mode: $DEPLOYMENT_MODE"
    log_info "Environment: $ENVIRONMENT"
    
    # Trap to ensure cleanup on exit
    trap cleanup EXIT
    
    check_dependencies
    validate_config
    build_images
    
    case "$DEPLOYMENT_MODE" in
        docker)
            deploy_docker
            ;;
        kubernetes|k8s)
            deploy_kubernetes
            ;;
        aws)
            deploy_aws
            ;;
        *)
            log_error "Unknown deployment mode: $DEPLOYMENT_MODE"
            log_info "Supported modes: docker, kubernetes, aws"
            exit 1
            ;;
    esac
    
    run_tests
    
    log_success "Deployment completed successfully!"
    
    # Remove trap so cleanup doesn't run on successful exit
    trap - EXIT
}

# Help function
show_help() {
    cat << EOF
Neural Operator Foundation Lab - Deployment Script

Usage: $0 [DEPLOYMENT_MODE] [ENVIRONMENT]

DEPLOYMENT_MODE:
    docker      Deploy using Docker Compose (default)
    kubernetes  Deploy to Kubernetes cluster
    aws         Deploy to AWS using CloudFormation

ENVIRONMENT:
    production  Production environment (default)
    staging     Staging environment
    development Development environment

Examples:
    $0                          # Deploy to Docker with production config
    $0 kubernetes staging       # Deploy to Kubernetes with staging config
    $0 aws production           # Deploy to AWS with production config

Environment Variables:
    AWS_KEY_PAIR_NAME          AWS EC2 Key Pair name (for AWS deployment)
    POSTGRES_PASSWORD          PostgreSQL password (for Docker deployment)
    GRAFANA_PASSWORD           Grafana admin password (for Docker deployment)

EOF
}

# Parse command line arguments
if [[ $# -gt 0 && ("$1" == "-h" || "$1" == "--help") ]]; then
    show_help
    exit 0
fi

# Run main function
main "$@"