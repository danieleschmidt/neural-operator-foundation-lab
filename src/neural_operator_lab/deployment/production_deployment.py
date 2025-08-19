"""Production deployment utilities for neural operators.

This module provides comprehensive production deployment capabilities including
model serving, health monitoring, A/B testing, and scalable inference.
"""

import torch
import torch.nn as nn
import asyncio
import uvicorn
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional, Union
import numpy as np
import time
import logging
import json
import os
from pathlib import Path
from datetime import datetime, timedelta
import hashlib
import pickle
from contextlib import asynccontextmanager

# Production monitoring
import psutil
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client import CONTENT_TYPE_LATEST

# Model versioning and A/B testing
from enum import Enum
import uuid


class ModelStatus(str, Enum):
    """Model deployment status."""
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    DEPRECATED = "deprecated"


class InferenceRequest(BaseModel):
    """Request model for neural operator inference."""
    model_id: str = Field(..., description="Model identifier")
    input_data: List[List[float]] = Field(..., description="Input tensor data")
    input_shape: List[int] = Field(..., description="Input tensor shape")
    grid_data: Optional[List[List[float]]] = Field(None, description="Optional grid coordinates")
    grid_shape: Optional[List[int]] = Field(None, description="Grid tensor shape")
    inference_config: Optional[Dict[str, Any]] = Field(default_factory=dict)


class InferenceResponse(BaseModel):
    """Response model for neural operator inference."""
    model_id: str
    prediction: List[List[float]]
    prediction_shape: List[int]
    inference_time_ms: float
    model_version: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ModelMetrics:
    """Metrics tracking for deployed models."""
    
    def __init__(self):
        # Prometheus metrics
        self.request_count = Counter(
            'neural_operator_requests_total',
            'Total number of inference requests',
            ['model_id', 'status']
        )
        
        self.inference_duration = Histogram(
            'neural_operator_inference_seconds',
            'Time spent on inference',
            ['model_id']
        )
        
        self.model_memory_usage = Gauge(
            'neural_operator_model_memory_bytes',
            'Memory usage per model',
            ['model_id']
        )
        
        self.active_models = Gauge(
            'neural_operator_active_models',
            'Number of active models'
        )
        
        self.gpu_utilization = Gauge(
            'neural_operator_gpu_utilization',
            'GPU utilization percentage'
        )
    
    def record_inference(self, model_id: str, duration: float, status: str = "success"):
        """Record inference metrics."""
        self.request_count.labels(model_id=model_id, status=status).inc()
        self.inference_duration.labels(model_id=model_id).observe(duration)
    
    def update_system_metrics(self):
        """Update system-level metrics."""
        # GPU metrics
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.utilization()
            self.gpu_utilization.set(gpu_usage)
        
        # Memory metrics
        memory_info = psutil.virtual_memory()
        for model_id in ModelRegistry().list_models():
            # Estimate model memory usage
            model_memory = self._estimate_model_memory(model_id)
            self.model_memory_usage.labels(model_id=model_id).set(model_memory)
    
    def _estimate_model_memory(self, model_id: str) -> float:
        """Estimate memory usage for a model."""
        try:
            model = ModelRegistry().get_model(model_id)
            if model:
                return sum(p.numel() * p.element_size() for p in model.parameters())
        except:
            pass
        return 0.0


class ModelRegistry:
    """Registry for managing deployed models."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.models: Dict[str, Dict[str, Any]] = {}
        self.model_configs: Dict[str, Dict[str, Any]] = {}
        self.ab_tests: Dict[str, Dict[str, Any]] = {}
        self.health_checks: Dict[str, datetime] = {}
        
        self._initialized = True
    
    def register_model(self, 
                      model_id: str,
                      model: nn.Module,
                      version: str,
                      config: Dict[str, Any] = None):
        """Register a model for deployment."""
        self.models[model_id] = {
            'model': model,
            'version': version,
            'status': ModelStatus.READY,
            'loaded_at': datetime.now(),
            'request_count': 0,
            'error_count': 0,
            'last_used': datetime.now()
        }
        
        self.model_configs[model_id] = config or {}
        self.health_checks[model_id] = datetime.now()
        
        logging.info(f"Registered model {model_id} version {version}")
    
    def get_model(self, model_id: str) -> Optional[nn.Module]:
        """Get model by ID."""
        if model_id in self.models:
            model_info = self.models[model_id]
            if model_info['status'] == ModelStatus.READY:
                model_info['last_used'] = datetime.now()
                return model_info['model']
        return None
    
    def list_models(self) -> List[str]:
        """List all registered model IDs."""
        return list(self.models.keys())
    
    def get_model_info(self, model_id: str) -> Optional[Dict[str, Any]]:
        """Get model information."""
        if model_id in self.models:
            info = self.models[model_id].copy()
            info.pop('model', None)  # Don't include model object
            return info
        return None
    
    def update_model_status(self, model_id: str, status: ModelStatus):
        """Update model status."""
        if model_id in self.models:
            self.models[model_id]['status'] = status
            logging.info(f"Updated model {model_id} status to {status}")
    
    def setup_ab_test(self, 
                     test_name: str,
                     model_a: str,
                     model_b: str,
                     traffic_split: float = 0.5):
        """Setup A/B test between two models."""
        self.ab_tests[test_name] = {
            'model_a': model_a,
            'model_b': model_b,
            'traffic_split': traffic_split,
            'created_at': datetime.now(),
            'requests_a': 0,
            'requests_b': 0
        }
        
        logging.info(f"Setup A/B test {test_name}: {model_a} vs {model_b}")
    
    def get_ab_test_model(self, test_name: str, request_id: str) -> Optional[str]:
        """Get model for A/B test based on request ID."""
        if test_name not in self.ab_tests:
            return None
        
        test = self.ab_tests[test_name]
        
        # Use request ID hash for consistent routing
        hash_value = int(hashlib.md5(request_id.encode()).hexdigest(), 16)
        traffic_ratio = (hash_value % 1000) / 1000.0
        
        if traffic_ratio < test['traffic_split']:
            test['requests_a'] += 1
            return test['model_a']
        else:
            test['requests_b'] += 1
            return test['model_b']


class HealthChecker:
    """Health checking for deployed models."""
    
    def __init__(self, registry: ModelRegistry, check_interval: float = 300.0):
        self.registry = registry
        self.check_interval = check_interval
        self._running = False
    
    async def start_health_checks(self):
        """Start periodic health checks."""
        self._running = True
        while self._running:
            await self._perform_health_checks()
            await asyncio.sleep(self.check_interval)
    
    def stop_health_checks(self):
        """Stop health checks."""
        self._running = False
    
    async def _perform_health_checks(self):
        """Perform health checks on all models."""
        for model_id in self.registry.list_models():
            try:
                await self._check_model_health(model_id)
            except Exception as e:
                logging.error(f"Health check failed for model {model_id}: {e}")
                self.registry.update_model_status(model_id, ModelStatus.ERROR)
    
    async def _check_model_health(self, model_id: str):
        """Check health of a specific model."""
        model = self.registry.get_model(model_id)
        if model is None:
            return
        
        # Simple health check: forward pass with dummy data
        try:
            model.eval()
            with torch.no_grad():
                # Create dummy input based on model's expected input
                dummy_input = torch.randn(1, 64, 4)  # Batch, points, features
                output = model(dummy_input)
                
                # Check output validity
                if torch.isnan(output).any() or torch.isinf(output).any():
                    raise ValueError("Model output contains NaN or Inf values")
            
            self.registry.health_checks[model_id] = datetime.now()
            logging.debug(f"Health check passed for model {model_id}")
            
        except Exception as e:
            logging.error(f"Health check failed for model {model_id}: {e}")
            self.registry.update_model_status(model_id, ModelStatus.ERROR)
            raise


class InferenceEngine:
    """High-performance inference engine for neural operators."""
    
    def __init__(self, 
                 registry: ModelRegistry,
                 metrics: ModelMetrics,
                 max_batch_size: int = 32,
                 batch_timeout_ms: float = 100.0):
        self.registry = registry
        self.metrics = metrics
        self.max_batch_size = max_batch_size
        self.batch_timeout_ms = batch_timeout_ms
        
        # Batching queues
        self.batch_queues: Dict[str, List] = {}
        self.batch_futures: Dict[str, List] = {}
        
        # Model warmup cache
        self.warmup_cache: Dict[str, bool] = {}
    
    async def predict(self, request: InferenceRequest) -> InferenceResponse:
        """Perform inference with a neural operator."""
        start_time = time.time()
        
        try:
            # Get model
            model = self.registry.get_model(request.model_id)
            if model is None:
                raise HTTPException(status_code=404, detail=f"Model {request.model_id} not found")
            
            # Warmup model if needed
            await self._warmup_model(request.model_id, model)
            
            # Prepare input tensors
            input_tensor = torch.tensor(request.input_data, dtype=torch.float32)
            input_tensor = input_tensor.view(request.input_shape)
            
            grid_tensor = None
            if request.grid_data and request.grid_shape:
                grid_tensor = torch.tensor(request.grid_data, dtype=torch.float32)
                grid_tensor = grid_tensor.view(request.grid_shape)
            
            # Perform inference
            model.eval()
            with torch.no_grad():
                if grid_tensor is not None:
                    output = model(input_tensor, grid_tensor)
                else:
                    output = model(input_tensor)
            
            # Prepare response
            prediction = output.cpu().numpy().tolist()
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Record metrics
            self.metrics.record_inference(request.model_id, inference_time_ms / 1000)
            
            # Update model stats
            model_info = self.registry.models[request.model_id]
            model_info['request_count'] += 1
            
            return InferenceResponse(
                model_id=request.model_id,
                prediction=prediction,
                prediction_shape=list(output.shape),
                inference_time_ms=inference_time_ms,
                model_version=model_info['version'],
                metadata={
                    'total_requests': model_info['request_count'],
                    'model_loaded_at': model_info['loaded_at'].isoformat()
                }
            )
            
        except Exception as e:
            # Record error metrics
            self.metrics.record_inference(request.model_id, 0, status="error")
            
            # Update error count
            if request.model_id in self.registry.models:
                self.registry.models[request.model_id]['error_count'] += 1
            
            logging.error(f"Inference error for model {request.model_id}: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    async def _warmup_model(self, model_id: str, model: nn.Module):
        """Warmup model with dummy inference."""
        if model_id in self.warmup_cache:
            return
        
        try:
            # Perform warmup inference
            dummy_input = torch.randn(1, 64, 4)
            with torch.no_grad():
                _ = model(dummy_input)
            
            self.warmup_cache[model_id] = True
            logging.info(f"Warmed up model {model_id}")
            
        except Exception as e:
            logging.warning(f"Model warmup failed for {model_id}: {e}")
    
    async def batch_predict(self, requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Batch inference for multiple requests."""
        if not requests:
            return []
        
        # Group by model ID
        model_groups = {}
        for req in requests:
            if req.model_id not in model_groups:
                model_groups[req.model_id] = []
            model_groups[req.model_id].append(req)
        
        # Process each model group
        all_responses = []
        for model_id, model_requests in model_groups.items():
            model_responses = await self._batch_predict_single_model(model_id, model_requests)
            all_responses.extend(model_responses)
        
        return all_responses
    
    async def _batch_predict_single_model(self, 
                                        model_id: str, 
                                        requests: List[InferenceRequest]) -> List[InferenceResponse]:
        """Batch inference for a single model."""
        model = self.registry.get_model(model_id)
        if model is None:
            # Return errors for all requests
            return [
                InferenceResponse(
                    model_id=model_id,
                    prediction=[],
                    prediction_shape=[],
                    inference_time_ms=0.0,
                    model_version="unknown",
                    metadata={"error": f"Model {model_id} not found"}
                )
                for _ in requests
            ]
        
        try:
            start_time = time.time()
            
            # Prepare batch input
            batch_inputs = []
            batch_grids = []
            
            for req in requests:
                input_tensor = torch.tensor(req.input_data, dtype=torch.float32)
                input_tensor = input_tensor.view(req.input_shape)
                batch_inputs.append(input_tensor)
                
                if req.grid_data and req.grid_shape:
                    grid_tensor = torch.tensor(req.grid_data, dtype=torch.float32)
                    grid_tensor = grid_tensor.view(req.grid_shape)
                    batch_grids.append(grid_tensor)
                else:
                    batch_grids.append(None)
            
            # Stack inputs (assuming same shape)
            try:
                batched_input = torch.stack(batch_inputs)
                batched_grid = None
                if all(g is not None for g in batch_grids):
                    batched_grid = torch.stack(batch_grids)
            except:
                # Fallback to individual inference if batching fails
                return [await self.predict(req) for req in requests]
            
            # Batch inference
            model.eval()
            with torch.no_grad():
                if batched_grid is not None:
                    batch_output = model(batched_input, batched_grid)
                else:
                    batch_output = model(batched_input)
            
            # Prepare responses
            inference_time_ms = (time.time() - start_time) * 1000
            model_info = self.registry.models[model_id]
            
            responses = []
            for i, req in enumerate(requests):
                output = batch_output[i]
                prediction = output.cpu().numpy().tolist()
                
                responses.append(InferenceResponse(
                    model_id=model_id,
                    prediction=prediction,
                    prediction_shape=list(output.shape),
                    inference_time_ms=inference_time_ms / len(requests),
                    model_version=model_info['version'],
                    metadata={
                        'batch_size': len(requests),
                        'batch_inference': True
                    }
                ))
            
            # Record metrics
            self.metrics.record_inference(model_id, inference_time_ms / 1000)
            model_info['request_count'] += len(requests)
            
            return responses
            
        except Exception as e:
            logging.error(f"Batch inference error for model {model_id}: {e}")
            # Return errors
            return [
                InferenceResponse(
                    model_id=model_id,
                    prediction=[],
                    prediction_shape=[],
                    inference_time_ms=0.0,
                    model_version="unknown",
                    metadata={"error": str(e)}
                )
                for _ in requests
            ]


# Global instances
registry = ModelRegistry()
metrics = ModelMetrics()
health_checker = HealthChecker(registry)
inference_engine = InferenceEngine(registry, metrics)

# FastAPI app
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logging.info("Starting Neural Operator Production Server")
    
    # Start health checks
    health_task = asyncio.create_task(health_checker.start_health_checks())
    
    # Start metrics updates
    async def update_metrics():
        while True:
            metrics.update_system_metrics()
            await asyncio.sleep(60)  # Update every minute
    
    metrics_task = asyncio.create_task(update_metrics())
    
    yield
    
    # Shutdown
    logging.info("Shutting down Neural Operator Production Server")
    health_checker.stop_health_checks()
    health_task.cancel()
    metrics_task.cancel()


app = FastAPI(
    title="Neural Operator Production API",
    description="Production deployment API for neural operators",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Neural Operator Production API",
        "version": "1.0.0",
        "status": "healthy"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    active_models = len(registry.list_models())
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "active_models": active_models,
        "system": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "gpu_available": torch.cuda.is_available(),
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0
        }
    }


@app.get("/models")
async def list_models():
    """List all deployed models."""
    models = []
    for model_id in registry.list_models():
        model_info = registry.get_model_info(model_id)
        if model_info:
            models.append({
                "model_id": model_id,
                "version": model_info["version"],
                "status": model_info["status"],
                "request_count": model_info["request_count"],
                "error_count": model_info["error_count"],
                "loaded_at": model_info["loaded_at"].isoformat(),
                "last_used": model_info["last_used"].isoformat()
            })
    
    return {"models": models}


@app.get("/models/{model_id}")
async def get_model_info(model_id: str):
    """Get information about a specific model."""
    model_info = registry.get_model_info(model_id)
    if not model_info:
        raise HTTPException(status_code=404, detail=f"Model {model_id} not found")
    
    return {
        "model_id": model_id,
        "version": model_info["version"],
        "status": model_info["status"],
        "request_count": model_info["request_count"],
        "error_count": model_info["error_count"],
        "loaded_at": model_info["loaded_at"].isoformat(),
        "last_used": model_info["last_used"].isoformat(),
        "config": registry.model_configs.get(model_id, {})
    }


@app.post("/predict", response_model=InferenceResponse)
async def predict(request: InferenceRequest):
    """Single inference endpoint."""
    return await inference_engine.predict(request)


@app.post("/predict/batch", response_model=List[InferenceResponse])
async def batch_predict(requests: List[InferenceRequest]):
    """Batch inference endpoint."""
    return await inference_engine.batch_predict(requests)


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(
        generate_latest(),
        media_type=CONTENT_TYPE_LATEST
    )


@app.get("/ab-tests")
async def list_ab_tests():
    """List active A/B tests."""
    tests = []
    for test_name, test_info in registry.ab_tests.items():
        tests.append({
            "test_name": test_name,
            "model_a": test_info["model_a"],
            "model_b": test_info["model_b"],
            "traffic_split": test_info["traffic_split"],
            "requests_a": test_info["requests_a"],
            "requests_b": test_info["requests_b"],
            "created_at": test_info["created_at"].isoformat()
        })
    
    return {"ab_tests": tests}


@app.post("/ab-tests/{test_name}/predict", response_model=InferenceResponse)
async def ab_test_predict(test_name: str, request: InferenceRequest):
    """Inference with A/B testing."""
    # Generate request ID for consistent routing
    request_id = str(uuid.uuid4())
    
    # Get model from A/B test
    selected_model = registry.get_ab_test_model(test_name, request_id)
    if not selected_model:
        raise HTTPException(status_code=404, detail=f"A/B test {test_name} not found")
    
    # Override model ID with selected model
    request.model_id = selected_model
    
    # Perform inference
    response = await inference_engine.predict(request)
    
    # Add A/B test metadata
    response.metadata.update({
        "ab_test": test_name,
        "selected_model": selected_model,
        "request_id": request_id
    })
    
    return response


def start_production_server(host: str = "0.0.0.0", 
                           port: int = 8000,
                           workers: int = 1,
                           reload: bool = False):
    """Start the production server."""
    uvicorn.run(
        "neural_operator_lab.deployment.production_deployment:app",
        host=host,
        port=port,
        workers=workers,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    start_production_server()