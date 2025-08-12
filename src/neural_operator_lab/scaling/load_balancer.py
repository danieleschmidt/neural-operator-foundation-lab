"""Load balancing for distributed neural operator inference."""

import threading
import time
import random
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass
from enum import Enum
import logging
from collections import deque, defaultdict

logger = logging.getLogger(__name__)


class LoadBalancingStrategy(Enum):
    """Load balancing strategies."""
    
    ROUND_ROBIN = "round_robin"
    LEAST_CONNECTIONS = "least_connections"
    WEIGHTED_ROUND_ROBIN = "weighted_round_robin"
    LEAST_RESPONSE_TIME = "least_response_time"
    RESOURCE_BASED = "resource_based"


@dataclass
class ServerInfo:
    """Information about a server/worker."""
    
    server_id: str
    host: str
    port: int
    weight: float = 1.0
    active_connections: int = 0
    total_requests: int = 0
    total_response_time: float = 0.0
    cpu_usage: float = 0.0
    memory_usage: float = 0.0
    gpu_usage: float = 0.0
    is_healthy: bool = True
    last_health_check: float = 0.0


class HealthChecker:
    """Health checker for servers."""
    
    def __init__(self, check_interval: float = 30.0, timeout: float = 5.0):
        """Initialize health checker.
        
        Args:
            check_interval: Health check interval in seconds
            timeout: Health check timeout in seconds
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self._servers: Dict[str, ServerInfo] = {}
        self._health_check_thread: Optional[threading.Thread] = None
        self._running = False
        
        logger.info(f"Health checker initialized - interval: {check_interval}s")
    
    def add_server(self, server: ServerInfo) -> None:
        """Add server to health monitoring.
        
        Args:
            server: Server information
        """
        self._servers[server.server_id] = server
        logger.info(f"Added server {server.server_id} to health monitoring")
    
    def remove_server(self, server_id: str) -> None:
        """Remove server from health monitoring.
        
        Args:
            server_id: Server ID to remove
        """
        if server_id in self._servers:
            del self._servers[server_id]
            logger.info(f"Removed server {server_id} from health monitoring")
    
    def start_monitoring(self) -> None:
        """Start health monitoring."""
        if self._running:
            return
        
        self._running = True
        self._health_check_thread = threading.Thread(
            target=self._health_check_loop,
            daemon=True
        )
        self._health_check_thread.start()
        logger.info("Health monitoring started")
    
    def stop_monitoring(self) -> None:
        """Stop health monitoring."""
        self._running = False
        if self._health_check_thread:
            self._health_check_thread.join(timeout=5.0)
        logger.info("Health monitoring stopped")
    
    def _health_check_loop(self) -> None:
        """Health check loop."""
        while self._running:
            try:
                self._perform_health_checks()
            except Exception as e:
                logger.error(f"Health check error: {e}")
            
            time.sleep(self.check_interval)
    
    def _perform_health_checks(self) -> None:
        """Perform health checks on all servers."""
        current_time = time.time()
        
        for server in self._servers.values():
            try:
                # Simulate health check (would make actual HTTP request in real implementation)
                is_healthy = self._check_server_health(server)
                server.is_healthy = is_healthy
                server.last_health_check = current_time
                
                if not is_healthy:
                    logger.warning(f"Server {server.server_id} marked as unhealthy")
                
            except Exception as e:
                logger.error(f"Health check failed for server {server.server_id}: {e}")
                server.is_healthy = False
                server.last_health_check = current_time
    
    def _check_server_health(self, server: ServerInfo) -> bool:
        """Check health of individual server.
        
        Args:
            server: Server to check
            
        Returns:
            True if server is healthy
        """
        try:
            # Simulate health check with random success/failure
            # In real implementation, this would make HTTP request to health endpoint
            success_rate = 0.95  # 95% success rate
            is_healthy = random.random() < success_rate
            
            # Update resource metrics (simulated)
            if is_healthy:
                server.cpu_usage = random.uniform(0.1, 0.8)
                server.memory_usage = random.uniform(0.2, 0.7)
                server.gpu_usage = random.uniform(0.0, 0.9)
            
            return is_healthy
            
        except Exception as e:
            logger.error(f"Health check exception for {server.server_id}: {e}")
            return False
    
    def get_healthy_servers(self) -> List[ServerInfo]:
        """Get list of healthy servers.
        
        Returns:
            List of healthy servers
        """
        return [server for server in self._servers.values() if server.is_healthy]


class LoadBalancer:
    """Load balancer for neural operator inference requests."""
    
    def __init__(
        self,
        strategy: LoadBalancingStrategy = LoadBalancingStrategy.LEAST_CONNECTIONS,
        enable_health_checks: bool = True
    ):
        """Initialize load balancer.
        
        Args:
            strategy: Load balancing strategy
            enable_health_checks: Whether to enable health monitoring
        """
        self.strategy = strategy
        self.servers: Dict[str, ServerInfo] = {}
        self._round_robin_index = 0
        self._lock = threading.Lock()
        
        # Health monitoring
        self.health_checker = HealthChecker() if enable_health_checks else None
        
        # Request tracking
        self._request_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=100))
        
        logger.info(f"Load balancer initialized - strategy: {strategy.value}")
    
    def add_server(
        self,
        server_id: str,
        host: str,
        port: int,
        weight: float = 1.0
    ) -> None:
        """Add server to load balancer.
        
        Args:
            server_id: Unique server identifier
            host: Server host/IP
            port: Server port
            weight: Server weight for weighted strategies
        """
        server = ServerInfo(
            server_id=server_id,
            host=host,
            port=port,
            weight=weight
        )
        
        with self._lock:
            self.servers[server_id] = server
        
        if self.health_checker:
            self.health_checker.add_server(server)
        
        logger.info(f"Added server {server_id} ({host}:{port}) with weight {weight}")
    
    def remove_server(self, server_id: str) -> None:
        """Remove server from load balancer.
        
        Args:
            server_id: Server ID to remove
        """
        with self._lock:
            if server_id in self.servers:
                del self.servers[server_id]
        
        if self.health_checker:
            self.health_checker.remove_server(server_id)
        
        logger.info(f"Removed server {server_id}")
    
    def get_next_server(self) -> Optional[ServerInfo]:
        """Get next server according to load balancing strategy.
        
        Returns:
            Selected server or None if no healthy servers available
        """
        # Get healthy servers
        healthy_servers = self._get_healthy_servers()
        
        if not healthy_servers:
            logger.warning("No healthy servers available")
            return None
        
        # Apply load balancing strategy
        if self.strategy == LoadBalancingStrategy.ROUND_ROBIN:
            return self._round_robin_select(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_CONNECTIONS:
            return self._least_connections_select(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.WEIGHTED_ROUND_ROBIN:
            return self._weighted_round_robin_select(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.LEAST_RESPONSE_TIME:
            return self._least_response_time_select(healthy_servers)
        elif self.strategy == LoadBalancingStrategy.RESOURCE_BASED:
            return self._resource_based_select(healthy_servers)
        else:
            # Default to round robin
            return self._round_robin_select(healthy_servers)
    
    def _get_healthy_servers(self) -> List[ServerInfo]:
        """Get list of healthy servers."""
        if self.health_checker:
            return self.health_checker.get_healthy_servers()
        else:
            with self._lock:
                return list(self.servers.values())
    
    def _round_robin_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """Round robin server selection."""
        with self._lock:
            server = servers[self._round_robin_index % len(servers)]
            self._round_robin_index += 1
        return server
    
    def _least_connections_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """Least connections server selection."""
        return min(servers, key=lambda s: s.active_connections)
    
    def _weighted_round_robin_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """Weighted round robin server selection."""
        # Create weighted list
        weighted_servers = []
        for server in servers:
            weight_count = max(1, int(server.weight * 10))  # Scale weights
            weighted_servers.extend([server] * weight_count)
        
        with self._lock:
            server = weighted_servers[self._round_robin_index % len(weighted_servers)]
            self._round_robin_index += 1
        
        return server
    
    def _least_response_time_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """Least response time server selection."""
        def avg_response_time(server):
            if server.total_requests == 0:
                return 0.0
            return server.total_response_time / server.total_requests
        
        return min(servers, key=avg_response_time)
    
    def _resource_based_select(self, servers: List[ServerInfo]) -> ServerInfo:
        """Resource-based server selection."""
        def resource_score(server):
            # Lower score is better (less loaded)
            cpu_score = server.cpu_usage
            memory_score = server.memory_usage
            gpu_score = server.gpu_usage
            connection_score = server.active_connections / 100.0  # Normalize
            
            # Weighted combination
            return (cpu_score * 0.3 + 
                   memory_score * 0.3 + 
                   gpu_score * 0.2 + 
                   connection_score * 0.2)
        
        return min(servers, key=resource_score)
    
    def record_request_start(self, server_id: str) -> None:
        """Record start of request processing.
        
        Args:
            server_id: Server handling the request
        """
        with self._lock:
            if server_id in self.servers:
                self.servers[server_id].active_connections += 1
                self.servers[server_id].total_requests += 1
    
    def record_request_end(
        self,
        server_id: str,
        response_time: float,
        success: bool = True
    ) -> None:
        """Record end of request processing.
        
        Args:
            server_id: Server that handled the request
            response_time: Response time in seconds
            success: Whether request was successful
        """
        with self._lock:
            if server_id in self.servers:
                server = self.servers[server_id]
                server.active_connections = max(0, server.active_connections - 1)
                
                if success:
                    server.total_response_time += response_time
                
                # Record in history
                self._request_history[server_id].append({
                    'timestamp': time.time(),
                    'response_time': response_time,
                    'success': success
                })
    
    def get_server_stats(self) -> Dict[str, Dict[str, Any]]:
        """Get statistics for all servers.
        
        Returns:
            Dictionary of server statistics
        """
        stats = {}
        
        with self._lock:
            for server_id, server in self.servers.items():
                avg_response_time = 0.0
                if server.total_requests > 0:
                    avg_response_time = server.total_response_time / server.total_requests
                
                stats[server_id] = {
                    'host': server.host,
                    'port': server.port,
                    'weight': server.weight,
                    'active_connections': server.active_connections,
                    'total_requests': server.total_requests,
                    'avg_response_time': avg_response_time,
                    'cpu_usage': server.cpu_usage,
                    'memory_usage': server.memory_usage,
                    'gpu_usage': server.gpu_usage,
                    'is_healthy': server.is_healthy,
                    'last_health_check': server.last_health_check
                }
        
        return stats
    
    def start_health_monitoring(self) -> None:
        """Start health monitoring if enabled."""
        if self.health_checker:
            self.health_checker.start_monitoring()
    
    def stop_health_monitoring(self) -> None:
        """Stop health monitoring if enabled."""
        if self.health_checker:
            self.health_checker.stop_monitoring()
    
    def __enter__(self):
        """Context manager entry."""
        self.start_health_monitoring()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_health_monitoring()