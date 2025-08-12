"""Security audit logging and monitoring."""

import json
import time
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import threading

logger = logging.getLogger(__name__)


class SecurityEventType(Enum):
    """Types of security events to audit."""
    
    LOGIN_ATTEMPT = "login_attempt"
    LOGIN_SUCCESS = "login_success" 
    LOGIN_FAILURE = "login_failure"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    MODEL_LOADED = "model_loaded"
    MODEL_SAVED = "model_saved"
    DATA_ACCESSED = "data_accessed"
    ENCRYPTION_USED = "encryption_used"
    VALIDATION_FAILED = "validation_failed"
    SUSPICIOUS_ACTIVITY = "suspicious_activity"
    CONFIGURATION_CHANGED = "configuration_changed"


@dataclass
class SecurityEvent:
    """Security event data structure."""
    
    event_type: SecurityEventType
    timestamp: float
    user_id: Optional[str] = None
    source_ip: Optional[str] = None
    resource: Optional[str] = None
    details: Optional[Dict[str, Any]] = None
    severity: str = "INFO"  # INFO, WARNING, ERROR, CRITICAL
    session_id: Optional[str] = None


class AuditLogger:
    """Security audit logger."""
    
    def __init__(
        self,
        log_file: Optional[str] = None,
        max_events: int = 10000,
        enable_console_logging: bool = True
    ):
        """Initialize audit logger.
        
        Args:
            log_file: Optional file path for audit logs
            max_events: Maximum events to keep in memory
            enable_console_logging: Whether to log to console
        """
        self.log_file = log_file
        self.max_events = max_events
        self.enable_console_logging = enable_console_logging
        
        self._events: List[SecurityEvent] = []
        self._lock = threading.Lock()
        
        # Setup file logging if specified
        if log_file:
            self._setup_file_logging()
        
        logger.info("Security audit logger initialized")
    
    def _setup_file_logging(self) -> None:
        """Setup file logging for audit events."""
        audit_logger = logging.getLogger('security_audit')
        audit_logger.setLevel(logging.INFO)
        
        # Create file handler
        file_handler = logging.FileHandler(self.log_file)
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter(
            '%(asctime)s - SECURITY_AUDIT - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        audit_logger.addHandler(file_handler)
        self._audit_logger = audit_logger
    
    def log_event(
        self,
        event_type: SecurityEventType,
        user_id: Optional[str] = None,
        source_ip: Optional[str] = None,
        resource: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        severity: str = "INFO",
        session_id: Optional[str] = None
    ) -> None:
        """Log a security event.
        
        Args:
            event_type: Type of security event
            user_id: User ID associated with event
            source_ip: Source IP address
            resource: Resource being accessed
            details: Additional event details
            severity: Event severity level
            session_id: Session ID
        """
        event = SecurityEvent(
            event_type=event_type,
            timestamp=time.time(),
            user_id=user_id,
            source_ip=source_ip,
            resource=resource,
            details=details or {},
            severity=severity,
            session_id=session_id
        )
        
        with self._lock:
            self._events.append(event)
            
            # Maintain max events limit
            if len(self._events) > self.max_events:
                self._events.pop(0)
        
        # Log to file if configured
        if hasattr(self, '_audit_logger'):
            self._log_to_file(event)
        
        # Log to console if enabled
        if self.enable_console_logging:
            self._log_to_console(event)
    
    def _log_to_file(self, event: SecurityEvent) -> None:
        """Log event to file."""
        log_data = {
            'event_type': event.event_type.value,
            'timestamp': event.timestamp,
            'datetime': datetime.fromtimestamp(event.timestamp).isoformat(),
            'user_id': event.user_id,
            'source_ip': event.source_ip,
            'resource': event.resource,
            'details': event.details,
            'severity': event.severity,
            'session_id': event.session_id
        }
        
        log_message = json.dumps(log_data)
        
        if event.severity == "CRITICAL":
            self._audit_logger.critical(log_message)
        elif event.severity == "ERROR":
            self._audit_logger.error(log_message)
        elif event.severity == "WARNING":
            self._audit_logger.warning(log_message)
        else:
            self._audit_logger.info(log_message)
    
    def _log_to_console(self, event: SecurityEvent) -> None:
        """Log event to console."""
        message = (
            f"SECURITY [{event.severity}] {event.event_type.value} - "
            f"User: {event.user_id or 'unknown'}, "
            f"Resource: {event.resource or 'none'}"
        )
        
        if event.severity == "CRITICAL":
            logger.critical(message)
        elif event.severity == "ERROR":
            logger.error(message)
        elif event.severity == "WARNING":
            logger.warning(message)
        else:
            logger.info(message)
    
    def get_recent_events(
        self,
        count: int = 100,
        event_type: Optional[SecurityEventType] = None,
        severity: Optional[str] = None
    ) -> List[SecurityEvent]:
        """Get recent security events.
        
        Args:
            count: Number of events to return
            event_type: Filter by event type
            severity: Filter by severity
            
        Returns:
            List of security events
        """
        with self._lock:
            events = self._events.copy()
        
        # Apply filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        if severity:
            events = [e for e in events if e.severity == severity]
        
        # Return most recent events
        return events[-count:]
    
    def get_events_by_user(self, user_id: str, count: int = 50) -> List[SecurityEvent]:
        """Get events for specific user.
        
        Args:
            user_id: User ID to filter by
            count: Number of events to return
            
        Returns:
            List of security events for user
        """
        with self._lock:
            user_events = [e for e in self._events if e.user_id == user_id]
        
        return user_events[-count:]
    
    def clear_events(self) -> None:
        """Clear all stored events."""
        with self._lock:
            self._events.clear()
        
        logger.info("Security audit events cleared")


class SecurityAuditor:
    """Security auditor for comprehensive monitoring."""
    
    def __init__(self, audit_logger: Optional[AuditLogger] = None):
        """Initialize security auditor.
        
        Args:
            audit_logger: Optional audit logger. Creates new if None.
        """
        self.audit_logger = audit_logger or AuditLogger()
        self._suspicious_activity_threshold = 5  # Failed attempts per minute
        self._activity_tracking: Dict[str, List[float]] = {}
        self._lock = threading.Lock()
        
        logger.info("Security auditor initialized")
    
    def audit_model_access(
        self,
        model_path: str,
        operation: str,
        user_id: Optional[str] = None,
        success: bool = True
    ) -> None:
        """Audit model access operations.
        
        Args:
            model_path: Path to model being accessed
            operation: Operation being performed (load, save, etc.)
            user_id: User performing operation
            success: Whether operation was successful
        """
        event_type = SecurityEventType.MODEL_LOADED if operation == "load" else SecurityEventType.MODEL_SAVED
        severity = "INFO" if success else "WARNING"
        
        details = {
            'operation': operation,
            'success': success,
            'model_path': model_path
        }
        
        self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            resource=model_path,
            details=details,
            severity=severity
        )
    
    def audit_data_access(
        self,
        data_source: str,
        user_id: Optional[str] = None,
        access_granted: bool = True,
        sensitive_data: bool = False
    ) -> None:
        """Audit data access operations.
        
        Args:
            data_source: Data source being accessed
            user_id: User accessing data
            access_granted: Whether access was granted
            sensitive_data: Whether data is marked as sensitive
        """
        event_type = SecurityEventType.ACCESS_GRANTED if access_granted else SecurityEventType.ACCESS_DENIED
        severity = "WARNING" if sensitive_data and access_granted else "INFO"
        
        details = {
            'data_source': data_source,
            'access_granted': access_granted,
            'sensitive_data': sensitive_data
        }
        
        self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            resource=data_source,
            details=details,
            severity=severity
        )
    
    def audit_authentication(
        self,
        user_id: str,
        success: bool,
        source_ip: Optional[str] = None,
        session_id: Optional[str] = None
    ) -> None:
        """Audit authentication attempts.
        
        Args:
            user_id: User ID attempting authentication
            success: Whether authentication was successful
            source_ip: Source IP address
            session_id: Session ID
        """
        event_type = SecurityEventType.LOGIN_SUCCESS if success else SecurityEventType.LOGIN_FAILURE
        severity = "INFO" if success else "WARNING"
        
        details = {
            'authentication_success': success,
            'source_ip': source_ip
        }
        
        self.audit_logger.log_event(
            event_type=event_type,
            user_id=user_id,
            source_ip=source_ip,
            details=details,
            severity=severity,
            session_id=session_id
        )
        
        # Track suspicious activity
        if not success:
            self._track_failed_attempt(user_id or source_ip or "unknown")
    
    def audit_validation_failure(
        self,
        input_data: str,
        validation_rule: str,
        user_id: Optional[str] = None
    ) -> None:
        """Audit input validation failures.
        
        Args:
            input_data: Data that failed validation (truncated/masked)
            validation_rule: Validation rule that failed
            user_id: User who provided invalid input
        """
        details = {
            'input_preview': input_data[:100] + "..." if len(input_data) > 100 else input_data,
            'validation_rule': validation_rule
        }
        
        self.audit_logger.log_event(
            event_type=SecurityEventType.VALIDATION_FAILED,
            user_id=user_id,
            details=details,
            severity="WARNING"
        )
    
    def _track_failed_attempt(self, identifier: str) -> None:
        """Track failed attempts for suspicious activity detection.
        
        Args:
            identifier: User ID or IP address
        """
        current_time = time.time()
        
        with self._lock:
            if identifier not in self._activity_tracking:
                self._activity_tracking[identifier] = []
            
            # Add current attempt
            self._activity_tracking[identifier].append(current_time)
            
            # Clean old attempts (older than 1 minute)
            cutoff_time = current_time - 60
            self._activity_tracking[identifier] = [
                t for t in self._activity_tracking[identifier] if t > cutoff_time
            ]
            
            # Check for suspicious activity
            if len(self._activity_tracking[identifier]) >= self._suspicious_activity_threshold:
                self._flag_suspicious_activity(identifier)
    
    def _flag_suspicious_activity(self, identifier: str) -> None:
        """Flag suspicious activity.
        
        Args:
            identifier: User ID or IP address showing suspicious activity
        """
        details = {
            'identifier': identifier,
            'failed_attempts_per_minute': len(self._activity_tracking[identifier]),
            'threshold': self._suspicious_activity_threshold
        }
        
        self.audit_logger.log_event(
            event_type=SecurityEventType.SUSPICIOUS_ACTIVITY,
            user_id=identifier if "@" in identifier else None,
            source_ip=identifier if "." in identifier else None,
            details=details,
            severity="CRITICAL"
        )
        
        logger.critical(f"Suspicious activity detected for {identifier}")
    
    def generate_security_report(self, hours: int = 24) -> Dict[str, Any]:
        """Generate security report for specified time period.
        
        Args:
            hours: Number of hours to include in report
            
        Returns:
            Security report dictionary
        """
        cutoff_time = time.time() - (hours * 3600)
        
        # Get all events in time window
        all_events = self.audit_logger.get_recent_events(count=10000)
        recent_events = [e for e in all_events if e.timestamp > cutoff_time]
        
        # Generate statistics
        event_counts = {}
        severity_counts = {}
        user_activity = {}
        
        for event in recent_events:
            # Count by event type
            event_type = event.event_type.value
            event_counts[event_type] = event_counts.get(event_type, 0) + 1
            
            # Count by severity
            severity_counts[event.severity] = severity_counts.get(event.severity, 0) + 1
            
            # Track user activity
            if event.user_id:
                if event.user_id not in user_activity:
                    user_activity[event.user_id] = {'events': 0, 'last_activity': 0}
                user_activity[event.user_id]['events'] += 1
                user_activity[event.user_id]['last_activity'] = max(
                    user_activity[event.user_id]['last_activity'],
                    event.timestamp
                )
        
        report = {
            'report_generated': datetime.now().isoformat(),
            'time_period_hours': hours,
            'total_events': len(recent_events),
            'event_type_breakdown': event_counts,
            'severity_breakdown': severity_counts,
            'user_activity': user_activity,
            'critical_events': len([e for e in recent_events if e.severity == "CRITICAL"]),
            'error_events': len([e for e in recent_events if e.severity == "ERROR"])
        }
        
        return report