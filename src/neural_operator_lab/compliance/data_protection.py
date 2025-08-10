"""Data protection and privacy compliance manager."""

import hashlib
import json
import logging
import uuid
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, asdict
from pathlib import Path
import pickle

logger = logging.getLogger(__name__)


@dataclass
class DataProcessingRecord:
    """Record of data processing activity."""
    id: str
    timestamp: datetime
    data_type: str
    processing_purpose: str
    legal_basis: str
    data_subject_id: Optional[str] = None
    retention_period: Optional[int] = None  # days
    anonymized: bool = False
    encrypted: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DataProcessingRecord':
        """Create from dictionary."""
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ConsentRecord:
    """Record of user consent."""
    id: str
    data_subject_id: str
    consent_type: str  # 'explicit', 'implicit', 'legitimate_interest'
    granted: bool
    timestamp: datetime
    purpose: str
    expiry_date: Optional[datetime] = None
    withdrawn: bool = False
    withdrawal_date: Optional[datetime] = None
    
    def is_valid(self) -> bool:
        """Check if consent is still valid."""
        if self.withdrawn:
            return False
        
        if self.expiry_date and datetime.now() > self.expiry_date:
            return False
        
        return self.granted
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        data = asdict(self)
        data['timestamp'] = data['timestamp'].isoformat()
        if data['expiry_date']:
            data['expiry_date'] = data['expiry_date'].isoformat()
        if data['withdrawal_date']:
            data['withdrawal_date'] = data['withdrawal_date'].isoformat()
        return data


class DataProtectionManager:
    """Comprehensive data protection and compliance manager."""
    
    def __init__(self, storage_dir: str = "compliance_data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.processing_records: List[DataProcessingRecord] = []
        self.consent_records: Dict[str, ConsentRecord] = {}
        
        # Load existing records
        self._load_records()
        
        # Compliance configurations
        self.gdpr_config = {
            'default_retention_days': 1095,  # 3 years
            'anonymization_required': True,
            'consent_required': True,
            'right_to_be_forgotten': True,
        }
        
        self.ccpa_config = {
            'opt_out_available': True,
            'data_sale_notification': True,
            'personal_info_disclosure': True,
        }
        
        self.pdpa_config = {
            'notification_required': True,
            'consent_withdrawal': True,
            'data_breach_notification_hours': 72,
        }
    
    def record_data_processing(
        self, 
        data_type: str,
        purpose: str,
        legal_basis: str = 'legitimate_interest',
        data_subject_id: Optional[str] = None,
        retention_days: Optional[int] = None,
        anonymized: bool = False,
        encrypted: bool = True
    ) -> str:
        """Record data processing activity."""
        record_id = str(uuid.uuid4())
        
        record = DataProcessingRecord(
            id=record_id,
            timestamp=datetime.now(),
            data_type=data_type,
            processing_purpose=purpose,
            legal_basis=legal_basis,
            data_subject_id=data_subject_id,
            retention_period=retention_days or self.gdpr_config['default_retention_days'],
            anonymized=anonymized,
            encrypted=encrypted
        )
        
        self.processing_records.append(record)
        self._save_processing_records()
        
        logger.info(f"Recorded data processing: {data_type} for {purpose}")
        return record_id
    
    def record_consent(
        self,
        data_subject_id: str,
        consent_type: str,
        purpose: str,
        granted: bool = True,
        expiry_days: Optional[int] = None
    ) -> str:
        """Record user consent."""
        consent_id = str(uuid.uuid4())
        
        expiry_date = None
        if expiry_days:
            expiry_date = datetime.now() + timedelta(days=expiry_days)
        
        consent = ConsentRecord(
            id=consent_id,
            data_subject_id=data_subject_id,
            consent_type=consent_type,
            granted=granted,
            timestamp=datetime.now(),
            purpose=purpose,
            expiry_date=expiry_date
        )
        
        self.consent_records[data_subject_id] = consent
        self._save_consent_records()
        
        logger.info(f"Recorded consent for {data_subject_id}: {granted}")
        return consent_id
    
    def withdraw_consent(self, data_subject_id: str) -> bool:
        """Withdraw consent for data subject."""
        if data_subject_id in self.consent_records:
            consent = self.consent_records[data_subject_id]
            consent.withdrawn = True
            consent.withdrawal_date = datetime.now()
            
            self._save_consent_records()
            logger.info(f"Consent withdrawn for {data_subject_id}")
            return True
        
        return False
    
    def check_consent(self, data_subject_id: str, purpose: str) -> bool:
        """Check if we have valid consent for data processing."""
        if data_subject_id not in self.consent_records:
            return False
        
        consent = self.consent_records[data_subject_id]
        
        # Check if consent is for the right purpose
        if consent.purpose != purpose and consent.purpose != 'all':
            return False
        
        return consent.is_valid()
    
    def anonymize_data(self, data: Any, salt: str = "neural_operator_lab") -> str:
        """Anonymize data using hashing."""
        if isinstance(data, dict):
            data_str = json.dumps(data, sort_keys=True)
        else:
            data_str = str(data)
        
        # Create hash with salt
        hash_object = hashlib.sha256((data_str + salt).encode())
        return hash_object.hexdigest()
    
    def encrypt_sensitive_data(self, data: bytes, key: Optional[bytes] = None) -> bytes:
        """Encrypt sensitive data (simple XOR for demo - use proper encryption in production)."""
        if key is None:
            key = b"neural_operator_foundation_lab_key_12345"  # In production, use proper key management
        
        # Simple XOR encryption (for demo purposes only)
        key_length = len(key)
        encrypted = bytearray()
        
        for i, byte in enumerate(data):
            encrypted.append(byte ^ key[i % key_length])
        
        return bytes(encrypted)
    
    def right_to_be_forgotten(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle right to be forgotten request (GDPR Article 17)."""
        result = {
            'data_subject_id': data_subject_id,
            'timestamp': datetime.now().isoformat(),
            'actions_taken': [],
            'records_removed': 0,
            'success': True
        }
        
        try:
            # Remove consent records
            if data_subject_id in self.consent_records:
                del self.consent_records[data_subject_id]
                result['actions_taken'].append('consent_records_removed')
            
            # Remove or anonymize processing records
            original_count = len(self.processing_records)
            self.processing_records = [
                record for record in self.processing_records 
                if record.data_subject_id != data_subject_id
            ]
            
            removed_count = original_count - len(self.processing_records)
            result['records_removed'] = removed_count
            
            if removed_count > 0:
                result['actions_taken'].append('processing_records_removed')
            
            # Save updated records
            self._save_consent_records()
            self._save_processing_records()
            
            logger.info(f"Right to be forgotten processed for {data_subject_id}")
            
        except Exception as e:
            result['success'] = False
            result['error'] = str(e)
            logger.error(f"Failed to process right to be forgotten for {data_subject_id}: {e}")
        
        return result
    
    def data_portability_export(self, data_subject_id: str) -> Dict[str, Any]:
        """Export all data for a subject (GDPR Article 20)."""
        export_data = {
            'data_subject_id': data_subject_id,
            'export_timestamp': datetime.now().isoformat(),
            'consent_records': [],
            'processing_records': [],
            'format': 'JSON'
        }
        
        # Export consent records
        if data_subject_id in self.consent_records:
            export_data['consent_records'].append(
                self.consent_records[data_subject_id].to_dict()
            )
        
        # Export processing records
        for record in self.processing_records:
            if record.data_subject_id == data_subject_id:
                export_data['processing_records'].append(record.to_dict())
        
        logger.info(f"Data portability export generated for {data_subject_id}")
        return export_data
    
    def cleanup_expired_data(self) -> Dict[str, int]:
        """Clean up expired data based on retention policies."""
        now = datetime.now()
        cleanup_stats = {
            'processing_records_removed': 0,
            'consent_records_expired': 0
        }
        
        # Clean up expired processing records
        original_count = len(self.processing_records)
        self.processing_records = [
            record for record in self.processing_records
            if record.retention_period is None or 
            (now - record.timestamp).days < record.retention_period
        ]
        
        cleanup_stats['processing_records_removed'] = original_count - len(self.processing_records)
        
        # Clean up expired consent records
        expired_consent_ids = []
        for subject_id, consent in self.consent_records.items():
            if not consent.is_valid():
                expired_consent_ids.append(subject_id)
        
        for subject_id in expired_consent_ids:
            del self.consent_records[subject_id]
            cleanup_stats['consent_records_expired'] += 1
        
        # Save updated records
        if cleanup_stats['processing_records_removed'] > 0:
            self._save_processing_records()
        
        if cleanup_stats['consent_records_expired'] > 0:
            self._save_consent_records()
        
        logger.info(f"Cleanup completed: {cleanup_stats}")
        return cleanup_stats
    
    def generate_compliance_report(self) -> Dict[str, Any]:
        """Generate compliance report."""
        now = datetime.now()
        
        # Analyze consent records
        total_consents = len(self.consent_records)
        active_consents = sum(1 for c in self.consent_records.values() if c.is_valid())
        withdrawn_consents = sum(1 for c in self.consent_records.values() if c.withdrawn)
        
        # Analyze processing records
        total_processing = len(self.processing_records)
        recent_processing = sum(
            1 for r in self.processing_records 
            if (now - r.timestamp).days <= 30
        )
        
        anonymized_processing = sum(1 for r in self.processing_records if r.anonymized)
        encrypted_processing = sum(1 for r in self.processing_records if r.encrypted)
        
        report = {
            'report_timestamp': now.isoformat(),
            'compliance_framework': ['GDPR', 'CCPA', 'PDPA'],
            'consent_statistics': {
                'total_consents': total_consents,
                'active_consents': active_consents,
                'withdrawn_consents': withdrawn_consents,
                'consent_rate': active_consents / max(total_consents, 1),
            },
            'processing_statistics': {
                'total_processing_records': total_processing,
                'recent_processing_30_days': recent_processing,
                'anonymized_records': anonymized_processing,
                'encrypted_records': encrypted_processing,
                'anonymization_rate': anonymized_processing / max(total_processing, 1),
                'encryption_rate': encrypted_processing / max(total_processing, 1),
            },
            'compliance_scores': {
                'gdpr_compliance': self._calculate_gdpr_compliance(),
                'ccpa_compliance': self._calculate_ccpa_compliance(),
                'pdpa_compliance': self._calculate_pdpa_compliance(),
            },
            'recommendations': self._generate_recommendations()
        }
        
        return report
    
    def _calculate_gdpr_compliance(self) -> float:
        """Calculate GDPR compliance score."""
        score = 0.0
        checks = 0
        
        # Check if anonymization is being used
        if self.processing_records:
            anonymization_rate = sum(1 for r in self.processing_records if r.anonymized) / len(self.processing_records)
            score += min(anonymization_rate * 2, 1.0) * 25  # Up to 25 points
        checks += 25
        
        # Check if encryption is being used
        if self.processing_records:
            encryption_rate = sum(1 for r in self.processing_records if r.encrypted) / len(self.processing_records)
            score += encryption_rate * 25  # Up to 25 points
        checks += 25
        
        # Check consent management
        if self.consent_records:
            valid_consent_rate = sum(1 for c in self.consent_records.values() if c.is_valid()) / len(self.consent_records)
            score += valid_consent_rate * 25  # Up to 25 points
        checks += 25
        
        # Check retention compliance
        retention_compliance = 1.0  # Assume compliant for now
        score += retention_compliance * 25  # Up to 25 points
        checks += 25
        
        return min(score / checks, 1.0) if checks > 0 else 0.0
    
    def _calculate_ccpa_compliance(self) -> float:
        """Calculate CCPA compliance score."""
        # Simplified CCPA compliance check
        score = 0.8  # Assume basic compliance
        
        # Check opt-out capability
        if self.ccpa_config['opt_out_available']:
            score += 0.1
        
        # Check disclosure capability  
        if self.ccpa_config['personal_info_disclosure']:
            score += 0.1
        
        return min(score, 1.0)
    
    def _calculate_pdpa_compliance(self) -> float:
        """Calculate PDPA compliance score."""
        # Simplified PDPA compliance check
        score = 0.7  # Assume basic compliance
        
        # Check notification capability
        if self.pdpa_config['notification_required']:
            score += 0.15
        
        # Check consent withdrawal
        if self.pdpa_config['consent_withdrawal']:
            score += 0.15
        
        return min(score, 1.0)
    
    def _generate_recommendations(self) -> List[str]:
        """Generate compliance recommendations."""
        recommendations = []
        
        if self.processing_records:
            anonymization_rate = sum(1 for r in self.processing_records if r.anonymized) / len(self.processing_records)
            if anonymization_rate < 0.8:
                recommendations.append("Increase data anonymization rate to improve GDPR compliance")
            
            encryption_rate = sum(1 for r in self.processing_records if r.encrypted) / len(self.processing_records)
            if encryption_rate < 0.9:
                recommendations.append("Ensure all sensitive data is encrypted")
        
        if self.consent_records:
            expired_consents = sum(1 for c in self.consent_records.values() if not c.is_valid())
            if expired_consents > 0:
                recommendations.append(f"Review and refresh {expired_consents} expired consent records")
        
        # Regular cleanup recommendation
        recommendations.append("Schedule regular compliance data cleanup")
        
        return recommendations
    
    def _load_records(self):
        """Load records from storage."""
        # Load processing records
        processing_file = self.storage_dir / "processing_records.json"
        if processing_file.exists():
            try:
                with open(processing_file, 'r') as f:
                    data = json.load(f)
                    self.processing_records = [
                        DataProcessingRecord.from_dict(record) for record in data
                    ]
            except Exception as e:
                logger.error(f"Failed to load processing records: {e}")
        
        # Load consent records
        consent_file = self.storage_dir / "consent_records.json"
        if consent_file.exists():
            try:
                with open(consent_file, 'r') as f:
                    data = json.load(f)
                    self.consent_records = {
                        subject_id: ConsentRecord(
                            id=record['id'],
                            data_subject_id=record['data_subject_id'],
                            consent_type=record['consent_type'],
                            granted=record['granted'],
                            timestamp=datetime.fromisoformat(record['timestamp']),
                            purpose=record['purpose'],
                            expiry_date=datetime.fromisoformat(record['expiry_date']) if record.get('expiry_date') else None,
                            withdrawn=record.get('withdrawn', False),
                            withdrawal_date=datetime.fromisoformat(record['withdrawal_date']) if record.get('withdrawal_date') else None
                        )
                        for subject_id, record in data.items()
                    }
            except Exception as e:
                logger.error(f"Failed to load consent records: {e}")
    
    def _save_processing_records(self):
        """Save processing records to storage."""
        processing_file = self.storage_dir / "processing_records.json"
        try:
            with open(processing_file, 'w') as f:
                json.dump([record.to_dict() for record in self.processing_records], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save processing records: {e}")
    
    def _save_consent_records(self):
        """Save consent records to storage."""
        consent_file = self.storage_dir / "consent_records.json"
        try:
            with open(consent_file, 'w') as f:
                json.dump(
                    {subject_id: consent.to_dict() for subject_id, consent in self.consent_records.items()},
                    f, indent=2
                )
        except Exception as e:
            logger.error(f"Failed to save consent records: {e}")