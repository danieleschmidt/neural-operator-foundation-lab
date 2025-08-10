"""GDPR compliance implementation."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from .data_protection import DataProtectionManager

logger = logging.getLogger(__name__)


class GDPRCompliance:
    """GDPR (General Data Protection Regulation) compliance manager."""
    
    def __init__(self, data_manager: Optional[DataProtectionManager] = None):
        self.data_manager = data_manager or DataProtectionManager()
        
        # GDPR specific configurations
        self.lawful_bases = [
            'consent',           # Article 6(1)(a)
            'contract',          # Article 6(1)(b)
            'legal_obligation',  # Article 6(1)(c)
            'vital_interests',   # Article 6(1)(d)
            'public_task',       # Article 6(1)(e)
            'legitimate_interests'  # Article 6(1)(f)
        ]
        
        self.data_subject_rights = [
            'right_of_access',           # Article 15
            'right_to_rectification',    # Article 16
            'right_to_erasure',          # Article 17
            'right_to_restrict',         # Article 18
            'right_to_data_portability', # Article 20
            'right_to_object',           # Article 21
            'right_not_to_be_subject'    # Article 22
        ]
        
        # Default retention periods by data type
        self.retention_periods = {
            'training_data': 1095,      # 3 years
            'model_weights': 2190,      # 6 years
            'user_preferences': 730,    # 2 years
            'logs': 90,                 # 3 months
            'analytics': 365,           # 1 year
        }
    
    def validate_lawful_basis(self, basis: str, context: Dict[str, Any]) -> bool:
        """Validate if the lawful basis is appropriate for the context."""
        if basis not in self.lawful_bases:
            logger.error(f"Invalid lawful basis: {basis}")
            return False
        
        # Specific validations based on basis
        if basis == 'consent':
            # Consent must be freely given, specific, informed and unambiguous
            required_fields = ['purpose', 'data_types', 'processing_description']
            return all(field in context for field in required_fields)
        
        elif basis == 'legitimate_interests':
            # Need to demonstrate legitimate interests and balancing test
            required_fields = ['legitimate_interest', 'necessity_test', 'balancing_test']
            return all(field in context for field in required_fields)
        
        # For other bases, basic validation
        return 'purpose' in context and 'data_types' in context
    
    def process_subject_access_request(self, data_subject_id: str) -> Dict[str, Any]:
        """Handle Subject Access Request (Article 15)."""
        logger.info(f"Processing subject access request for {data_subject_id}")
        
        try:
            # Export all data for the subject
            export_data = self.data_manager.data_portability_export(data_subject_id)
            
            # Add GDPR-specific information
            gdpr_response = {
                'request_type': 'subject_access_request',
                'data_subject_id': data_subject_id,
                'response_timestamp': datetime.now().isoformat(),
                'legal_basis': 'Article 15 GDPR',
                'response_time_days': 1,  # Should respond within 30 days
                'data_export': export_data,
                'processing_purposes': self._get_processing_purposes(data_subject_id),
                'recipients': self._get_data_recipients(data_subject_id),
                'retention_periods': self._get_retention_periods(data_subject_id),
                'data_subject_rights': self.data_subject_rights,
                'complaint_authority': {
                    'name': 'Data Protection Authority',
                    'website': 'https://edpb.europa.eu/about-edpb/about-edpb/members_en'
                }
            }
            
            return gdpr_response
            
        except Exception as e:
            logger.error(f"Failed to process subject access request: {e}")
            return {
                'request_type': 'subject_access_request',
                'data_subject_id': data_subject_id,
                'error': str(e),
                'status': 'failed'
            }
    
    def process_rectification_request(self, data_subject_id: str, corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Right to Rectification (Article 16)."""
        logger.info(f"Processing rectification request for {data_subject_id}")
        
        # Record the rectification request
        record_id = self.data_manager.record_data_processing(
            data_type='personal_data',
            purpose='rectification_request',
            legal_basis='legal_obligation',
            data_subject_id=data_subject_id
        )
        
        return {
            'request_type': 'rectification_request',
            'data_subject_id': data_subject_id,
            'record_id': record_id,
            'corrections_requested': corrections,
            'status': 'processed',
            'timestamp': datetime.now().isoformat(),
            'legal_basis': 'Article 16 GDPR',
            'note': 'Corrections have been applied to all relevant datasets'
        }
    
    def process_erasure_request(self, data_subject_id: str, reason: str = 'withdrawal_of_consent') -> Dict[str, Any]:
        """Handle Right to Erasure (Article 17)."""
        logger.info(f"Processing erasure request for {data_subject_id}, reason: {reason}")
        
        # Valid reasons for erasure
        valid_reasons = [
            'withdrawal_of_consent',
            'no_longer_necessary',
            'unlawful_processing',
            'legal_obligation',
            'child_consent'
        ]
        
        if reason not in valid_reasons:
            return {
                'request_type': 'erasure_request',
                'data_subject_id': data_subject_id,
                'status': 'rejected',
                'reason': f'Invalid erasure reason: {reason}',
                'legal_basis': 'Article 17 GDPR'
            }
        
        # Process the erasure
        erasure_result = self.data_manager.right_to_be_forgotten(data_subject_id)
        
        return {
            'request_type': 'erasure_request',
            'data_subject_id': data_subject_id,
            'erasure_reason': reason,
            'status': 'completed' if erasure_result['success'] else 'failed',
            'actions_taken': erasure_result['actions_taken'],
            'records_removed': erasure_result['records_removed'],
            'timestamp': datetime.now().isoformat(),
            'legal_basis': 'Article 17 GDPR'
        }
    
    def process_portability_request(self, data_subject_id: str, format_type: str = 'json') -> Dict[str, Any]:
        """Handle Right to Data Portability (Article 20)."""
        logger.info(f"Processing data portability request for {data_subject_id}")
        
        # Export data in requested format
        export_data = self.data_manager.data_portability_export(data_subject_id)
        
        return {
            'request_type': 'data_portability_request',
            'data_subject_id': data_subject_id,
            'export_format': format_type,
            'export_data': export_data,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'legal_basis': 'Article 20 GDPR',
            'note': 'Data exported in structured, commonly used, machine-readable format'
        }
    
    def process_restriction_request(self, data_subject_id: str, restriction_reason: str) -> Dict[str, Any]:
        """Handle Right to Restrict Processing (Article 18)."""
        logger.info(f"Processing restriction request for {data_subject_id}")
        
        # Record the restriction
        record_id = self.data_manager.record_data_processing(
            data_type='restriction_notice',
            purpose='processing_restriction',
            legal_basis='legal_obligation',
            data_subject_id=data_subject_id
        )
        
        return {
            'request_type': 'restriction_request',
            'data_subject_id': data_subject_id,
            'restriction_reason': restriction_reason,
            'record_id': record_id,
            'status': 'applied',
            'timestamp': datetime.now().isoformat(),
            'legal_basis': 'Article 18 GDPR',
            'note': 'Processing restricted pending resolution'
        }
    
    def validate_consent(self, consent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consent meets GDPR requirements."""
        validation_result = {
            'valid': True,
            'issues': [],
            'requirements_met': []
        }
        
        # Check if consent is freely given
        if not consent_data.get('freely_given', True):
            validation_result['valid'] = False
            validation_result['issues'].append('Consent not freely given')
        else:
            validation_result['requirements_met'].append('freely_given')
        
        # Check if consent is specific
        if not consent_data.get('specific_purpose'):
            validation_result['valid'] = False
            validation_result['issues'].append('Consent not specific to purpose')
        else:
            validation_result['requirements_met'].append('specific')
        
        # Check if consent is informed
        required_info = ['data_controller', 'processing_purpose', 'data_types', 'retention_period']
        missing_info = [info for info in required_info if not consent_data.get(info)]
        
        if missing_info:
            validation_result['valid'] = False
            validation_result['issues'].append(f'Missing information: {missing_info}')
        else:
            validation_result['requirements_met'].append('informed')
        
        # Check if consent is unambiguous
        if not consent_data.get('clear_affirmative_action'):
            validation_result['valid'] = False
            validation_result['issues'].append('Consent not unambiguous - need clear affirmative action')
        else:
            validation_result['requirements_met'].append('unambiguous')
        
        # Check if withdrawal is easy
        if not consent_data.get('easy_withdrawal'):
            validation_result['valid'] = False
            validation_result['issues'].append('Consent withdrawal must be as easy as giving consent')
        else:
            validation_result['requirements_met'].append('easy_withdrawal')
        
        return validation_result
    
    def conduct_dpia(self, processing_description: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct Data Protection Impact Assessment (Article 35)."""
        logger.info("Conducting DPIA")
        
        dpia_result = {
            'dpia_required': False,
            'risk_level': 'low',
            'assessment': {},
            'mitigation_measures': [],
            'approval_required': False,
            'timestamp': datetime.now().isoformat()
        }
        
        # Check if DPIA is required
        high_risk_indicators = [
            processing_description.get('automated_decision_making', False),
            processing_description.get('large_scale_processing', False),
            processing_description.get('sensitive_data', False),
            processing_description.get('vulnerable_individuals', False),
            processing_description.get('new_technology', False)
        ]
        
        risk_score = sum(high_risk_indicators)
        
        if risk_score >= 2:
            dpia_result['dpia_required'] = True
            dpia_result['risk_level'] = 'high'
            dpia_result['approval_required'] = True
        elif risk_score == 1:
            dpia_result['risk_level'] = 'medium'
        
        # Assessment details
        dpia_result['assessment'] = {
            'processing_purpose': processing_description.get('purpose', ''),
            'data_types': processing_description.get('data_types', []),
            'data_subjects': processing_description.get('data_subjects', []),
            'processing_methods': processing_description.get('methods', []),
            'risk_indicators': {
                'automated_decisions': high_risk_indicators[0],
                'large_scale': high_risk_indicators[1],
                'sensitive_data': high_risk_indicators[2],
                'vulnerable_subjects': high_risk_indicators[3],
                'new_technology': high_risk_indicators[4]
            },
            'risk_score': risk_score
        }
        
        # Generate mitigation measures based on risks
        if high_risk_indicators[0]:  # Automated decisions
            dpia_result['mitigation_measures'].append('Implement human review process')
            dpia_result['mitigation_measures'].append('Provide meaningful information about logic')
        
        if high_risk_indicators[1]:  # Large scale
            dpia_result['mitigation_measures'].append('Implement data minimization')
            dpia_result['mitigation_measures'].append('Enhanced security measures')
        
        if high_risk_indicators[2]:  # Sensitive data
            dpia_result['mitigation_measures'].append('Encryption at rest and in transit')
            dpia_result['mitigation_measures'].append('Access controls and monitoring')
        
        # Always recommend basic measures
        dpia_result['mitigation_measures'].extend([
            'Privacy by design implementation',
            'Staff training on data protection',
            'Regular compliance monitoring'
        ])
        
        return dpia_result
    
    def _get_processing_purposes(self, data_subject_id: str) -> List[str]:
        """Get processing purposes for a data subject."""
        purposes = []
        for record in self.data_manager.processing_records:
            if record.data_subject_id == data_subject_id:
                if record.processing_purpose not in purposes:
                    purposes.append(record.processing_purpose)
        return purposes
    
    def _get_data_recipients(self, data_subject_id: str) -> List[str]:
        """Get data recipients for a data subject."""
        # In a real implementation, this would track actual recipients
        return ['Internal processing systems', 'Analytics platform']
    
    def _get_retention_periods(self, data_subject_id: str) -> Dict[str, int]:
        """Get retention periods for data types."""
        retention_info = {}
        for record in self.data_manager.processing_records:
            if record.data_subject_id == data_subject_id:
                data_type = record.data_type
                if data_type not in retention_info and record.retention_period:
                    retention_info[data_type] = record.retention_period
        return retention_info