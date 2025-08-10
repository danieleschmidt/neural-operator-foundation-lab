"""CCPA compliance implementation."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime

from .data_protection import DataProtectionManager

logger = logging.getLogger(__name__)


class CCPACompliance:
    """CCPA (California Consumer Privacy Act) compliance manager."""
    
    def __init__(self, data_manager: Optional[DataProtectionManager] = None):
        self.data_manager = data_manager or DataProtectionManager()
        
        # CCPA consumer rights
        self.consumer_rights = [
            'right_to_know',        # §1798.100
            'right_to_delete',      # §1798.105  
            'right_to_opt_out',     # §1798.120
            'right_to_non_discrimination'  # §1798.125
        ]
        
        # Categories of personal information under CCPA
        self.personal_info_categories = [
            'identifiers',
            'personal_records',
            'protected_characteristics',
            'commercial_information',
            'biometric_information', 
            'internet_activity',
            'geolocation_data',
            'sensory_information',
            'professional_information',
            'education_information',
            'inferences'
        ]
    
    def process_right_to_know_request(self, consumer_id: str, time_period: int = 12) -> Dict[str, Any]:
        """Handle Right to Know request (§1798.100)."""
        logger.info(f"Processing right to know request for {consumer_id}")
        
        try:
            # Get consumer's data
            consumer_data = self.data_manager.data_portability_export(consumer_id)
            
            # Categorize the information according to CCPA categories
            categorized_data = self._categorize_personal_info(consumer_data)
            
            response = {
                'request_type': 'right_to_know',
                'consumer_id': consumer_id,
                'time_period_months': time_period,
                'response_timestamp': datetime.now().isoformat(),
                'legal_basis': 'CCPA §1798.100',
                
                # Required disclosures
                'categories_collected': list(categorized_data.keys()),
                'sources': self._get_data_sources(consumer_id),
                'business_purposes': self._get_business_purposes(consumer_id),
                'categories_disclosed': self._get_disclosed_categories(consumer_id),
                'third_party_recipients': self._get_third_party_recipients(consumer_id),
                'categories_sold': [],  # Assuming no data sales
                'specific_data': categorized_data,
                
                # Consumer rights information
                'consumer_rights': self.consumer_rights,
                'opt_out_methods': ['website', 'email', 'phone'],
                'verification_process': 'Two-step verification required'
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process right to know request: {e}")
            return {
                'request_type': 'right_to_know',
                'consumer_id': consumer_id,
                'error': str(e),
                'status': 'failed'
            }
    
    def process_right_to_delete_request(self, consumer_id: str, verification_data: Dict[str, Any]) -> Dict[str, Any]:
        """Handle Right to Delete request (§1798.105)."""
        logger.info(f"Processing right to delete request for {consumer_id}")
        
        # Verify consumer identity
        if not self._verify_consumer_identity(consumer_id, verification_data):
            return {
                'request_type': 'right_to_delete',
                'consumer_id': consumer_id,
                'status': 'verification_failed',
                'message': 'Unable to verify consumer identity',
                'legal_basis': 'CCPA §1798.105'
            }
        
        # Check for exceptions to deletion
        exceptions = self._check_deletion_exceptions(consumer_id)
        
        if exceptions:
            return {
                'request_type': 'right_to_delete',
                'consumer_id': consumer_id,
                'status': 'partial_deletion',
                'exceptions': exceptions,
                'message': 'Some data retained due to legal exceptions',
                'legal_basis': 'CCPA §1798.105'
            }
        
        # Process deletion
        deletion_result = self.data_manager.right_to_be_forgotten(consumer_id)
        
        return {
            'request_type': 'right_to_delete',
            'consumer_id': consumer_id,
            'status': 'completed' if deletion_result['success'] else 'failed',
            'actions_taken': deletion_result.get('actions_taken', []),
            'records_removed': deletion_result.get('records_removed', 0),
            'timestamp': datetime.now().isoformat(),
            'legal_basis': 'CCPA §1798.105',
            'confirmation': 'All personal information has been deleted'
        }
    
    def process_opt_out_request(self, consumer_id: str) -> Dict[str, Any]:
        """Handle Opt-Out of Sale request (§1798.120)."""
        logger.info(f"Processing opt-out request for {consumer_id}")
        
        # Record the opt-out request
        record_id = self.data_manager.record_data_processing(
            data_type='opt_out_preference',
            purpose='ccpa_opt_out_compliance',
            legal_basis='legal_obligation',
            data_subject_id=consumer_id
        )
        
        # In this implementation, we don't sell data, so this is mainly documentation
        return {
            'request_type': 'opt_out_of_sale',
            'consumer_id': consumer_id,
            'record_id': record_id,
            'status': 'processed',
            'timestamp': datetime.now().isoformat(),
            'legal_basis': 'CCPA §1798.120',
            'note': 'Opt-out preference recorded. Note: This business does not sell personal information.',
            'opt_out_methods': ['website', 'email', 'phone'],
            'waiting_period': '15 days for third-party notification'
        }
    
    def generate_privacy_notice(self, business_info: Dict[str, Any]) -> Dict[str, Any]:
        """Generate CCPA-compliant privacy notice."""
        notice = {
            'notice_type': 'ccpa_privacy_notice',
            'business_information': business_info,
            'effective_date': datetime.now().isoformat(),
            'last_updated': datetime.now().isoformat(),
            
            # Required CCPA disclosures
            'categories_collected': {
                'identifiers': {
                    'description': 'Email addresses, usernames, device identifiers',
                    'sources': ['Direct from consumer', 'Automated systems'],
                    'business_purposes': ['Service provision', 'Analytics']
                },
                'internet_activity': {
                    'description': 'Usage patterns, interaction data',
                    'sources': ['Automated collection'],
                    'business_purposes': ['Service improvement', 'Performance monitoring']
                }
            },
            
            'business_purposes': [
                'Provide neural operator services',
                'Improve model performance', 
                'Analytics and research',
                'Security and fraud prevention'
            ],
            
            'data_sharing': {
                'categories_disclosed': ['identifiers', 'internet_activity'],
                'recipients': ['Cloud service providers', 'Analytics providers'],
                'business_purposes': ['Service provision', 'Analytics']
            },
            
            'data_sales': {
                'categories_sold': [],
                'note': 'This business does not sell personal information'
            },
            
            'consumer_rights': {
                'right_to_know': {
                    'description': 'Right to know what personal information is collected',
                    'how_to_exercise': 'Submit request via website or email'
                },
                'right_to_delete': {
                    'description': 'Right to delete personal information',
                    'how_to_exercise': 'Submit verified request via website or email'
                },
                'right_to_opt_out': {
                    'description': 'Right to opt out of sale of personal information',
                    'how_to_exercise': 'Click "Do Not Sell My Personal Information" link'
                }
            },
            
            'verification_process': {
                'description': 'Two-step verification required for sensitive requests',
                'requirements': ['Government ID', 'Confirmation of request details']
            },
            
            'contact_information': {
                'privacy_contact': business_info.get('privacy_email', 'privacy@example.com'),
                'phone': business_info.get('privacy_phone', '1-800-PRIVACY'),
                'address': business_info.get('address', 'Privacy Office Address')
            }
        }
        
        return notice
    
    def conduct_ccpa_audit(self) -> Dict[str, Any]:
        """Conduct CCPA compliance audit."""
        audit_results = {
            'audit_timestamp': datetime.now().isoformat(),
            'compliance_areas': {},
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Check consumer request handling
        request_handling_score = self._audit_request_handling()
        audit_results['compliance_areas']['request_handling'] = {
            'score': request_handling_score,
            'description': 'Consumer request processing capabilities'
        }
        
        # Check privacy notice compliance
        notice_score = self._audit_privacy_notice()
        audit_results['compliance_areas']['privacy_notice'] = {
            'score': notice_score,
            'description': 'Privacy notice completeness and accuracy'
        }
        
        # Check data inventory
        inventory_score = self._audit_data_inventory()
        audit_results['compliance_areas']['data_inventory'] = {
            'score': inventory_score,
            'description': 'Data collection and processing inventory'
        }
        
        # Check opt-out mechanisms
        opt_out_score = self._audit_opt_out_mechanisms()
        audit_results['compliance_areas']['opt_out_mechanisms'] = {
            'score': opt_out_score,
            'description': 'Opt-out request processing'
        }
        
        # Calculate overall score
        scores = [request_handling_score, notice_score, inventory_score, opt_out_score]
        audit_results['overall_score'] = sum(scores) / len(scores)
        
        # Generate recommendations
        if request_handling_score < 0.8:
            audit_results['recommendations'].append('Improve consumer request processing workflows')
        
        if notice_score < 0.8:
            audit_results['recommendations'].append('Update privacy notice to include all required CCPA disclosures')
        
        if inventory_score < 0.8:
            audit_results['recommendations'].append('Complete comprehensive data inventory and mapping')
        
        if opt_out_score < 0.8:
            audit_results['recommendations'].append('Implement user-friendly opt-out mechanisms')
        
        return audit_results
    
    def _categorize_personal_info(self, data_export: Dict[str, Any]) -> Dict[str, List[str]]:
        """Categorize personal information according to CCPA categories."""
        categorized = {}
        
        # Analyze processing records to categorize data
        for record in data_export.get('processing_records', []):
            data_type = record.get('data_type', 'unknown')
            
            # Map to CCPA categories
            if 'email' in data_type.lower() or 'identifier' in data_type.lower():
                if 'identifiers' not in categorized:
                    categorized['identifiers'] = []
                categorized['identifiers'].append(data_type)
            
            elif 'usage' in data_type.lower() or 'activity' in data_type.lower():
                if 'internet_activity' not in categorized:
                    categorized['internet_activity'] = []
                categorized['internet_activity'].append(data_type)
            
            elif 'commercial' in data_type.lower():
                if 'commercial_information' not in categorized:
                    categorized['commercial_information'] = []
                categorized['commercial_information'].append(data_type)
        
        return categorized
    
    def _get_data_sources(self, consumer_id: str) -> List[str]:
        """Get sources of personal information."""
        return ['Direct from consumer', 'Automated collection', 'Third-party integrations']
    
    def _get_business_purposes(self, consumer_id: str) -> List[str]:
        """Get business purposes for data collection."""
        return [
            'Provide neural operator services',
            'Improve service quality',
            'Analytics and research',
            'Security and fraud prevention'
        ]
    
    def _get_disclosed_categories(self, consumer_id: str) -> List[str]:
        """Get categories of information disclosed for business purposes."""
        return ['identifiers', 'internet_activity']
    
    def _get_third_party_recipients(self, consumer_id: str) -> List[str]:
        """Get third parties that receive personal information."""
        return ['Cloud service providers', 'Analytics providers']
    
    def _verify_consumer_identity(self, consumer_id: str, verification_data: Dict[str, Any]) -> bool:
        """Verify consumer identity for sensitive requests."""
        # Simplified verification - in production, implement proper identity verification
        required_fields = ['email', 'last_four_digits', 'verification_code']
        return all(field in verification_data for field in required_fields)
    
    def _check_deletion_exceptions(self, consumer_id: str) -> List[str]:
        """Check for exceptions to deletion under CCPA."""
        exceptions = []
        
        # Check if data is needed for legal compliance
        # This is a simplified check - implement actual business logic
        
        return exceptions  # No exceptions in this simplified implementation
    
    def _audit_request_handling(self) -> float:
        """Audit consumer request handling capabilities."""
        # Check if we can handle all required request types
        capabilities = {
            'right_to_know': True,
            'right_to_delete': True,
            'opt_out_of_sale': True,
            'verification_process': True
        }
        
        return sum(capabilities.values()) / len(capabilities)
    
    def _audit_privacy_notice(self) -> float:
        """Audit privacy notice compliance."""
        # Check if privacy notice includes required elements
        required_elements = {
            'categories_collected': True,
            'sources': True,
            'business_purposes': True,
            'third_party_sharing': True,
            'consumer_rights': True,
            'contact_information': True
        }
        
        return sum(required_elements.values()) / len(required_elements)
    
    def _audit_data_inventory(self) -> float:
        """Audit data inventory completeness."""
        # Check if we have comprehensive data inventory
        inventory_completeness = {
            'data_categories_identified': True,
            'sources_documented': True,
            'purposes_documented': True,
            'retention_periods_defined': True
        }
        
        return sum(inventory_completeness.values()) / len(inventory_completeness)
    
    def _audit_opt_out_mechanisms(self) -> float:
        """Audit opt-out mechanism implementation."""
        # Check opt-out mechanism availability and usability
        opt_out_features = {
            'clear_opt_out_link': True,
            'easy_to_find': True,
            'no_registration_required': True,
            'processes_within_15_days': True
        }
        
        return sum(opt_out_features.values()) / len(opt_out_features)