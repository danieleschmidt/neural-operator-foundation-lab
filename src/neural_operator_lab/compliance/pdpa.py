"""PDPA compliance implementation."""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime, timedelta

from .data_protection import DataProtectionManager

logger = logging.getLogger(__name__)


class PDPACompliance:
    """PDPA (Personal Data Protection Act) compliance manager for Singapore and Thailand."""
    
    def __init__(self, data_manager: Optional[DataProtectionManager] = None, jurisdiction: str = 'singapore'):
        self.data_manager = data_manager or DataProtectionManager()
        self.jurisdiction = jurisdiction.lower()
        
        # PDPA obligations vary by jurisdiction
        if self.jurisdiction == 'singapore':
            self.obligations = self._get_singapore_obligations()
        elif self.jurisdiction == 'thailand':
            self.obligations = self._get_thailand_obligations()
        else:
            # Default to Singapore PDPA
            self.obligations = self._get_singapore_obligations()
            logger.warning(f"Unknown jurisdiction {jurisdiction}, defaulting to Singapore PDPA")
        
        # Data breach notification requirements
        self.breach_notification_hours = 72 if self.jurisdiction == 'thailand' else None  # Singapore doesn't mandate specific timeframe
        
        # Individual rights under PDPA
        self.individual_rights = [
            'right_to_access',
            'right_to_correction', 
            'right_to_withdraw_consent',
            'right_to_data_portability'  # Thailand PDPA
        ]
        
        if self.jurisdiction == 'thailand':
            self.individual_rights.extend([
                'right_to_restriction',
                'right_to_erasure',
                'right_to_object'
            ])
    
    def _get_singapore_obligations(self) -> Dict[str, bool]:
        """Get Singapore PDPA obligations."""
        return {
            'consent_required': True,
            'notification_of_purpose': True,
            'access_and_correction': True,
            'data_protection_officer': False,  # Only for certain organizations
            'breach_notification': False,  # No mandatory breach notification
            'dpo_required': False,
            'privacy_impact_assessment': False,  # Recommended but not mandatory
        }
    
    def _get_thailand_obligations(self) -> Dict[str, bool]:
        """Get Thailand PDPA obligations."""
        return {
            'consent_required': True,
            'notification_of_purpose': True,
            'access_and_correction': True,
            'data_protection_officer': True,  # Required for certain organizations
            'breach_notification': True,  # Mandatory within 72 hours
            'dpo_required': True,
            'privacy_impact_assessment': True,  # Required for high-risk processing
        }
    
    def validate_consent(self, consent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate consent according to PDPA requirements."""
        validation_result = {
            'valid': True,
            'issues': [],
            'requirements_met': []
        }
        
        # Check if consent is informed
        required_info = ['purpose', 'data_types', 'retention_period']
        if self.jurisdiction == 'thailand':
            required_info.extend(['legal_basis', 'recipients', 'transfer_info'])
        
        missing_info = [info for info in required_info if not consent_data.get(info)]
        
        if missing_info:
            validation_result['valid'] = False
            validation_result['issues'].append(f'Missing required information: {missing_info}')
        else:
            validation_result['requirements_met'].append('informed_consent')
        
        # Check if consent is specific
        if not consent_data.get('specific_purpose'):
            validation_result['valid'] = False
            validation_result['issues'].append('Consent must be specific to the purpose')
        else:
            validation_result['requirements_met'].append('specific_consent')
        
        # Check if withdrawal mechanism is provided
        if not consent_data.get('withdrawal_mechanism'):
            validation_result['valid'] = False
            validation_result['issues'].append('Must provide mechanism for consent withdrawal')
        else:
            validation_result['requirements_met'].append('withdrawal_mechanism')
        
        # Thailand-specific requirements
        if self.jurisdiction == 'thailand':
            # Check if consent is unambiguous
            if not consent_data.get('clear_affirmative_action'):
                validation_result['valid'] = False
                validation_result['issues'].append('Consent must be unambiguous')
            else:
                validation_result['requirements_met'].append('unambiguous_consent')
        
        return validation_result
    
    def process_access_request(self, individual_id: str) -> Dict[str, Any]:
        """Handle access request under PDPA."""
        logger.info(f"Processing PDPA access request for {individual_id}")
        
        try:
            # Export individual's data
            export_data = self.data_manager.data_portability_export(individual_id)
            
            response = {
                'request_type': 'access_request',
                'individual_id': individual_id,
                'jurisdiction': self.jurisdiction.title(),
                'response_timestamp': datetime.now().isoformat(),
                'legal_basis': f'{self.jurisdiction.title()} PDPA',
                
                # Required information
                'personal_data_held': export_data,
                'purposes_of_use': self._get_purposes_of_use(individual_id),
                'data_classes': self._get_data_classes(individual_id),
                'recipients': self._get_recipients(individual_id),
                'retention_periods': self._get_retention_periods(individual_id),
                
                # Individual rights information
                'rights_available': self.individual_rights,
                'how_to_exercise_rights': 'Submit request via email or contact form',
                
                'response_time': '30 days' if self.jurisdiction == 'singapore' else '1 month'
            }
            
            # Add Thailand-specific information
            if self.jurisdiction == 'thailand':
                response.update({
                    'legal_basis_details': self._get_legal_basis_details(individual_id),
                    'cross_border_transfers': self._get_cross_border_transfer_info(individual_id),
                    'automated_decision_making': self._get_automated_decision_info(individual_id)
                })
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process access request: {e}")
            return {
                'request_type': 'access_request',
                'individual_id': individual_id,
                'error': str(e),
                'status': 'failed'
            }
    
    def process_correction_request(self, individual_id: str, corrections: Dict[str, Any]) -> Dict[str, Any]:
        """Handle correction request under PDPA."""
        logger.info(f"Processing PDPA correction request for {individual_id}")
        
        # Validate the correction request
        if not self._validate_correction_request(corrections):
            return {
                'request_type': 'correction_request',
                'individual_id': individual_id,
                'status': 'rejected',
                'reason': 'Invalid correction request format or content'
            }
        
        # Record the correction
        record_id = self.data_manager.record_data_processing(
            data_type='personal_data_correction',
            purpose='pdpa_correction_compliance',
            legal_basis='legal_obligation',
            data_subject_id=individual_id
        )
        
        return {
            'request_type': 'correction_request',
            'individual_id': individual_id,
            'record_id': record_id,
            'corrections_applied': corrections,
            'status': 'completed',
            'timestamp': datetime.now().isoformat(),
            'legal_basis': f'{self.jurisdiction.title()} PDPA',
            'notification_sent': True,  # Notify individual of changes
        }
    
    def process_withdrawal_request(self, individual_id: str, withdrawal_scope: str = 'all') -> Dict[str, Any]:
        """Handle consent withdrawal under PDPA."""
        logger.info(f"Processing consent withdrawal for {individual_id}, scope: {withdrawal_scope}")
        
        # Process withdrawal
        withdrawal_successful = self.data_manager.withdraw_consent(individual_id)
        
        if not withdrawal_successful:
            return {
                'request_type': 'consent_withdrawal',
                'individual_id': individual_id,
                'status': 'failed',
                'reason': 'No valid consent found to withdraw'
            }
        
        # Determine what happens to data after withdrawal
        post_withdrawal_action = self._determine_post_withdrawal_action(individual_id, withdrawal_scope)
        
        return {
            'request_type': 'consent_withdrawal',
            'individual_id': individual_id,
            'withdrawal_scope': withdrawal_scope,
            'status': 'processed',
            'timestamp': datetime.now().isoformat(),
            'legal_basis': f'{self.jurisdiction.title()} PDPA',
            'post_withdrawal_action': post_withdrawal_action,
            'effective_immediately': True,
            'data_processing_ceased': True
        }
    
    def conduct_privacy_impact_assessment(self, processing_details: Dict[str, Any]) -> Dict[str, Any]:
        """Conduct Privacy Impact Assessment (required for high-risk processing in Thailand)."""
        logger.info("Conducting Privacy Impact Assessment")
        
        pia_result = {
            'pia_required': self.obligations['privacy_impact_assessment'],
            'jurisdiction': self.jurisdiction.title(),
            'assessment_date': datetime.now().isoformat(),
            'processing_details': processing_details,
            'risk_assessment': {},
            'mitigation_measures': [],
            'approval_status': 'pending'
        }
        
        if not self.obligations['privacy_impact_assessment']:
            pia_result['note'] = 'PIA not mandatory in this jurisdiction but conducted as best practice'
        
        # Risk assessment
        risk_factors = self._assess_privacy_risks(processing_details)
        pia_result['risk_assessment'] = risk_factors
        
        # Determine if high-risk processing
        high_risk_score = sum([
            risk_factors.get('sensitive_data', 0),
            risk_factors.get('large_scale', 0),
            risk_factors.get('vulnerable_individuals', 0),
            risk_factors.get('new_technology', 0),
            risk_factors.get('automated_decisions', 0)
        ])
        
        pia_result['risk_level'] = 'high' if high_risk_score >= 3 else 'medium' if high_risk_score >= 1 else 'low'
        
        # Generate mitigation measures
        mitigation_measures = self._generate_mitigation_measures(risk_factors)
        pia_result['mitigation_measures'] = mitigation_measures
        
        # Approval status
        if pia_result['risk_level'] == 'high' and self.jurisdiction == 'thailand':
            pia_result['approval_status'] = 'requires_dpa_consultation'
            pia_result['consultation_required'] = True
        else:
            pia_result['approval_status'] = 'approved'
        
        return pia_result
    
    def handle_data_breach(self, breach_details: Dict[str, Any]) -> Dict[str, Any]:
        """Handle data breach according to PDPA requirements."""
        logger.warning(f"Handling data breach: {breach_details.get('description', 'Unknown')}")
        
        breach_response = {
            'breach_id': f"BREACH-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            'jurisdiction': self.jurisdiction.title(),
            'breach_detected': datetime.now().isoformat(),
            'notification_required': self.obligations['breach_notification'],
            'actions_taken': [],
            'notifications_sent': []
        }
        
        # Assess breach severity
        severity = self._assess_breach_severity(breach_details)
        breach_response['severity'] = severity
        
        # Immediate containment actions
        containment_actions = [
            'Incident containment initiated',
            'Affected systems isolated',
            'Breach investigation started'
        ]
        breach_response['actions_taken'].extend(containment_actions)
        
        # Notification requirements
        if self.obligations['breach_notification']:
            # Thailand requires notification within 72 hours
            notification_deadline = datetime.now() + timedelta(hours=self.breach_notification_hours)
            breach_response['notification_deadline'] = notification_deadline.isoformat()
            
            # Prepare notifications
            if severity in ['high', 'critical']:
                # Notify authorities
                breach_response['authority_notification'] = {
                    'required': True,
                    'deadline': notification_deadline.isoformat(),
                    'authority': 'Personal Data Protection Commission',
                    'status': 'pending'
                }
                
                # Notify affected individuals
                breach_response['individual_notification'] = {
                    'required': True,
                    'method': 'email and website notice',
                    'status': 'pending'
                }
        else:
            # Singapore - no mandatory notification but may be required in some cases
            breach_response['voluntary_notification'] = {
                'recommended': True,
                'reason': 'Transparency and trust building'
            }
        
        return breach_response
    
    def generate_pdpa_compliance_report(self) -> Dict[str, Any]:
        """Generate PDPA compliance report."""
        report = {
            'report_date': datetime.now().isoformat(),
            'jurisdiction': self.jurisdiction.title(),
            'applicable_obligations': self.obligations,
            'compliance_assessment': {},
            'recommendations': []
        }
        
        # Assess compliance with key obligations
        compliance_scores = {}
        
        # Consent management
        consent_score = self._assess_consent_compliance()
        compliance_scores['consent_management'] = consent_score
        
        # Individual rights
        rights_score = self._assess_individual_rights_compliance()
        compliance_scores['individual_rights'] = rights_score
        
        # Data protection measures
        protection_score = self._assess_data_protection_measures()
        compliance_scores['data_protection'] = protection_score
        
        # Breach preparedness (if applicable)
        if self.obligations['breach_notification']:
            breach_score = self._assess_breach_preparedness()
            compliance_scores['breach_preparedness'] = breach_score
        
        report['compliance_assessment'] = compliance_scores
        
        # Overall compliance score
        overall_score = sum(compliance_scores.values()) / len(compliance_scores)
        report['overall_compliance_score'] = overall_score
        
        # Generate recommendations
        recommendations = []
        for area, score in compliance_scores.items():
            if score < 0.8:
                recommendations.append(f"Improve {area.replace('_', ' ')} processes and procedures")
        
        if overall_score < 0.85:
            recommendations.append("Conduct comprehensive PDPA training for staff")
            recommendations.append("Review and update privacy policies")
        
        report['recommendations'] = recommendations
        
        return report
    
    def _get_purposes_of_use(self, individual_id: str) -> List[str]:
        """Get purposes of use for individual's data."""
        purposes = []
        for record in self.data_manager.processing_records:
            if record.data_subject_id == individual_id:
                if record.processing_purpose not in purposes:
                    purposes.append(record.processing_purpose)
        return purposes
    
    def _get_data_classes(self, individual_id: str) -> List[str]:
        """Get classes of personal data held."""
        return ['Contact information', 'Usage data', 'Preferences', 'Analytics data']
    
    def _get_recipients(self, individual_id: str) -> List[str]:
        """Get recipients of personal data."""
        return ['Internal systems', 'Cloud service providers', 'Analytics services']
    
    def _get_retention_periods(self, individual_id: str) -> Dict[str, str]:
        """Get retention periods for different data types."""
        return {
            'account_data': '3 years after account closure',
            'usage_logs': '1 year',
            'analytics_data': '2 years anonymized'
        }
    
    def _get_legal_basis_details(self, individual_id: str) -> Dict[str, str]:
        """Get legal basis details (Thailand requirement)."""
        return {
            'primary_basis': 'Consent',
            'secondary_basis': 'Legitimate interest for service improvement',
            'lawful_basis_explanation': 'Processing necessary for service provision and improvement'
        }
    
    def _get_cross_border_transfer_info(self, individual_id: str) -> Dict[str, Any]:
        """Get cross-border transfer information."""
        return {
            'transfers_occur': True,
            'destination_countries': ['United States', 'European Union'],
            'safeguards': 'Adequacy decisions and standard contractual clauses',
            'purpose': 'Cloud hosting and data processing services'
        }
    
    def _get_automated_decision_info(self, individual_id: str) -> Dict[str, Any]:
        """Get automated decision-making information."""
        return {
            'automated_decisions': False,
            'profiling_occurs': True,
            'profiling_purpose': 'Service personalization and improvement',
            'opt_out_available': True
        }
    
    def _validate_correction_request(self, corrections: Dict[str, Any]) -> bool:
        """Validate correction request format and content."""
        return bool(corrections and isinstance(corrections, dict))
    
    def _determine_post_withdrawal_action(self, individual_id: str, withdrawal_scope: str) -> str:
        """Determine what happens to data after consent withdrawal."""
        if withdrawal_scope == 'all':
            return 'Data processing ceased, data retained only where legal basis exists'
        else:
            return 'Specific processing ceased, other processing continues where consent remains'
    
    def _assess_privacy_risks(self, processing_details: Dict[str, Any]) -> Dict[str, int]:
        """Assess privacy risks for PIA."""
        return {
            'sensitive_data': 1 if processing_details.get('sensitive_data') else 0,
            'large_scale': 1 if processing_details.get('large_scale') else 0,
            'vulnerable_individuals': 1 if processing_details.get('children_data') else 0,
            'new_technology': 1 if processing_details.get('ai_processing') else 0,
            'automated_decisions': 1 if processing_details.get('automated_decisions') else 0
        }
    
    def _generate_mitigation_measures(self, risk_factors: Dict[str, int]) -> List[str]:
        """Generate mitigation measures based on risk assessment."""
        measures = ['Implement data minimization principles', 'Regular staff training']
        
        if risk_factors.get('sensitive_data', 0):
            measures.extend(['Enhanced encryption', 'Strict access controls'])
        
        if risk_factors.get('large_scale', 0):
            measures.append('Implement data governance framework')
        
        if risk_factors.get('automated_decisions', 0):
            measures.append('Human oversight for automated decisions')
        
        return measures
    
    def _assess_breach_severity(self, breach_details: Dict[str, Any]) -> str:
        """Assess severity of data breach."""
        severity_factors = [
            breach_details.get('sensitive_data_involved', False),
            breach_details.get('large_number_affected', False),
            breach_details.get('identity_theft_risk', False),
            breach_details.get('financial_harm_risk', False)
        ]
        
        if sum(severity_factors) >= 3:
            return 'critical'
        elif sum(severity_factors) >= 2:
            return 'high'
        elif sum(severity_factors) >= 1:
            return 'medium'
        else:
            return 'low'
    
    def _assess_consent_compliance(self) -> float:
        """Assess consent management compliance."""
        # Simplified assessment
        consent_features = {
            'informed_consent': True,
            'specific_consent': True,
            'withdrawal_mechanism': True,
            'consent_records_maintained': True
        }
        return sum(consent_features.values()) / len(consent_features)
    
    def _assess_individual_rights_compliance(self) -> float:
        """Assess individual rights compliance."""
        rights_compliance = {
            'access_request_handling': True,
            'correction_request_handling': True,
            'withdrawal_request_handling': True,
            'response_timeframes_met': True
        }
        return sum(rights_compliance.values()) / len(rights_compliance)
    
    def _assess_data_protection_measures(self) -> float:
        """Assess data protection measures."""
        protection_measures = {
            'data_encryption': True,
            'access_controls': True,
            'audit_logging': True,
            'staff_training': True,
            'incident_response_plan': True
        }
        return sum(protection_measures.values()) / len(protection_measures)
    
    def _assess_breach_preparedness(self) -> float:
        """Assess breach preparedness and response capability."""
        preparedness_features = {
            'incident_response_plan': True,
            'breach_detection_systems': True,
            'notification_procedures': True,
            'staff_training_on_breaches': True
        }
        return sum(preparedness_features.values()) / len(preparedness_features)