"""Global deployment orchestration for neural operators."""

import asyncio
import json
import logging
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
from enum import Enum
import os
from pathlib import Path

# Import compliance modules
try:
    from ..compliance.gdpr import GDPRCompliance
    from ..compliance.ccpa import CCPACompliance
    from ..compliance.pdpa import PDPACompliance
    HAS_COMPLIANCE = True
except ImportError:
    HAS_COMPLIANCE = False

# Import i18n modules
try:
    from ..i18n.config import I18nConfig, supported_languages
    from ..i18n.translator import Translator
    HAS_I18N = True
except ImportError:
    HAS_I18N = False


class DeploymentRegion(Enum):
    """Global deployment regions."""
    US_EAST = "us-east-1"
    US_WEST = "us-west-2"
    EU_WEST = "eu-west-1"
    EU_CENTRAL = "eu-central-1"
    ASIA_PACIFIC = "ap-southeast-1"
    ASIA_NORTHEAST = "ap-northeast-1"
    CANADA = "ca-central-1"
    AUSTRALIA = "ap-southeast-2"
    BRAZIL = "sa-east-1"
    INDIA = "ap-south-1"


@dataclass
class RegionConfig:
    """Configuration for deployment region."""
    region: DeploymentRegion
    primary_language: str
    compliance_requirements: List[str]
    data_residency: bool = True
    encryption_required: bool = True
    local_compute_only: bool = False
    max_latency_ms: int = 100
    
    def __post_init__(self):
        # Set compliance requirements based on region
        if not self.compliance_requirements:
            if self.region.value.startswith('eu-'):
                self.compliance_requirements = ['GDPR']
            elif self.region.value.startswith('us-'):
                self.compliance_requirements = ['CCPA']
            elif self.region.value.startswith('ap-'):
                self.compliance_requirements = ['PDPA']


@dataclass
class GlobalDeploymentConfig:
    """Global deployment configuration."""
    primary_region: DeploymentRegion = DeploymentRegion.US_EAST
    regions: List[RegionConfig] = None
    enable_multi_region: bool = True
    enable_auto_failover: bool = True
    global_load_balancing: bool = True
    cross_region_replication: bool = True
    compliance_mode: str = "strict"  # strict, standard, minimal
    data_localization: bool = True
    encryption_in_transit: bool = True
    encryption_at_rest: bool = True
    
    def __post_init__(self):
        if self.regions is None:
            self.regions = [
                RegionConfig(DeploymentRegion.US_EAST, 'en', []),
                RegionConfig(DeploymentRegion.EU_WEST, 'en', ['GDPR']),
                RegionConfig(DeploymentRegion.ASIA_PACIFIC, 'en', ['PDPA']),
            ]


class ComplianceOrchestrator:
    """Orchestrate compliance across multiple jurisdictions."""
    
    def __init__(self):
        self.compliance_managers = {}
        self.logger = logging.getLogger(__name__)
        
        if HAS_COMPLIANCE:
            self.compliance_managers['GDPR'] = GDPRCompliance()
            self.compliance_managers['CCPA'] = CCPACompliance()
            self.compliance_managers['PDPA'] = PDPACompliance()
    
    def validate_data_processing(self, data: Dict[str, Any], region: DeploymentRegion) -> Dict[str, Any]:
        """Validate data processing for specific region."""
        validation_results = {
            'compliant': True,
            'violations': [],
            'recommendations': [],
            'applied_policies': []
        }
        
        # Get region compliance requirements
        compliance_reqs = self._get_compliance_requirements(region)
        
        for requirement in compliance_reqs:
            if requirement in self.compliance_managers:
                manager = self.compliance_managers[requirement]
                
                try:
                    # Validate with specific compliance framework
                    if requirement == 'GDPR':
                        result = self._validate_gdpr(data, manager)
                    elif requirement == 'CCPA':
                        result = self._validate_ccpa(data, manager)
                    elif requirement == 'PDPA':
                        result = self._validate_pdpa(data, manager)
                    
                    validation_results['applied_policies'].append(requirement)
                    
                    if not result.get('compliant', True):
                        validation_results['compliant'] = False
                        validation_results['violations'].extend(result.get('violations', []))
                    
                    validation_results['recommendations'].extend(result.get('recommendations', []))
                
                except Exception as e:
                    self.logger.error(f"Compliance validation error for {requirement}: {e}")
                    validation_results['violations'].append(f"{requirement} validation failed: {e}")
                    validation_results['compliant'] = False
        
        return validation_results
    
    def _get_compliance_requirements(self, region: DeploymentRegion) -> List[str]:
        """Get compliance requirements for region."""
        region_compliance = {
            DeploymentRegion.US_EAST: ['CCPA'],
            DeploymentRegion.US_WEST: ['CCPA'],
            DeploymentRegion.EU_WEST: ['GDPR'],
            DeploymentRegion.EU_CENTRAL: ['GDPR'],
            DeploymentRegion.ASIA_PACIFIC: ['PDPA'],
            DeploymentRegion.ASIA_NORTHEAST: ['PDPA'],
            DeploymentRegion.CANADA: ['GDPR'],  # Similar to GDPR
            DeploymentRegion.AUSTRALIA: ['PDPA'],  # Similar to PDPA
            DeploymentRegion.BRAZIL: ['LGPD'],  # Lei Geral de Proteção de Dados
            DeploymentRegion.INDIA: ['PDPA']
        }
        
        return region_compliance.get(region, [])
    
    def _validate_gdpr(self, data: Dict[str, Any], manager) -> Dict[str, Any]:
        """Validate GDPR compliance."""
        try:
            # Check for personal data
            has_personal_data = self._contains_personal_data(data)
            
            if has_personal_data:
                # Validate lawful basis
                lawful_basis = data.get('lawful_basis', 'legitimate_interests')
                if not manager.validate_lawful_basis(lawful_basis, data):
                    return {
                        'compliant': False,
                        'violations': ['Missing or invalid lawful basis for processing'],
                        'recommendations': ['Ensure valid lawful basis is established']
                    }
                
                # Check consent if required
                if lawful_basis == 'consent':
                    consent = data.get('consent_status', False)
                    if not consent:
                        return {
                            'compliant': False,
                            'violations': ['Consent required but not obtained'],
                            'recommendations': ['Obtain explicit consent before processing']
                        }
            
            return {'compliant': True, 'violations': [], 'recommendations': []}
        
        except Exception as e:
            return {
                'compliant': False,
                'violations': [f'GDPR validation error: {e}'],
                'recommendations': ['Review GDPR compliance implementation']
            }
    
    def _validate_ccpa(self, data: Dict[str, Any], manager) -> Dict[str, Any]:
        """Validate CCPA compliance."""
        try:
            # CCPA validation logic
            if self._contains_personal_data(data):
                # Check opt-out rights
                opt_out_provided = data.get('opt_out_available', False)
                if not opt_out_provided:
                    return {
                        'compliant': False,
                        'violations': ['Opt-out mechanism not provided'],
                        'recommendations': ['Implement opt-out mechanism for data sales']
                    }
            
            return {'compliant': True, 'violations': [], 'recommendations': []}
        
        except Exception as e:
            return {
                'compliant': False,
                'violations': [f'CCPA validation error: {e}'],
                'recommendations': ['Review CCPA compliance implementation']
            }
    
    def _validate_pdpa(self, data: Dict[str, Any], manager) -> Dict[str, Any]:
        """Validate PDPA compliance."""
        try:
            # PDPA validation logic
            if self._contains_personal_data(data):
                # Check notification requirements
                notification_provided = data.get('notification_provided', False)
                if not notification_provided:
                    return {
                        'compliant': False,
                        'violations': ['Data processing notification not provided'],
                        'recommendations': ['Provide notification of data processing activities']
                    }
            
            return {'compliant': True, 'violations': [], 'recommendations': []}
        
        except Exception as e:
            return {
                'compliant': False,
                'violations': [f'PDPA validation error: {e}'],
                'recommendations': ['Review PDPA compliance implementation']
            }
    
    def _contains_personal_data(self, data: Dict[str, Any]) -> bool:
        """Check if data contains personal information."""
        personal_data_indicators = [
            'user_id', 'email', 'name', 'address', 'phone',
            'ip_address', 'device_id', 'biometric_data',
            'location_data', 'behavioral_data'
        ]
        
        return any(key in data for key in personal_data_indicators)


class GlobalLoadBalancer:
    """Global load balancer for neural operator deployments."""
    
    def __init__(self, config: GlobalDeploymentConfig):
        self.config = config
        self.region_health = {}
        self.region_latency = {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize region health monitoring
        for region_config in config.regions:
            self.region_health[region_config.region] = True
            self.region_latency[region_config.region] = 0
    
    async def route_request(self, request: Dict[str, Any], user_location: Optional[str] = None) -> DeploymentRegion:
        """Route request to optimal region."""
        # Get user preferences
        preferred_region = request.get('preferred_region')
        if preferred_region and self._is_region_healthy(preferred_region):
            return DeploymentRegion(preferred_region)
        
        # Geographic routing
        if user_location:
            optimal_region = self._get_geographic_optimal_region(user_location)
            if self._is_region_healthy(optimal_region):
                return optimal_region
        
        # Performance-based routing
        return self._get_performance_optimal_region()
    
    def _is_region_healthy(self, region: Union[str, DeploymentRegion]) -> bool:
        """Check if region is healthy."""
        if isinstance(region, str):
            try:
                region = DeploymentRegion(region)
            except ValueError:
                return False
        
        return self.region_health.get(region, False)
    
    def _get_geographic_optimal_region(self, user_location: str) -> DeploymentRegion:
        """Get geographically optimal region."""
        location_mapping = {
            'us': DeploymentRegion.US_EAST,
            'usa': DeploymentRegion.US_EAST,
            'canada': DeploymentRegion.CANADA,
            'eu': DeploymentRegion.EU_WEST,
            'europe': DeploymentRegion.EU_WEST,
            'germany': DeploymentRegion.EU_CENTRAL,
            'asia': DeploymentRegion.ASIA_PACIFIC,
            'japan': DeploymentRegion.ASIA_NORTHEAST,
            'singapore': DeploymentRegion.ASIA_PACIFIC,
            'australia': DeploymentRegion.AUSTRALIA,
            'brazil': DeploymentRegion.BRAZIL,
            'india': DeploymentRegion.INDIA
        }
        
        location_lower = user_location.lower()
        return location_mapping.get(location_lower, self.config.primary_region)
    
    def _get_performance_optimal_region(self) -> DeploymentRegion:
        """Get performance-optimal region based on current metrics."""
        healthy_regions = [region for region, healthy in self.region_health.items() if healthy]
        
        if not healthy_regions:
            return self.config.primary_region
        
        # Choose region with lowest latency
        optimal_region = min(healthy_regions, key=lambda r: self.region_latency.get(r, float('inf')))
        return optimal_region
    
    async def update_region_health(self, region: DeploymentRegion, healthy: bool, latency_ms: float = 0):
        """Update region health status."""
        self.region_health[region] = healthy
        self.region_latency[region] = latency_ms
        
        if not healthy:
            self.logger.warning(f"Region {region.value} marked unhealthy")
        else:
            self.logger.info(f"Region {region.value} healthy (latency: {latency_ms}ms)")


class GlobalDeploymentOrchestrator:
    """Orchestrate global deployment of neural operators."""
    
    def __init__(self, config: GlobalDeploymentConfig):
        self.config = config
        self.compliance_orchestrator = ComplianceOrchestrator()
        self.load_balancer = GlobalLoadBalancer(config)
        self.logger = logging.getLogger(__name__)
        
        # I18n support
        self.translator = None
        if HAS_I18N:
            self.translator = Translator()
        
        # Deployment state
        self.regional_deployments = {}
        self.deployment_status = {}
    
    async def deploy_globally(self, model_artifact: str, deployment_config: Dict[str, Any]) -> Dict[str, Any]:
        """Deploy neural operator globally across all configured regions."""
        deployment_results = {
            'overall_status': 'success',
            'regional_results': {},
            'compliance_status': {},
            'failed_regions': [],
            'warnings': []
        }
        
        # Deploy to each region
        for region_config in self.config.regions:
            try:
                self.logger.info(f"Deploying to region {region_config.region.value}")
                
                # Validate compliance for region
                compliance_result = self.compliance_orchestrator.validate_data_processing(
                    deployment_config, region_config.region
                )
                
                deployment_results['compliance_status'][region_config.region.value] = compliance_result
                
                if not compliance_result['compliant']:
                    self.logger.error(f"Compliance validation failed for {region_config.region.value}")
                    deployment_results['failed_regions'].append(region_config.region.value)
                    deployment_results['overall_status'] = 'partial'
                    continue
                
                # Deploy to region
                region_result = await self._deploy_to_region(
                    model_artifact, deployment_config, region_config
                )
                
                deployment_results['regional_results'][region_config.region.value] = region_result
                
                if not region_result['success']:
                    deployment_results['failed_regions'].append(region_config.region.value)
                    deployment_results['overall_status'] = 'partial'
                
            except Exception as e:
                self.logger.error(f"Deployment failed for {region_config.region.value}: {e}")
                deployment_results['failed_regions'].append(region_config.region.value)
                deployment_results['overall_status'] = 'partial'
        
        # Update load balancer health
        await self._update_load_balancer_health()
        
        return deployment_results
    
    async def _deploy_to_region(self, model_artifact: str, deployment_config: Dict[str, Any], region_config: RegionConfig) -> Dict[str, Any]:
        """Deploy to specific region."""
        try:
            # Localize configuration for region
            localized_config = await self._localize_config(deployment_config, region_config)
            
            # Apply encryption if required
            if region_config.encryption_required:
                localized_config['encryption'] = {
                    'in_transit': True,
                    'at_rest': True,
                    'key_management': 'regional'
                }
            
            # Apply data residency constraints
            if region_config.data_residency:
                localized_config['data_residency'] = {
                    'enforce_local_storage': True,
                    'cross_border_transfer': False
                }
            
            # Simulate deployment (in production, this would call cloud provider APIs)
            await asyncio.sleep(1)  # Simulate deployment time
            
            # Update regional deployment state
            self.regional_deployments[region_config.region] = {
                'artifact': model_artifact,
                'config': localized_config,
                'deployed_at': time.time(),
                'status': 'active'
            }
            
            return {
                'success': True,
                'region': region_config.region.value,
                'deployed_at': time.time(),
                'config': localized_config
            }
        
        except Exception as e:
            return {
                'success': False,
                'region': region_config.region.value,
                'error': str(e)
            }
    
    async def _localize_config(self, config: Dict[str, Any], region_config: RegionConfig) -> Dict[str, Any]:
        """Localize configuration for specific region."""
        localized = config.copy()
        
        # Set language
        localized['language'] = region_config.primary_language
        
        # Translate user-facing messages if translator available
        if self.translator and 'messages' in config:
            localized['messages'] = {}
            for key, message in config['messages'].items():
                try:
                    translated = await self.translator.translate(
                        message, 'en', region_config.primary_language
                    )
                    localized['messages'][key] = translated
                except Exception:
                    # Fall back to original message
                    localized['messages'][key] = message
        
        # Apply regional optimizations
        localized['optimization'] = {
            'max_latency_ms': region_config.max_latency_ms,
            'local_compute_only': region_config.local_compute_only,
            'region': region_config.region.value
        }
        
        return localized
    
    async def _update_load_balancer_health(self):
        """Update load balancer with current region health."""
        for region, deployment in self.regional_deployments.items():
            healthy = deployment.get('status') == 'active'
            latency = deployment.get('latency_ms', 0)
            
            await self.load_balancer.update_region_health(region, healthy, latency)
    
    async def handle_request(self, request: Dict[str, Any]) -> Dict[str, Any]:
        """Handle incoming request with global routing."""
        try:
            # Route to optimal region
            optimal_region = await self.load_balancer.route_request(
                request, request.get('user_location')
            )
            
            # Check if region is available
            if optimal_region not in self.regional_deployments:
                # Fallback to primary region
                optimal_region = self.config.primary_region
            
            # Add routing information to response
            response = {
                'routed_to_region': optimal_region.value,
                'compliance_status': 'verified',
                'processing_location': optimal_region.value
            }
            
            # Add compliance information
            compliance_reqs = self.compliance_orchestrator._get_compliance_requirements(optimal_region)
            response['applicable_regulations'] = compliance_reqs
            
            return response
        
        except Exception as e:
            self.logger.error(f"Request handling error: {e}")
            return {
                'error': 'Request routing failed',
                'fallback_region': self.config.primary_region.value
            }
    
    def get_deployment_status(self) -> Dict[str, Any]:
        """Get comprehensive deployment status."""
        return {
            'config': asdict(self.config),
            'regional_deployments': {
                region.value: deployment for region, deployment in self.regional_deployments.items()
            },
            'region_health': {
                region.value: healthy for region, healthy in self.load_balancer.region_health.items()
            },
            'region_latency': {
                region.value: latency for region, latency in self.load_balancer.region_latency.items()
            },
            'compliance_managers': list(self.compliance_orchestrator.compliance_managers.keys())
        }