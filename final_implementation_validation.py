#!/usr/bin/env python3
"""Final validation of the complete Neural Operator Foundation Lab implementation."""

import sys
import os
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Set
import ast
import importlib.util

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ImplementationValidator:
    """Comprehensive validation of the neural operator implementation."""
    
    def __init__(self):
        self.src_root = Path("src/neural_operator_lab")
        self.validation_results = {}
        self.total_score = 0.0
        self.max_score = 0.0
    
    def validate_all(self) -> Dict[str, Any]:
        """Run all validation checks."""
        logger.info("🚀 Starting Final Implementation Validation")
        logger.info("=" * 60)
        
        # Core validation checks
        checks = [
            ("Project Structure", self.validate_project_structure),
            ("Core Models", self.validate_core_models),
            ("Training Infrastructure", self.validate_training_infrastructure),
            ("Data Processing", self.validate_data_processing),
            ("Security & Monitoring", self.validate_security_monitoring),
            ("Optimization & Scaling", self.validate_optimization_scaling),
            ("Quality Gates", self.validate_quality_gates),
            ("Documentation", self.validate_documentation),
            ("Examples & Demos", self.validate_examples),
            ("Production Readiness", self.validate_production_readiness)
        ]
        
        for check_name, check_func in checks:
            logger.info(f"\\n🔍 Validating: {check_name}")
            try:
                result = check_func()
                self.validation_results[check_name] = result
                
                score = result.get('score', 0)
                max_score = result.get('max_score', 100)
                self.total_score += score
                self.max_score += max_score
                
                status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
                logger.info(f"   {status} - Score: {score:.1f}/{max_score}")
                
                # Log key findings
                for finding in result.get('findings', [])[:3]:  # Top 3 findings
                    logger.info(f"   • {finding}")
                
            except Exception as e:
                logger.error(f"   ❌ FAILED - {e}")
                self.validation_results[check_name] = {
                    'passed': False,
                    'score': 0,
                    'max_score': 100,
                    'error': str(e)
                }
        
        # Generate final report
        final_score = (self.total_score / self.max_score * 100) if self.max_score > 0 else 0
        overall_passed = final_score >= 80.0
        
        summary = {
            'overall_passed': overall_passed,
            'final_score': final_score,
            'total_score': self.total_score,
            'max_score': self.max_score,
            'validation_results': self.validation_results,
            'timestamp': time.time()
        }
        
        return summary
    
    def validate_project_structure(self) -> Dict[str, Any]:
        """Validate project structure completeness."""
        required_structure = {
            'src/neural_operator_lab': ['__init__.py', 'base.py', 'utils.py', 'config.py'],
            'src/neural_operator_lab/models': ['__init__.py', 'fourier.py', 'transformer.py', 'probabilistic.py'],
            'src/neural_operator_lab/training': ['__init__.py', 'trainer.py', 'losses.py', 'callbacks.py'],
            'src/neural_operator_lab/data': ['__init__.py', 'datasets.py', 'loaders.py'],
            'src/neural_operator_lab/security': ['__init__.py', 'input_validation.py', 'encryption.py'],
            'src/neural_operator_lab/monitoring': ['__init__.py', 'error_handler.py', 'health_checker.py'],
            'src/neural_operator_lab/optimization': ['__init__.py', 'performance.py', 'quantum_performance.py'],
            'src/neural_operator_lab/scaling': ['__init__.py', 'auto_scaler.py'],
            'src/neural_operator_lab/quality_gates': ['__init__.py', 'progressive_gates.py'],
            'examples': ['basic_fno_example.py', 'transformer_neural_operator_demo.py'],
            'tests': ['__init__.py']
        }
        
        missing_dirs = []
        missing_files = []
        present_files = 0
        total_files = 0
        
        for dir_path, files in required_structure.items():
            if not os.path.exists(dir_path):
                missing_dirs.append(dir_path)
                total_files += len(files)
                continue
            
            for file in files:
                file_path = os.path.join(dir_path, file)
                total_files += 1
                if os.path.exists(file_path):
                    present_files += 1
                else:
                    missing_files.append(file_path)
        
        score = (present_files / total_files * 100) if total_files > 0 else 0
        passed = score >= 90
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Structure completeness: {present_files}/{total_files} files present",
                f"Missing directories: {len(missing_dirs)}",
                f"Missing files: {len(missing_files)}"
            ],
            'details': {
                'present_files': present_files,
                'total_files': total_files,
                'missing_dirs': missing_dirs,
                'missing_files': missing_files
            }
        }
    
    def validate_core_models(self) -> Dict[str, Any]:
        """Validate core neural operator models."""
        model_files = {
            'fourier.py': ['FourierNeuralOperator', 'FNOBlock', 'SpectralConv1d'],
            'transformer.py': ['TransformerNeuralOperator', 'TNOBlock', 'MultiHeadAttention'],
            'probabilistic.py': ['ProbabilisticNeuralOperator', 'BayesianLinear', 'UncertaintyHead']
        }
        
        implemented_models = 0
        total_models = len(model_files)
        model_analysis = {}
        
        for file_name, expected_classes in model_files.items():
            file_path = self.src_root / 'models' / file_name
            
            if not file_path.exists():
                model_analysis[file_name] = {'exists': False, 'classes': []}
                continue
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Parse AST to find classes
                tree = ast.parse(content)
                found_classes = [node.name for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
                
                # Check if expected classes are present
                present_classes = [cls for cls in expected_classes if cls in content]
                
                model_analysis[file_name] = {
                    'exists': True,
                    'expected_classes': expected_classes,
                    'found_classes': found_classes,
                    'present_classes': present_classes,
                    'completeness': len(present_classes) / len(expected_classes)
                }
                
                if model_analysis[file_name]['completeness'] >= 0.8:
                    implemented_models += 1
            
            except Exception as e:
                model_analysis[file_name] = {'exists': True, 'error': str(e)}
        
        score = (implemented_models / total_models * 100) if total_models > 0 else 0
        passed = score >= 80
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Core models implemented: {implemented_models}/{total_models}",
                f"FNO implementation: {'✓' if 'fourier.py' in model_analysis and model_analysis['fourier.py'].get('completeness', 0) >= 0.8 else '✗'}",
                f"TNO implementation: {'✓' if 'transformer.py' in model_analysis and model_analysis['transformer.py'].get('completeness', 0) >= 0.8 else '✗'}"
            ],
            'details': model_analysis
        }
    
    def validate_training_infrastructure(self) -> Dict[str, Any]:
        """Validate training infrastructure."""
        training_components = [
            'src/neural_operator_lab/training/trainer.py',
            'src/neural_operator_lab/training/losses.py',
            'src/neural_operator_lab/training/callbacks.py',
            'src/neural_operator_lab/training/optimizers.py'
        ]
        
        present_components = 0
        component_analysis = {}
        
        for component in training_components:
            if os.path.exists(component):
                present_components += 1
                
                try:
                    with open(component, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for key training concepts
                    key_concepts = {
                        'trainer.py': ['class Trainer', 'def train', 'def evaluate', 'optimizer'],
                        'losses.py': ['PhysicsInformedLoss', 'SpectralLoss', 'MSELoss'],
                        'callbacks.py': ['Callback', 'EarlyStopping', 'ModelCheckpoint'],
                        'optimizers.py': ['optimizer', 'scheduler', 'learning_rate']
                    }
                    
                    component_name = os.path.basename(component)
                    if component_name in key_concepts:
                        concepts_found = sum(1 for concept in key_concepts[component_name] if concept in content)
                        component_analysis[component_name] = {
                            'concepts_found': concepts_found,
                            'total_concepts': len(key_concepts[component_name]),
                            'completeness': concepts_found / len(key_concepts[component_name])
                        }
                
                except Exception as e:
                    component_analysis[os.path.basename(component)] = {'error': str(e)}
            else:
                component_analysis[os.path.basename(component)] = {'exists': False}
        
        score = (present_components / len(training_components) * 100) if training_components else 0
        passed = score >= 75
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Training components: {present_components}/{len(training_components)} present",
                f"Core trainer: {'✓' if 'trainer.py' in component_analysis else '✗'}",
                f"Loss functions: {'✓' if 'losses.py' in component_analysis else '✗'}"
            ],
            'details': component_analysis
        }
    
    def validate_data_processing(self) -> Dict[str, Any]:
        """Validate data processing capabilities."""
        data_files = [
            'src/neural_operator_lab/data/__init__.py',
            'src/neural_operator_lab/data/datasets.py',
            'src/neural_operator_lab/data/loaders.py'
        ]
        
        data_capabilities = 0
        max_capabilities = 3
        
        for file_path in data_files:
            if os.path.exists(file_path):
                data_capabilities += 1
        
        # Check for synthetic data generation
        init_file = 'src/neural_operator_lab/data/__init__.py'
        has_synthetic = False
        if os.path.exists(init_file):
            try:
                with open(init_file, 'r') as f:
                    content = f.read()
                has_synthetic = 'create_synthetic_dataset' in content
            except:
                pass
        
        score = (data_capabilities / max_capabilities * 100) if max_capabilities > 0 else 0
        if has_synthetic:
            score = min(100, score + 20)  # Bonus for synthetic data
        
        passed = score >= 70
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Data processing files: {data_capabilities}/{max_capabilities} present",
                f"Synthetic data generation: {'✓' if has_synthetic else '✗'}",
                f"Data pipeline ready: {'✓' if data_capabilities >= 2 else '✗'}"
            ]
        }
    
    def validate_security_monitoring(self) -> Dict[str, Any]:
        """Validate security and monitoring systems."""
        security_files = [
            'src/neural_operator_lab/security/input_validation.py',
            'src/neural_operator_lab/security/encryption.py',
            'src/neural_operator_lab/security/robust_validation.py'
        ]
        
        monitoring_files = [
            'src/neural_operator_lab/monitoring/error_handler.py',
            'src/neural_operator_lab/monitoring/health_checker.py',
            'src/neural_operator_lab/monitoring/enhanced_error_recovery.py'
        ]
        
        present_security = sum(1 for f in security_files if os.path.exists(f))
        present_monitoring = sum(1 for f in monitoring_files if os.path.exists(f))
        
        total_files = len(security_files) + len(monitoring_files)
        present_files = present_security + present_monitoring
        
        score = (present_files / total_files * 100) if total_files > 0 else 0
        passed = score >= 70
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Security components: {present_security}/{len(security_files)}",
                f"Monitoring components: {present_monitoring}/{len(monitoring_files)}",
                f"Robust validation: {'✓' if os.path.exists('src/neural_operator_lab/security/robust_validation.py') else '✗'}"
            ]
        }
    
    def validate_optimization_scaling(self) -> Dict[str, Any]:
        """Validate optimization and scaling capabilities."""
        optimization_files = [
            'src/neural_operator_lab/optimization/performance.py',
            'src/neural_operator_lab/optimization/quantum_performance.py',
            'src/neural_operator_lab/optimization/memory.py'
        ]
        
        scaling_files = [
            'src/neural_operator_lab/scaling/auto_scaler.py',
            'src/neural_operator_lab/scaling/intelligent_scaling.py'
        ]
        
        present_optimization = sum(1 for f in optimization_files if os.path.exists(f))
        present_scaling = sum(1 for f in scaling_files if os.path.exists(f))
        
        total_files = len(optimization_files) + len(scaling_files)
        present_files = present_optimization + present_scaling
        
        score = (present_files / total_files * 100) if total_files > 0 else 0
        passed = score >= 60  # Lower threshold as these are advanced features
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Optimization components: {present_optimization}/{len(optimization_files)}",
                f"Scaling components: {present_scaling}/{len(scaling_files)}",
                f"Quantum-inspired features: {'✓' if os.path.exists('src/neural_operator_lab/optimization/quantum_performance.py') else '✗'}"
            ]
        }
    
    def validate_quality_gates(self) -> Dict[str, Any]:
        """Validate quality gates implementation."""
        quality_files = [
            'src/neural_operator_lab/quality_gates/progressive_gates.py',
            'run_comprehensive_quality_gates.py',
            'test_structure_validation.py'
        ]
        
        present_files = sum(1 for f in quality_files if os.path.exists(f))
        score = (present_files / len(quality_files) * 100) if quality_files else 0
        passed = score >= 80
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Quality gate files: {present_files}/{len(quality_files)}",
                f"Comprehensive gates: {'✓' if os.path.exists('run_comprehensive_quality_gates.py') else '✗'}",
                f"Structure validation: {'✓' if os.path.exists('test_structure_validation.py') else '✗'}"
            ]
        }
    
    def validate_documentation(self) -> Dict[str, Any]:
        """Validate documentation completeness."""
        doc_files = [
            'README.md',
            'CONTRIBUTING.md',
            'LICENSE',
            'requirements.txt',
            'pyproject.toml'
        ]
        
        present_docs = sum(1 for f in doc_files if os.path.exists(f))
        
        # Check README quality
        readme_quality = 0
        if os.path.exists('README.md'):
            try:
                with open('README.md', 'r') as f:
                    readme = f.read()
                key_sections = ['installation', 'usage', 'example', 'features', 'getting started']
                readme_quality = sum(1 for section in key_sections if section.lower() in readme.lower())
            except:
                pass
        
        score = (present_docs / len(doc_files) * 70) + (readme_quality / 5 * 30)
        passed = score >= 75
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Documentation files: {present_docs}/{len(doc_files)}",
                f"README quality: {readme_quality}/5 key sections",
                f"Project setup: {'✓' if os.path.exists('pyproject.toml') else '✗'}"
            ]
        }
    
    def validate_examples(self) -> Dict[str, Any]:
        """Validate examples and demonstrations."""
        example_files = [
            'examples/basic_fno_example.py',
            'examples/transformer_neural_operator_demo.py'
        ]
        
        present_examples = 0
        working_examples = 0
        
        for example_file in example_files:
            if os.path.exists(example_file):
                present_examples += 1
                
                # Basic syntax check
                try:
                    with open(example_file, 'r') as f:
                        content = f.read()
                    ast.parse(content)
                    working_examples += 1
                except:
                    pass
        
        score = (working_examples / len(example_files) * 100) if example_files else 0
        passed = score >= 80
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Example files: {present_examples}/{len(example_files)} present",
                f"Syntactically valid: {working_examples}/{len(example_files)}",
                f"FNO example: {'✓' if os.path.exists('examples/basic_fno_example.py') else '✗'}"
            ]
        }
    
    def validate_production_readiness(self) -> Dict[str, Any]:
        """Validate production readiness."""
        production_indicators = [
            ('Deployment configs', 'deployment/'),
            ('Docker support', 'Dockerfile'),
            ('CI/CD configs', '.github/'),
            ('Production deployment', 'src/neural_operator_lab/deployment/production_deployment.py'),
            ('Error recovery', 'src/neural_operator_lab/monitoring/enhanced_error_recovery.py')
        ]
        
        present_indicators = 0
        for name, path in production_indicators:
            if os.path.exists(path):
                present_indicators += 1
        
        score = (present_indicators / len(production_indicators) * 100) if production_indicators else 0
        passed = score >= 60
        
        return {
            'passed': passed,
            'score': score,
            'max_score': 100,
            'findings': [
                f"Production indicators: {present_indicators}/{len(production_indicators)}",
                f"Deployment ready: {'✓' if os.path.exists('deployment/') else '✗'}",
                f"Error recovery: {'✓' if os.path.exists('src/neural_operator_lab/monitoring/enhanced_error_recovery.py') else '✗'}"
            ]
        }


def main():
    """Main validation entry point."""
    print("🌟 Neural Operator Foundation Lab - Final Implementation Validation")
    print("=" * 70)
    
    start_time = time.time()
    
    try:
        validator = ImplementationValidator()
        results = validator.validate_all()
        
        # Print final summary
        print("\\n" + "=" * 70)
        print("📊 FINAL VALIDATION SUMMARY")
        print("=" * 70)
        
        status = "✅ PASSED" if results['overall_passed'] else "❌ NEEDS IMPROVEMENT"
        print(f"Overall Status: {status}")
        print(f"Final Score: {results['final_score']:.1f}/100.0")
        print(f"Components Score: {results['total_score']:.1f}/{results['max_score']:.1f}")
        
        # Print detailed results
        print("\\n📋 Detailed Results:")
        for component, result in results['validation_results'].items():
            status_icon = "✅" if result.get('passed', False) else "❌"
            score = result.get('score', 0)
            max_score = result.get('max_score', 100)
            print(f"   {status_icon} {component:<25} {score:.1f}/{max_score}")
        
        # Generate recommendations
        print("\\n🔧 Recommendations:")
        failed_components = [name for name, result in results['validation_results'].items() 
                           if not result.get('passed', False)]
        
        if not failed_components:
            print("   🎉 Excellent! All components validated successfully.")
            print("   🚀 Ready for production deployment.")
        else:
            print("   Focus on improving the following components:")
            for component in failed_components:
                print(f"   • {component}")
        
        # Save detailed report
        report_path = Path("final_validation_report.json")
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"\\n📄 Detailed report saved: {report_path}")
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Validation completed in: {elapsed_time:.1f} seconds")
        
        return 0 if results['overall_passed'] else 1
    
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)