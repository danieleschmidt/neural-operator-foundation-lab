#!/usr/bin/env python3
"""Progressive Quality Gates Runner Script.

This script provides a command-line interface for running the progressive
quality gates system with various configuration options.
"""

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from neural_operator_lab.quality_gates import (
        ProgressiveQualityGateSystem,
        IntelligentQualityOrchestrator,
        AutonomousQualityValidator,
        ResearchQualityValidator,
        ComprehensiveSecurityGate,
        PublicationReadinessGate
    )
except ImportError as e:
    logger.error(f"Failed to import quality gates modules: {e}")
    logger.error("Make sure the package is properly installed with: pip install -e .")
    sys.exit(1)


class QualityGatesRunner:
    """Main runner for progressive quality gates."""
    
    def __init__(self, source_dir: Path, config: Optional[Dict[str, Any]] = None):
        self.source_dir = source_dir
        self.config = config or {}
        self.results: Dict[str, Any] = {}
    
    async def run_basic_validation(self) -> Dict[str, Any]:
        """Run basic progressive validation (Generation 1-3)."""
        logger.info("🚀 Starting Basic Progressive Validation")
        
        system = ProgressiveQualityGateSystem(self.source_dir)
        results = await system.execute_progressive_validation()
        
        self.results['basic_validation'] = results
        return results
    
    async def run_autonomous_validation(self) -> Dict[str, Any]:
        """Run autonomous validation with self-improvement."""
        logger.info("🤖 Starting Autonomous Validation")
        
        validator = AutonomousQualityValidator(self.source_dir)
        results = await validator.run_autonomous_validation()
        
        self.results['autonomous_validation'] = results
        return results
    
    async def run_research_validation(self) -> Dict[str, Any]:
        """Run research-specific validation."""
        logger.info("🔬 Starting Research Quality Validation")
        
        validator = ResearchQualityValidator(self.source_dir)
        results = await validator.validate_research_quality()
        
        self.results['research_validation'] = results
        return results
    
    async def run_security_validation(self) -> Dict[str, Any]:
        """Run comprehensive security validation."""
        logger.info("🛡️ Starting Security Validation")
        
        security_gate = ComprehensiveSecurityGate()
        context = {"source_dir": self.source_dir}
        result = await security_gate.execute(context)
        
        self.results['security_validation'] = result.to_dict()
        return result.to_dict()
    
    async def run_publication_readiness(self) -> Dict[str, Any]:
        """Check publication readiness."""
        logger.info("📖 Checking Publication Readiness")
        
        pub_gate = PublicationReadinessGate()
        context = {"source_dir": self.source_dir}
        result = await pub_gate.execute(context)
        
        self.results['publication_readiness'] = result.to_dict()
        return result.to_dict()
    
    async def run_orchestrated_validation(self) -> Dict[str, Any]:
        """Run full orchestrated validation."""
        logger.info("🧠 Starting Intelligent Orchestrated Validation")
        
        orchestrator = IntelligentQualityOrchestrator(self.source_dir)
        results = await orchestrator.orchestrate_autonomous_validation()
        
        self.results['orchestrated_validation'] = results
        return results
    
    async def run_all_validations(self) -> Dict[str, Any]:
        """Run all available validations."""
        logger.info("🎯 Running All Quality Validations")
        
        start_time = time.time()
        
        # Run validations in sequence
        await self.run_basic_validation()
        await self.run_security_validation()
        
        # Optional validations based on config
        if self.config.get('enable_autonomous', True):
            await self.run_autonomous_validation()
        
        if self.config.get('enable_research', False):
            await self.run_research_validation()
            await self.run_publication_readiness()
        
        if self.config.get('enable_orchestration', True):
            await self.run_orchestrated_validation()
        
        total_time = time.time() - start_time
        
        # Generate summary
        summary = self._generate_summary(total_time)
        self.results['summary'] = summary
        
        return self.results
    
    def _generate_summary(self, total_time: float) -> Dict[str, Any]:
        """Generate a summary of all validation results."""
        summary = {
            'total_execution_time': total_time,
            'validations_run': list(self.results.keys()),
            'overall_passed': True,
            'overall_score': 0.0,
            'critical_issues': [],
            'recommendations': [],
            'production_ready': False
        }
        
        scores = []
        all_recommendations = []
        
        # Analyze basic validation
        if 'basic_validation' in self.results:
            basic = self.results['basic_validation']
            summary['overall_passed'] &= basic.get('overall_passed', False)
            
            # Extract scores from generation results
            for gen_results in basic.get('generation_results', {}).values():
                if 'average_score' in gen_results:
                    scores.append(gen_results['average_score'])
        
        # Analyze security validation
        if 'security_validation' in self.results:
            security = self.results['security_validation']
            summary['overall_passed'] &= security.get('passed', False)
            scores.append(security.get('score', 0.0))
            
            # Add security-specific issues
            if not security.get('passed', False):
                summary['critical_issues'].append("Security validation failed")
            
            all_recommendations.extend(security.get('recommendations', []))
        
        # Analyze orchestrated validation
        if 'orchestrated_validation' in self.results:
            orchestrated = self.results['orchestrated_validation']
            summary['overall_passed'] &= orchestrated.get('production_ready', False)
            scores.append(orchestrated.get('overall_quality_score', 0.0))
            summary['production_ready'] = orchestrated.get('production_ready', False)
            
            # Add orchestrated recommendations
            analysis = orchestrated.get('analysis', {})
            all_recommendations.extend(analysis.get('recommendations', []))
            
            # Add critical issues
            for issue in analysis.get('critical_issues', []):
                summary['critical_issues'].append(f"Critical issue in {issue.get('gate', 'unknown')}")
        
        # Calculate overall score
        if scores:
            summary['overall_score'] = sum(scores) / len(scores)
        
        # Deduplicate recommendations
        summary['recommendations'] = list(set(all_recommendations))
        
        return summary
    
    def print_summary(self):
        """Print a formatted summary of results."""
        if 'summary' not in self.results:
            logger.error("No summary available. Run validations first.")
            return
        
        summary = self.results['summary']
        
        print("\n" + "=" * 80)
        print("🏆 PROGRESSIVE QUALITY GATES SUMMARY")
        print("=" * 80)
        
        # Overall status
        status = "✅ PASSED" if summary['overall_passed'] else "❌ FAILED"
        print(f"Overall Status: {status}")
        print(f"Overall Score: {summary['overall_score']:.2f}/1.00")
        print(f"Production Ready: {'✅ YES' if summary['production_ready'] else '❌ NO'}")
        print(f"Total Execution Time: {summary['total_execution_time']:.2f}s")
        
        # Validations run
        print(f"\n📋 Validations Run: {len(summary['validations_run'])}")
        for validation in summary['validations_run']:
            print(f"  • {validation.replace('_', ' ').title()}")
        
        # Critical issues
        if summary['critical_issues']:
            print(f"\n🚨 Critical Issues ({len(summary['critical_issues'])}):")
            for issue in summary['critical_issues']:
                print(f"  • {issue}")
        
        # Recommendations
        if summary['recommendations']:
            print(f"\n💡 Recommendations ({len(summary['recommendations'])}):")
            for rec in summary['recommendations'][:10]:  # Show top 10
                print(f"  • {rec}")
            
            if len(summary['recommendations']) > 10:
                print(f"  ... and {len(summary['recommendations']) - 10} more")
        
        # Quality assessment
        score = summary['overall_score']
        if score >= 0.9:
            quality_level = "EXCELLENT 🌟"
        elif score >= 0.8:
            quality_level = "GOOD ✅"
        elif score >= 0.7:
            quality_level = "ACCEPTABLE ⚠️"
        else:
            quality_level = "NEEDS IMPROVEMENT ❌"
        
        print(f"\n🎯 Quality Level: {quality_level}")
        print("=" * 80)
    
    def save_results(self, output_file: Path):
        """Save results to a JSON file."""
        try:
            with open(output_file, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            logger.info(f"Results saved to: {output_file}")
        except Exception as e:
            logger.error(f"Failed to save results: {e}")


async def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Progressive Quality Gates Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all validations
  python run_progressive_quality_gates.py

  # Run specific validation
  python run_progressive_quality_gates.py --mode security

  # Run with custom source directory
  python run_progressive_quality_gates.py --source-dir /path/to/project

  # Run research validation
  python run_progressive_quality_gates.py --mode research --enable-research

  # Save results to file
  python run_progressive_quality_gates.py --output results.json
        """
    )
    
    parser.add_argument(
        '--source-dir',
        type=Path,
        default=Path.cwd().parent,  # Default to parent directory (repo root)
        help='Source directory to validate (default: repository root)'
    )
    
    parser.add_argument(
        '--mode',
        choices=['all', 'basic', 'autonomous', 'security', 'research', 'publication', 'orchestrated'],
        default='all',
        help='Validation mode to run (default: all)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        help='Output file for results (JSON format)'
    )
    
    parser.add_argument(
        '--enable-research',
        action='store_true',
        help='Enable research-specific validations'
    )
    
    parser.add_argument(
        '--enable-autonomous',
        action='store_true',
        default=True,
        help='Enable autonomous validation features (default: true)'
    )
    
    parser.add_argument(
        '--enable-orchestration',
        action='store_true',
        default=True,
        help='Enable intelligent orchestration (default: true)'
    )
    
    parser.add_argument(
        '--config',
        type=Path,
        help='Configuration file (JSON format)'
    )
    
    parser.add_argument(
        '--ci-mode',
        action='store_true',
        help='Run in CI mode (exit with non-zero code on failure)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate source directory
    if not args.source_dir.exists():
        logger.error(f"Source directory does not exist: {args.source_dir}")
        sys.exit(1)
    
    # Load configuration
    config = {}
    if args.config and args.config.exists():
        try:
            with open(args.config) as f:
                config = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load config file: {e}")
            sys.exit(1)
    
    # Override config with command line arguments
    config['enable_research'] = args.enable_research
    config['enable_autonomous'] = args.enable_autonomous
    config['enable_orchestration'] = args.enable_orchestration
    
    # Initialize runner
    runner = QualityGatesRunner(args.source_dir, config)
    
    try:
        logger.info(f"🔍 Analyzing project: {args.source_dir.absolute()}")
        
        # Run specified validation mode
        if args.mode == 'all':
            await runner.run_all_validations()
        elif args.mode == 'basic':
            await runner.run_basic_validation()
        elif args.mode == 'autonomous':
            await runner.run_autonomous_validation()
        elif args.mode == 'security':
            await runner.run_security_validation()
        elif args.mode == 'research':
            await runner.run_research_validation()
        elif args.mode == 'publication':
            await runner.run_publication_readiness()
        elif args.mode == 'orchestrated':
            await runner.run_orchestrated_validation()
        
        # Print summary
        runner.print_summary()
        
        # Save results if requested
        if args.output:
            runner.save_results(args.output)
        
        # Exit with appropriate code for CI mode
        if args.ci_mode:
            if 'summary' in runner.results:
                if runner.results['summary']['overall_passed']:
                    logger.info("✅ CI Mode: All quality gates passed")
                    sys.exit(0)
                else:
                    logger.error("❌ CI Mode: Quality gates failed")
                    sys.exit(1)
            else:
                logger.error("❌ CI Mode: No summary available")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("❌ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Execution failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())