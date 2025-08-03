"""Main CLI entry point for neural operator lab."""

import argparse
import sys
import logging
from typing import List, Optional
from pathlib import Path

from .commands import TrainCommand, EvaluateCommand, BenchmarkCommand, ConfigCommand
from ..config import configure_security


def create_parser() -> argparse.ArgumentParser:
    """Create main argument parser."""
    parser = argparse.ArgumentParser(
        prog='neural-operator-lab',
        description='Neural Operator Foundation Lab - Training and benchmarking framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  neural-operator-lab train --config config.yaml
  neural-operator-lab evaluate --model checkpoint.pt --data test.h5
  neural-operator-lab benchmark --suite standard
  neural-operator-lab config --create --output config.yaml
        """
    )
    
    # Global options
    parser.add_argument('--version', action='version', version='%(prog)s 0.1.0')
    parser.add_argument(
        '--verbose', '-v', 
        action='count', 
        default=0,
        help='Increase verbosity (-v, -vv, -vvv)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Configuration file path'
    )
    parser.add_argument(
        '--log-file',
        type=str,
        help='Log file path'
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(
        dest='command',
        title='Available commands',
        description='Neural operator lab commands'
    )
    
    # Train command
    train_cmd = TrainCommand()
    train_parser = subparsers.add_parser(
        'train',
        help='Train neural operator models',
        parents=[train_cmd.create_parser()],
        add_help=False
    )
    train_parser.set_defaults(func=train_cmd.run)
    
    # Evaluate command
    eval_cmd = EvaluateCommand()
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Evaluate trained models',
        parents=[eval_cmd.create_parser()],
        add_help=False
    )
    eval_parser.set_defaults(func=eval_cmd.run)
    
    # Benchmark command
    benchmark_cmd = BenchmarkCommand()
    benchmark_parser = subparsers.add_parser(
        'benchmark',
        help='Run benchmarking suite',
        parents=[benchmark_cmd.create_parser()],
        add_help=False
    )
    benchmark_parser.set_defaults(func=benchmark_cmd.run)
    
    # Config command
    config_cmd = ConfigCommand()
    config_parser = subparsers.add_parser(
        'config',
        help='Configuration management',
        parents=[config_cmd.create_parser()],
        add_help=False
    )
    config_parser.set_defaults(func=config_cmd.run)
    
    return parser


def setup_logging(verbose: int, log_file: Optional[str] = None):
    """Setup logging configuration."""
    levels = [logging.WARNING, logging.INFO, logging.DEBUG]
    level = levels[min(verbose, len(levels) - 1)]
    
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def main(argv: Optional[List[str]] = None) -> int:
    """Main CLI entry point."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Setup logging
    setup_logging(args.verbose, args.log_file)
    logger = logging.getLogger(__name__)
    
    # Configure security
    try:
        configure_security()
    except Exception as e:
        logger.error(f"Security configuration failed: {e}")
        return 1
    
    # Handle no command
    if not hasattr(args, 'func'):
        parser.print_help()
        return 0
    
    # Load global config if provided
    if args.config:
        config_path = Path(args.config)
        if not config_path.exists():
            logger.error(f"Configuration file not found: {config_path}")
            return 1
        logger.info(f"Using configuration file: {config_path}")
    
    # Execute command
    try:
        return args.func(args)
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Command failed: {e}")
        if args.verbose >= 2:
            logger.exception("Full traceback:")
        return 1


if __name__ == '__main__':
    sys.exit(main())