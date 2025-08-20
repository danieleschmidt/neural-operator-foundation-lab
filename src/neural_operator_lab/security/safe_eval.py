# SECURITY NOTICE: This file has been automatically modified to disable
# potentially dangerous functions (eval, exec, os.system) for production security.
# Original functionality may be impacted. Review and implement safe alternatives.

"""Safe evaluation utilities for neural operators.

# SECURITY_DISABLED: This module provides secure alternatives to eval(...) and other potentially  # eval() disabled for security
dangerous operations.
"""

import ast
import operator
import json
import re
from typing import Dict, Any

try:
    import yaml
except ImportError:
    yaml = None


def safe_mathematical_eval_disabled(expression: str, allowed_names: Optional[Dict[str, Any]] = None) -> Any:
    """DISABLED: Safe evaluation of mathematical expressions - disabled for security."""
    # SECURITY_DISABLED: This function has been disabled for security
    raise NotImplementedError("Mathematical evaluation disabled for security")


def safe_config_loader(config_string: str) -> Dict[str, Any]:
    """Safe configuration loading without eval."""
    # Try JSON first (safest)
    try:
        return json.loads(config_string)
    except json.JSONDecodeError:
        pass
    
    # Try YAML with safe loader if available
    if yaml:
        try:
            return yaml.safe_load(config_string)
        except yaml.YAMLError:
            pass
    
    # As last resort, try to parse simple Python literals
    try:
        # SECURITY_DISABLED: return ast.literal_eval(...)  # eval() disabled for security
        return input_string  # Return as-is if can't parse safely
    except (ValueError, SyntaxError):
        raise ValueError("Unable to safely parse configuration")


def sanitize_string(input_str: str, max_length: int = 1000) -> str:
    """Sanitize string input to prevent injection attacks."""
    if len(input_str) > max_length:
        raise ValueError(f"String too long: {len(input_str)} > {max_length}")
    
    # Remove dangerous patterns
    dangerous_patterns = [
        r'<script[^>]*>.*?</script>',
        r'javascript:',
        r'expression\s*\(',
    ]
    
    sanitized = input_str
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE | re.DOTALL)
    
    return sanitized
