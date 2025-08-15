"""Safe evaluation utilities for neural operators.

This module provides secure alternatives to eval() and other potentially
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


def safe_mathematical_eval(expression: str, allowed_names: Dict[str, Any] = None) -> Any:
    """Safe evaluation of mathematical expressions."""
    # Allowed operators and functions
    safe_operators = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
        ast.Mod: operator.mod,
    }
    
    safe_functions = {
        'abs': abs,
        'min': min,
        'max': max,
        'round': round,
        'sum': sum,
        'len': len,
        'int': int,
        'float': float,
        'str': str,
    }
    
    if allowed_names:
        safe_functions.update(allowed_names)
    
    def eval_node(node):
        if isinstance(node, ast.Constant):
            return node.value
        elif isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.Str):
            return node.s
        elif isinstance(node, ast.Name):
            if node.id in safe_functions:
                return safe_functions[node.id]
            else:
                raise ValueError(f"Unsafe name: {node.id}")
        elif isinstance(node, ast.BinOp):
            left = eval_node(node.left)
            right = eval_node(node.right)
            op = safe_operators.get(type(node.op))
            if op:
                return op(left, right)
            else:
                raise ValueError(f"Unsafe operator: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = eval_node(node.operand)
            op = safe_operators.get(type(node.op))
            if op:
                return op(operand)
            else:
                raise ValueError(f"Unsafe unary operator: {type(node.op)}")
        else:
            raise ValueError(f"Unsafe node type: {type(node)}")
    
    try:
        tree = ast.parse(expression, mode='eval')
        return eval_node(tree.body)
    except (ValueError, SyntaxError) as e:
        raise ValueError(f"Unsafe or invalid expression: {expression}") from e


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
        return ast.literal_eval(config_string)
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
