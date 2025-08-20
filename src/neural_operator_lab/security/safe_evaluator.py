# SECURITY NOTICE: This file provides safe alternatives to eval() and exec()
# for production use in the neural operator framework.

"""Safe alternatives to dangerous evaluation functions."""

import ast
import json
import operator
from typing import Any, Dict, Union, Callable

# Safe operators mapping
SAFE_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.Mod: operator.mod,
    ast.FloorDiv: operator.floordiv,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

class SafeEvaluator:
    """Safe evaluator that avoids eval() and exec()."""
    
    @staticmethod
    def safe_literal_eval(expression: str) -> Any:
        """Safely evaluate literal expressions."""
        try:
            return ast.literal_eval(expression)
        except (ValueError, SyntaxError) as e:
            raise ValueError(f"Invalid literal expression: {expression}") from e
    
    @staticmethod
    def safe_json_eval(expression: str) -> Any:
        """Safely evaluate JSON expressions."""
        try:
            return json.loads(expression)
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON expression: {expression}") from e
    
    @staticmethod
    def safe_math_eval(expression: str) -> float:
        """Safely evaluate simple mathematical expressions."""
        try:
            # Parse the expression
            node = ast.parse(expression, mode='eval')
            return SafeEvaluator._eval_node(node.body)
        except Exception as e:
            raise ValueError(f"Invalid mathematical expression: {expression}") from e
    
    @staticmethod
    def _eval_node(node):
        """Recursively evaluate AST nodes safely."""
        if isinstance(node, ast.Constant):  # Numbers, strings, etc.
            return node.value
        elif isinstance(node, ast.BinOp):  # Binary operations
            left = SafeEvaluator._eval_node(node.left)
            right = SafeEvaluator._eval_node(node.right)
            op = SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported operation: {type(node.op)}")
            return op(left, right)
        elif isinstance(node, ast.UnaryOp):  # Unary operations
            operand = SafeEvaluator._eval_node(node.operand)
            op = SAFE_OPERATORS.get(type(node.op))
            if op is None:
                raise ValueError(f"Unsupported unary operation: {type(node.op)}")
            return op(operand)
        else:
            raise ValueError(f"Unsupported node type: {type(node)}")

# Legacy function replacements (return safe defaults)
def eval_replacement(*args, **kwargs):
    """Replacement for eval() that returns a safe default."""
    import warnings
    warnings.warn("eval() has been disabled for security. Use SafeEvaluator instead.", 
                  SecurityWarning, stacklevel=2)
    return None

def exec_replacement(*args, **kwargs):
    """Replacement for exec() that does nothing safely."""
    import warnings
    warnings.warn("exec() has been disabled for security.", 
                  SecurityWarning, stacklevel=2)
    return None

# Monkey patch dangerous functions (optional, use with caution)
# eval = eval_replacement
# exec = exec_replacement
