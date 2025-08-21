"""Secure configuration parsing without eval()."""

import ast
import json
import yaml
from typing import Any, Dict, Union

class SecureConfigParser:
    """Secure configuration parser that avoids eval()."""
    
    @staticmethod
    def parse_value(value: str) -> Any:
        """Parse configuration value securely."""
        # Try JSON parsing first
        try:
            return json.loads(value)
        except json.JSONDecodeError:
            pass
            
        # Try literal eval for basic Python literals
        try:
            return ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
            
        # Return as string if nothing else works
        return value
    
    @staticmethod
    def parse_dict(config_dict: Dict[str, str]) -> Dict[str, Any]:
        """Parse dictionary of configuration values."""
        return {key: SecureConfigParser.parse_value(str(val)) 
                for key, val in config_dict.items()}
