#!/usr/bin/env python3
"""Standalone config test that doesn't require torch."""

import sys
from pathlib import Path

# Add src to path  
sys.path.insert(0, str(Path(__file__).parent / 'src'))

def test_config():
    """Test config module directly."""
    
    # Direct import to avoid __init__.py torch dependencies
    from neural_operator_lab.config import (
        Config, SecurityConfig, configure_security, get_security_config,
        ExperimentConfig, TrainingConfig, ModelConfig, DataConfig
    )
    
    print("✅ Config imports successful")
    
    # Test basic config creation
    config = Config()
    assert config.experiment.name == "neural_operator_experiment"
    print("✅ Config creation")
    
    # Test config validation
    errors = config.validate()
    print(f"✅ Config validation (found {len(errors)} expected errors)")
    
    # Test security config
    security_config = SecurityConfig()
    assert security_config.validate_models == True
    print("✅ Security config")
    
    # Test security update
    configure_security(max_model_size="500MB", secure_mode=True)
    updated_config = get_security_config()
    assert updated_config.max_model_size == "500MB"
    print("✅ Security config update")
    
    # Test config serialization
    config_dict = config.to_dict()
    assert 'experiment' in config_dict
    assert 'training' in config_dict
    print("✅ Config serialization")
    
    # Test config from dict
    new_config = Config.from_dict(config_dict)
    assert new_config.experiment.name == config.experiment.name
    print("✅ Config deserialization")
    
    print("\n🎉 All config tests passed!")
    return True

if __name__ == '__main__':
    try:
        test_config()
        print("\n✨ Config module fully functional without external dependencies!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Config test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)