"""I18n configuration for global deployment."""

from typing import Dict, List
from dataclasses import dataclass
import os

@dataclass
class I18nConfig:
    """Configuration for internationalization."""
    default_language: str = 'en'
    supported_languages: List[str] = None
    translation_dir: str = 'translations'
    auto_detect: bool = True
    
    def __post_init__(self):
        if self.supported_languages is None:
            self.supported_languages = [
                'en',  # English
                'es',  # Spanish  
                'fr',  # French
                'de',  # German
                'ja',  # Japanese
                'zh',  # Chinese
                'ko',  # Korean
                'pt',  # Portuguese
                'ru',  # Russian
                'ar',  # Arabic
            ]

# Global supported languages
supported_languages = {
    'en': 'English',
    'es': 'Español',
    'fr': 'Français', 
    'de': 'Deutsch',
    'ja': '日本語',
    'zh': '中文',
    'ko': '한국어',
    'pt': 'Português',
    'ru': 'Русский',
    'ar': 'العربية',
}

# Default configuration
default_config = I18nConfig()

def get_system_language() -> str:
    """Detect system language."""
    import locale
    try:
        system_locale = locale.getdefaultlocale()[0]
        if system_locale:
            lang_code = system_locale.split('_')[0].lower()
            if lang_code in supported_languages:
                return lang_code
    except Exception:
        pass
    
    # Check environment variables
    for env_var in ['LANG', 'LANGUAGE', 'LC_ALL', 'LC_MESSAGES']:
        lang = os.environ.get(env_var, '').split('.')[0].split('_')[0].lower()
        if lang in supported_languages:
            return lang
    
    return 'en'  # Default to English