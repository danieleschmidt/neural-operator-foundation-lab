"""Internationalization support for Neural Operator Foundation Lab."""

from .translator import Translator, get_translator
from .config import I18nConfig, supported_languages

__all__ = ['Translator', 'get_translator', 'I18nConfig', 'supported_languages']