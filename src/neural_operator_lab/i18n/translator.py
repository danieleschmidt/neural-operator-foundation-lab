"""Translation system for global deployment."""

import json
import os
from typing import Dict, Any, Optional
from pathlib import Path
import logging

from .config import I18nConfig, get_system_language, supported_languages

logger = logging.getLogger(__name__)


class Translator:
    """Translation manager for internationalization."""
    
    def __init__(self, config: Optional[I18nConfig] = None):
        self.config = config or I18nConfig()
        self.current_language = self.config.default_language
        self.translations: Dict[str, Dict[str, str]] = {}
        
        # Auto-detect language if enabled
        if self.config.auto_detect:
            detected_lang = get_system_language()
            if detected_lang in self.config.supported_languages:
                self.current_language = detected_lang
        
        # Load translations
        self._load_translations()
    
    def _load_translations(self):
        """Load translation files."""
        # Built-in translations
        self.translations = {
            'en': self._get_english_translations(),
            'es': self._get_spanish_translations(),
            'fr': self._get_french_translations(),
            'de': self._get_german_translations(),
            'ja': self._get_japanese_translations(),
            'zh': self._get_chinese_translations(),
        }
        
        # Load external translation files if available
        translation_dir = Path(self.config.translation_dir)
        if translation_dir.exists():
            for lang_file in translation_dir.glob('*.json'):
                lang_code = lang_file.stem
                try:
                    with open(lang_file, 'r', encoding='utf-8') as f:
                        external_translations = json.load(f)
                        if lang_code in self.translations:
                            self.translations[lang_code].update(external_translations)
                        else:
                            self.translations[lang_code] = external_translations
                except Exception as e:
                    logger.warning(f"Failed to load translations for {lang_code}: {e}")
    
    def set_language(self, language_code: str) -> bool:
        """Set current language."""
        if language_code in self.config.supported_languages:
            self.current_language = language_code
            logger.info(f"Language set to {supported_languages.get(language_code, language_code)}")
            return True
        else:
            logger.warning(f"Unsupported language: {language_code}")
            return False
    
    def get_language(self) -> str:
        """Get current language code."""
        return self.current_language
    
    def translate(self, key: str, **kwargs) -> str:
        """Translate a key to current language."""
        # Get translation for current language
        translations = self.translations.get(self.current_language, {})
        
        # Fallback to English if not found
        if key not in translations:
            translations = self.translations.get('en', {})
        
        # Get translated text
        translated = translations.get(key, key)
        
        # Format with provided arguments
        try:
            return translated.format(**kwargs)
        except KeyError as e:
            logger.warning(f"Missing format argument {e} for key {key}")
            return translated
        except Exception as e:
            logger.error(f"Error formatting translation for key {key}: {e}")
            return key
    
    def _get_english_translations(self) -> Dict[str, str]:
        """English translations (base language)."""
        return {
            # General
            'welcome': 'Welcome to Neural Operator Foundation Lab',
            'error': 'Error',
            'warning': 'Warning',
            'info': 'Information',
            'success': 'Success',
            'loading': 'Loading...',
            'processing': 'Processing...',
            'completed': 'Completed',
            'failed': 'Failed',
            
            # Training
            'training_started': 'Training started',
            'training_completed': 'Training completed successfully',
            'training_failed': 'Training failed: {error}',
            'epoch': 'Epoch {current}/{total}',
            'loss': 'Loss: {loss:.6f}',
            'accuracy': 'Accuracy: {accuracy:.2%}',
            'validation_loss': 'Validation Loss: {loss:.6f}',
            'learning_rate': 'Learning Rate: {lr:.2e}',
            
            # Models
            'model_loaded': 'Model loaded successfully',
            'model_saved': 'Model saved to {path}',
            'model_architecture': 'Model Architecture: {name}',
            'parameters': 'Parameters: {count:,}',
            'memory_usage': 'Memory Usage: {mb:.1f} MB',
            
            # Data
            'dataset_loaded': 'Dataset loaded: {samples} samples',
            'batch_processed': 'Batch {current}/{total} processed',
            'data_validation_failed': 'Data validation failed: {error}',
            
            # Performance
            'performance_optimized': 'Performance optimizations applied',
            'gpu_detected': 'GPU detected: {name}',
            'using_cpu': 'Using CPU for computation',
            'mixed_precision_enabled': 'Mixed precision training enabled',
            
            # Errors
            'invalid_input': 'Invalid input: {details}',
            'file_not_found': 'File not found: {path}',
            'permission_denied': 'Permission denied: {path}',
            'out_of_memory': 'Out of memory error',
            'cuda_error': 'CUDA error: {details}',
            
            # Research
            'benchmark_started': 'Benchmark started',
            'benchmark_completed': 'Benchmark completed',
            'experiment_running': 'Running experiment: {name}',
            'results_saved': 'Results saved to {path}',
            
            # CLI
            'cli_help': 'Neural Operator Foundation Lab CLI',
            'command_not_found': 'Command not found: {command}',
            'invalid_argument': 'Invalid argument: {argument}',
        }
    
    def _get_spanish_translations(self) -> Dict[str, str]:
        """Spanish translations."""
        return {
            'welcome': 'Bienvenido a Neural Operator Foundation Lab',
            'error': 'Error',
            'warning': 'Advertencia',
            'info': 'Información',
            'success': 'Éxito',
            'loading': 'Cargando...',
            'processing': 'Procesando...',
            'completed': 'Completado',
            'failed': 'Falló',
            
            'training_started': 'Entrenamiento iniciado',
            'training_completed': 'Entrenamiento completado exitosamente',
            'training_failed': 'Entrenamiento falló: {error}',
            'epoch': 'Época {current}/{total}',
            'loss': 'Pérdida: {loss:.6f}',
            'accuracy': 'Precisión: {accuracy:.2%}',
            'validation_loss': 'Pérdida de Validación: {loss:.6f}',
            
            'model_loaded': 'Modelo cargado exitosamente',
            'model_saved': 'Modelo guardado en {path}',
            'parameters': 'Parámetros: {count:,}',
            'memory_usage': 'Uso de Memoria: {mb:.1f} MB',
            
            'dataset_loaded': 'Dataset cargado: {samples} muestras',
            'gpu_detected': 'GPU detectada: {name}',
            'using_cpu': 'Usando CPU para cálculos',
            
            'file_not_found': 'Archivo no encontrado: {path}',
            'permission_denied': 'Permiso denegado: {path}',
            'out_of_memory': 'Error de memoria insuficiente',
        }
    
    def _get_french_translations(self) -> Dict[str, str]:
        """French translations."""
        return {
            'welcome': 'Bienvenue à Neural Operator Foundation Lab',
            'error': 'Erreur',
            'warning': 'Avertissement',
            'info': 'Information',
            'success': 'Succès',
            'loading': 'Chargement...',
            'processing': 'Traitement...',
            'completed': 'Terminé',
            'failed': 'Échoué',
            
            'training_started': 'Entraînement commencé',
            'training_completed': 'Entraînement terminé avec succès',
            'training_failed': 'Entraînement échoué: {error}',
            'epoch': 'Époque {current}/{total}',
            'loss': 'Perte: {loss:.6f}',
            'accuracy': 'Précision: {accuracy:.2%}',
            
            'model_loaded': 'Modèle chargé avec succès',
            'model_saved': 'Modèle sauvegardé dans {path}',
            'parameters': 'Paramètres: {count:,}',
            'memory_usage': 'Utilisation Mémoire: {mb:.1f} MB',
            
            'dataset_loaded': 'Dataset chargé: {samples} échantillons',
            'gpu_detected': 'GPU détectée: {name}',
            'using_cpu': 'Utilisation du CPU pour les calculs',
            
            'file_not_found': 'Fichier non trouvé: {path}',
            'permission_denied': 'Permission refusée: {path}',
        }
    
    def _get_german_translations(self) -> Dict[str, str]:
        """German translations."""
        return {
            'welcome': 'Willkommen bei Neural Operator Foundation Lab',
            'error': 'Fehler',
            'warning': 'Warnung',
            'info': 'Information',
            'success': 'Erfolg',
            'loading': 'Laden...',
            'processing': 'Verarbeitung...',
            'completed': 'Abgeschlossen',
            'failed': 'Fehlgeschlagen',
            
            'training_started': 'Training gestartet',
            'training_completed': 'Training erfolgreich abgeschlossen',
            'training_failed': 'Training fehlgeschlagen: {error}',
            'epoch': 'Epoche {current}/{total}',
            'loss': 'Verlust: {loss:.6f}',
            'accuracy': 'Genauigkeit: {accuracy:.2%}',
            
            'model_loaded': 'Modell erfolgreich geladen',
            'model_saved': 'Modell gespeichert unter {path}',
            'parameters': 'Parameter: {count:,}',
            'memory_usage': 'Speicherverbrauch: {mb:.1f} MB',
            
            'dataset_loaded': 'Datensatz geladen: {samples} Proben',
            'gpu_detected': 'GPU erkannt: {name}',
            'using_cpu': 'CPU für Berechnungen verwenden',
        }
    
    def _get_japanese_translations(self) -> Dict[str, str]:
        """Japanese translations."""
        return {
            'welcome': 'Neural Operator Foundation Labへようこそ',
            'error': 'エラー',
            'warning': '警告',
            'info': '情報',
            'success': '成功',
            'loading': '読み込み中...',
            'processing': '処理中...',
            'completed': '完了',
            'failed': '失敗',
            
            'training_started': 'トレーニング開始',
            'training_completed': 'トレーニングが正常に完了しました',
            'training_failed': 'トレーニング失敗: {error}',
            'epoch': 'エポック {current}/{total}',
            'loss': '損失: {loss:.6f}',
            'accuracy': '精度: {accuracy:.2%}',
            
            'model_loaded': 'モデルが正常に読み込まれました',
            'model_saved': 'モデルを{path}に保存しました',
            'parameters': 'パラメータ数: {count:,}',
            'memory_usage': 'メモリ使用量: {mb:.1f} MB',
            
            'dataset_loaded': 'データセット読み込み完了: {samples}サンプル',
            'gpu_detected': 'GPU検出: {name}',
            'using_cpu': 'CPU使用中',
        }
    
    def _get_chinese_translations(self) -> Dict[str, str]:
        """Chinese translations."""
        return {
            'welcome': '欢迎使用Neural Operator Foundation Lab',
            'error': '错误',
            'warning': '警告',
            'info': '信息',
            'success': '成功',
            'loading': '加载中...',
            'processing': '处理中...',
            'completed': '已完成',
            'failed': '失败',
            
            'training_started': '训练开始',
            'training_completed': '训练成功完成',
            'training_failed': '训练失败: {error}',
            'epoch': '轮数 {current}/{total}',
            'loss': '损失: {loss:.6f}',
            'accuracy': '准确率: {accuracy:.2%}',
            
            'model_loaded': '模型加载成功',
            'model_saved': '模型已保存至 {path}',
            'parameters': '参数数量: {count:,}',
            'memory_usage': '内存使用: {mb:.1f} MB',
            
            'dataset_loaded': '数据集已加载: {samples}个样本',
            'gpu_detected': '检测到GPU: {name}',
            'using_cpu': '使用CPU进行计算',
        }


# Global translator instance
_global_translator: Optional[Translator] = None


def get_translator(config: Optional[I18nConfig] = None) -> Translator:
    """Get global translator instance."""
    global _global_translator
    
    if _global_translator is None:
        _global_translator = Translator(config)
    
    return _global_translator


def translate(key: str, **kwargs) -> str:
    """Convenience function for translation."""
    translator = get_translator()
    return translator.translate(key, **kwargs)


def set_language(language_code: str) -> bool:
    """Set global language."""
    translator = get_translator()
    return translator.set_language(language_code)