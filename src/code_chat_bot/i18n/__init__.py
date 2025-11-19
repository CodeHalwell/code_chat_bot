"""Multi-language support and translation."""
from typing import Dict, Optional
from dataclasses import dataclass

try:
    from langdetect import detect, DetectorFactory
    LANGDETECT_AVAILABLE = True
    # Set seed for consistent results
    DetectorFactory.seed = 0
except ImportError:
    LANGDETECT_AVAILABLE = False


@dataclass
class Language:
    """Language information."""
    code: str
    name: str
    native_name: str


# Supported languages
SUPPORTED_LANGUAGES = {
    "en": Language("en", "English", "English"),
    "es": Language("es", "Spanish", "Español"),
    "fr": Language("fr", "French", "Français"),
    "de": Language("de", "German", "Deutsch"),
    "it": Language("it", "Italian", "Italiano"),
    "pt": Language("pt", "Portuguese", "Português"),
    "ru": Language("ru", "Russian", "Русский"),
    "zh": Language("zh", "Chinese", "中文"),
    "ja": Language("ja", "Japanese", "日本語"),
    "ko": Language("ko", "Korean", "한국어"),
    "ar": Language("ar", "Arabic", "العربية"),
    "hi": Language("hi", "Hindi", "हिन्दी"),
    "nl": Language("nl", "Dutch", "Nederlands"),
    "pl": Language("pl", "Polish", "Polski"),
    "tr": Language("tr", "Turkish", "Türkçe"),
}


# UI Translations
UI_TRANSLATIONS = {
    "en": {
        "app_title": "AI ChatBot",
        "chat_history": "Chat History",
        "clear_chat": "Clear Chat",
        "select_provider": "Select AI Provider",
        "select_model": "Select Model",
        "temperature": "Temperature",
        "max_tokens": "Max Tokens",
        "system_prompt": "System Prompt",
        "upload_document": "Upload Document",
        "voice_input": "Voice Input",
        "send_message": "Send Message",
        "user": "User",
        "assistant": "Assistant",
        "cost": "Cost",
        "tokens": "Tokens",
        "sentiment": "Sentiment",
        "entities": "Entities",
        "save_chat": "Save Chat",
        "load_chat": "Load Chat",
        "export": "Export",
        "settings": "Settings",
        "language": "Language",
        "theme": "Theme",
    },
    "es": {
        "app_title": "ChatBot IA",
        "chat_history": "Historial de Chat",
        "clear_chat": "Limpiar Chat",
        "select_provider": "Seleccionar Proveedor IA",
        "select_model": "Seleccionar Modelo",
        "temperature": "Temperatura",
        "max_tokens": "Tokens Máximos",
        "system_prompt": "Prompt del Sistema",
        "upload_document": "Subir Documento",
        "voice_input": "Entrada de Voz",
        "send_message": "Enviar Mensaje",
        "user": "Usuario",
        "assistant": "Asistente",
        "cost": "Costo",
        "tokens": "Tokens",
        "sentiment": "Sentimiento",
        "entities": "Entidades",
        "save_chat": "Guardar Chat",
        "load_chat": "Cargar Chat",
        "export": "Exportar",
        "settings": "Configuración",
        "language": "Idioma",
        "theme": "Tema",
    },
    "fr": {
        "app_title": "ChatBot IA",
        "chat_history": "Historique du Chat",
        "clear_chat": "Effacer le Chat",
        "select_provider": "Sélectionner le Fournisseur IA",
        "select_model": "Sélectionner le Modèle",
        "temperature": "Température",
        "max_tokens": "Tokens Maximum",
        "system_prompt": "Prompt Système",
        "upload_document": "Télécharger un Document",
        "voice_input": "Entrée Vocale",
        "send_message": "Envoyer un Message",
        "user": "Utilisateur",
        "assistant": "Assistant",
        "cost": "Coût",
        "tokens": "Tokens",
        "sentiment": "Sentiment",
        "entities": "Entités",
        "save_chat": "Sauvegarder le Chat",
        "load_chat": "Charger le Chat",
        "export": "Exporter",
        "settings": "Paramètres",
        "language": "Langue",
        "theme": "Thème",
    },
    "de": {
        "app_title": "KI-ChatBot",
        "chat_history": "Chat-Verlauf",
        "clear_chat": "Chat Löschen",
        "select_provider": "KI-Anbieter Auswählen",
        "select_model": "Modell Auswählen",
        "temperature": "Temperatur",
        "max_tokens": "Maximale Tokens",
        "system_prompt": "System-Prompt",
        "upload_document": "Dokument Hochladen",
        "voice_input": "Spracheingabe",
        "send_message": "Nachricht Senden",
        "user": "Benutzer",
        "assistant": "Assistent",
        "cost": "Kosten",
        "tokens": "Tokens",
        "sentiment": "Stimmung",
        "entities": "Entitäten",
        "save_chat": "Chat Speichern",
        "load_chat": "Chat Laden",
        "export": "Exportieren",
        "settings": "Einstellungen",
        "language": "Sprache",
        "theme": "Thema",
    },
    "zh": {
        "app_title": "AI 聊天机器人",
        "chat_history": "聊天历史",
        "clear_chat": "清除聊天",
        "select_provider": "选择AI提供商",
        "select_model": "选择模型",
        "temperature": "温度",
        "max_tokens": "最大令牌数",
        "system_prompt": "系统提示",
        "upload_document": "上传文档",
        "voice_input": "语音输入",
        "send_message": "发送消息",
        "user": "用户",
        "assistant": "助手",
        "cost": "成本",
        "tokens": "令牌",
        "sentiment": "情感",
        "entities": "实体",
        "save_chat": "保存聊天",
        "load_chat": "加载聊天",
        "export": "导出",
        "settings": "设置",
        "language": "语言",
        "theme": "主题",
    },
}


class LanguageDetector:
    """Language detection."""

    def __init__(self):
        if not LANGDETECT_AVAILABLE:
            raise ImportError("langdetect is required for language detection")

    def detect(self, text: str) -> Optional[str]:
        """
        Detect language of text.

        Args:
            text: Text to analyze

        Returns:
            Language code (e.g., 'en', 'es') or None if failed
        """
        try:
            return detect(text)
        except Exception as e:
            print(f"Error detecting language: {e}")
            return None

    def detect_with_confidence(self, text: str) -> Optional[tuple]:
        """
        Detect language with confidence score.

        Args:
            text: Text to analyze

        Returns:
            (language_code, confidence) tuple or None if failed
        """
        try:
            from langdetect import detect_langs
            results = detect_langs(text)
            if results:
                best = results[0]
                return (best.lang, best.prob)
        except Exception as e:
            print(f"Error detecting language: {e}")
        return None


class I18nManager:
    """Internationalization manager."""

    def __init__(self, default_language: str = "en"):
        """
        Initialize I18n manager.

        Args:
            default_language: Default language code
        """
        self.default_language = default_language
        self.current_language = default_language
        self.detector = None

        if LANGDETECT_AVAILABLE:
            try:
                self.detector = LanguageDetector()
            except Exception as e:
                print(f"Could not initialize language detector: {e}")

    def set_language(self, language_code: str):
        """Set current language."""
        if language_code in SUPPORTED_LANGUAGES:
            self.current_language = language_code
        else:
            print(f"Language '{language_code}' not supported, using default")

    def get_language(self) -> Language:
        """Get current language info."""
        return SUPPORTED_LANGUAGES.get(
            self.current_language,
            SUPPORTED_LANGUAGES[self.default_language]
        )

    def translate(self, key: str) -> str:
        """
        Get translation for a key in current language.

        Args:
            key: Translation key

        Returns:
            Translated string
        """
        if self.current_language in UI_TRANSLATIONS:
            return UI_TRANSLATIONS[self.current_language].get(
                key,
                UI_TRANSLATIONS[self.default_language].get(key, key)
            )
        return UI_TRANSLATIONS[self.default_language].get(key, key)

    def t(self, key: str) -> str:
        """Shorthand for translate."""
        return self.translate(key)

    def detect_language(self, text: str) -> Optional[str]:
        """Detect language of text."""
        if self.detector:
            return self.detector.detect(text)
        return None

    def get_all_translations(self) -> Dict[str, str]:
        """Get all translations for current language."""
        if self.current_language in UI_TRANSLATIONS:
            return UI_TRANSLATIONS[self.current_language]
        return UI_TRANSLATIONS[self.default_language]

    def get_supported_languages(self) -> Dict[str, Language]:
        """Get all supported languages."""
        return SUPPORTED_LANGUAGES

    def format_system_prompt(self, language_code: str) -> str:
        """
        Get a system prompt instruction for responding in a specific language.

        Args:
            language_code: Language code

        Returns:
            System prompt instruction
        """
        language = SUPPORTED_LANGUAGES.get(language_code)
        if language:
            return f"Please respond in {language.name} ({language.native_name})."
        return ""

    @property
    def detection_available(self) -> bool:
        """Check if language detection is available."""
        return self.detector is not None
