"""Voice input and output functionality."""
import io
import os
from typing import Optional
from abc import ABC, abstractmethod

try:
    import speech_recognition as sr
    SPEECH_RECOGNITION_AVAILABLE = True
except ImportError:
    SPEECH_RECOGNITION_AVAILABLE = False

try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    PYTTSX3_AVAILABLE = False

try:
    from gtts import gTTS
    GTTS_AVAILABLE = True
except ImportError:
    GTTS_AVAILABLE = False


class VoiceInputProvider(ABC):
    """Base class for voice input providers."""

    @abstractmethod
    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen to microphone and convert speech to text."""
        pass


class VoiceOutputProvider(ABC):
    """Base class for voice output providers."""

    @abstractmethod
    def speak(self, text: str, save_to_file: Optional[str] = None) -> bool:
        """Convert text to speech."""
        pass


class SpeechRecognitionInput(VoiceInputProvider):
    """Voice input using SpeechRecognition library."""

    def __init__(self, language: str = "en-US"):
        if not SPEECH_RECOGNITION_AVAILABLE:
            raise ImportError("SpeechRecognition is required for voice input")

        self.recognizer = sr.Recognizer()
        self.language = language

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """
        Listen to microphone and convert speech to text.

        Args:
            timeout: Maximum time to wait for speech to start
            phrase_time_limit: Maximum duration of a phrase

        Returns:
            Recognized text or None if failed
        """
        try:
            with sr.Microphone() as source:
                # Adjust for ambient noise
                self.recognizer.adjust_for_ambient_noise(source, duration=1)

                print("Listening...")
                audio = self.recognizer.listen(
                    source,
                    timeout=timeout,
                    phrase_time_limit=phrase_time_limit
                )

                print("Processing speech...")
                # Try multiple recognition engines
                try:
                    # Google Speech Recognition (free)
                    text = self.recognizer.recognize_google(audio, language=self.language)
                    return text
                except sr.UnknownValueError:
                    print("Could not understand audio")
                    return None
                except sr.RequestError as e:
                    print(f"Could not request results from Google Speech Recognition: {e}")
                    return None

        except Exception as e:
            print(f"Error during voice input: {e}")
            return None

    def listen_from_file(self, audio_file: str) -> Optional[str]:
        """
        Recognize speech from an audio file.

        Args:
            audio_file: Path to the audio file

        Returns:
            Recognized text or None if failed
        """
        try:
            with sr.AudioFile(audio_file) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio, language=self.language)
                return text
        except Exception as e:
            print(f"Error recognizing speech from file: {e}")
            return None


class PyTTSx3Output(VoiceOutputProvider):
    """Voice output using pyttsx3 (offline TTS)."""

    def __init__(self, rate: int = 150, volume: float = 0.9, voice_id: Optional[int] = None):
        if not PYTTSX3_AVAILABLE:
            raise ImportError("pyttsx3 is required for offline voice output")

        self.engine = pyttsx3.init()
        self.engine.setProperty('rate', rate)
        self.engine.setProperty('volume', volume)

        # Set voice (0 for male, 1 for female typically)
        if voice_id is not None:
            voices = self.engine.getProperty('voices')
            if 0 <= voice_id < len(voices):
                self.engine.setProperty('voice', voices[voice_id].id)

    def speak(self, text: str, save_to_file: Optional[str] = None) -> bool:
        """
        Convert text to speech.

        Args:
            text: Text to convert to speech
            save_to_file: Optional file path to save audio

        Returns:
            True if successful, False otherwise
        """
        try:
            if save_to_file:
                self.engine.save_to_file(text, save_to_file)
                self.engine.runAndWait()
            else:
                self.engine.say(text)
                self.engine.runAndWait()
            return True
        except Exception as e:
            print(f"Error during text-to-speech: {e}")
            return False

    def set_rate(self, rate: int):
        """Set speaking rate."""
        self.engine.setProperty('rate', rate)

    def set_volume(self, volume: float):
        """Set volume (0.0 to 1.0)."""
        self.engine.setProperty('volume', volume)


class GTTSOutput(VoiceOutputProvider):
    """Voice output using Google Text-to-Speech (online)."""

    def __init__(self, language: str = "en", slow: bool = False):
        if not GTTS_AVAILABLE:
            raise ImportError("gTTS is required for Google Text-to-Speech")

        self.language = language
        self.slow = slow

    def speak(self, text: str, save_to_file: Optional[str] = None) -> bool:
        """
        Convert text to speech using Google TTS.

        Args:
            text: Text to convert to speech
            save_to_file: File path to save audio (required for gTTS)

        Returns:
            True if successful, False otherwise
        """
        try:
            tts = gTTS(text=text, lang=self.language, slow=self.slow)

            if save_to_file:
                tts.save(save_to_file)
            else:
                # Save to temporary file
                import tempfile
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as fp:
                    tts.save(fp.name)
                    return fp.name

            return True
        except Exception as e:
            print(f"Error during Google TTS: {e}")
            return False

    def get_audio_bytes(self, text: str) -> Optional[bytes]:
        """
        Get audio as bytes for streaming in Streamlit.

        Args:
            text: Text to convert to speech

        Returns:
            Audio bytes or None if failed
        """
        try:
            tts = gTTS(text=text, lang=self.language, slow=self.slow)
            fp = io.BytesIO()
            tts.write_to_fp(fp)
            fp.seek(0)
            return fp.read()
        except Exception as e:
            print(f"Error getting audio bytes: {e}")
            return None


class VoiceManager:
    """Manager for voice input and output."""

    def __init__(
        self,
        input_provider: Optional[VoiceInputProvider] = None,
        output_provider: Optional[VoiceOutputProvider] = None,
        language: str = "en"
    ):
        """
        Initialize voice manager.

        Args:
            input_provider: Voice input provider (auto-detect if None)
            output_provider: Voice output provider (auto-detect if None)
            language: Language code (e.g., 'en', 'es', 'fr')
        """
        self.language = language

        # Auto-detect input provider
        if input_provider is None:
            if SPEECH_RECOGNITION_AVAILABLE:
                try:
                    self.input_provider = SpeechRecognitionInput(language=f"{language}-US")
                except Exception as e:
                    print(f"Could not initialize voice input: {e}")
                    self.input_provider = None
            else:
                self.input_provider = None
        else:
            self.input_provider = input_provider

        # Auto-detect output provider (prefer gTTS for better quality)
        if output_provider is None:
            if GTTS_AVAILABLE:
                try:
                    self.output_provider = GTTSOutput(language=language)
                except Exception as e:
                    print(f"Could not initialize gTTS, trying pyttsx3: {e}")
                    if PYTTSX3_AVAILABLE:
                        self.output_provider = PyTTSx3Output()
                    else:
                        self.output_provider = None
            elif PYTTSX3_AVAILABLE:
                self.output_provider = PyTTSx3Output()
            else:
                self.output_provider = None
        else:
            self.output_provider = output_provider

    def listen(self, timeout: int = 5, phrase_time_limit: int = 10) -> Optional[str]:
        """Listen to microphone and return recognized text."""
        if self.input_provider is None:
            raise RuntimeError("Voice input provider not available")
        return self.input_provider.listen(timeout, phrase_time_limit)

    def speak(self, text: str, save_to_file: Optional[str] = None) -> bool:
        """Convert text to speech."""
        if self.output_provider is None:
            raise RuntimeError("Voice output provider not available")
        return self.output_provider.speak(text, save_to_file)

    def get_audio_bytes(self, text: str) -> Optional[bytes]:
        """Get audio as bytes (for streaming in Streamlit)."""
        if isinstance(self.output_provider, GTTSOutput):
            return self.output_provider.get_audio_bytes(text)
        return None

    @property
    def input_available(self) -> bool:
        """Check if voice input is available."""
        return self.input_provider is not None

    @property
    def output_available(self) -> bool:
        """Check if voice output is available."""
        return self.output_provider is not None
