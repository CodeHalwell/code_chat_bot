"""NLP analysis tools for sentiment analysis and entity recognition."""
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
except ImportError:
    TEXTBLOB_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
except ImportError:
    SPACY_AVAILABLE = False


@dataclass
class SentimentResult:
    """Sentiment analysis result."""
    polarity: float  # -1 (negative) to 1 (positive)
    subjectivity: float  # 0 (objective) to 1 (subjective)
    label: str  # "positive", "negative", or "neutral"

    def __str__(self):
        return f"{self.label.capitalize()} (polarity: {self.polarity:.2f}, subjectivity: {self.subjectivity:.2f})"


@dataclass
class Entity:
    """Named entity."""
    text: str
    label: str
    start: int
    end: int

    def __str__(self):
        return f"{self.text} ({self.label})"


@dataclass
class NLPAnalysisResult:
    """Complete NLP analysis result."""
    sentiment: SentimentResult
    entities: List[Entity]
    keywords: List[Tuple[str, float]]  # (keyword, relevance_score)
    language: str


class SentimentAnalyzer:
    """Sentiment analysis using TextBlob."""

    def __init__(self):
        if not TEXTBLOB_AVAILABLE:
            raise ImportError("textblob is required for sentiment analysis")

    def analyze(self, text: str) -> SentimentResult:
        """
        Analyze sentiment of text.

        Args:
            text: Text to analyze

        Returns:
            SentimentResult with polarity and subjectivity
        """
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity

        # Classify sentiment
        if polarity > 0.1:
            label = "positive"
        elif polarity < -0.1:
            label = "negative"
        else:
            label = "neutral"

        return SentimentResult(
            polarity=polarity,
            subjectivity=subjectivity,
            label=label
        )

    def analyze_by_sentence(self, text: str) -> List[Tuple[str, SentimentResult]]:
        """
        Analyze sentiment of each sentence.

        Args:
            text: Text to analyze

        Returns:
            List of (sentence, SentimentResult) tuples
        """
        blob = TextBlob(text)
        results = []

        for sentence in blob.sentences:
            sentiment = self.analyze(str(sentence))
            results.append((str(sentence), sentiment))

        return results


class EntityRecognizer:
    """Named entity recognition using spaCy."""

    def __init__(self, model_name: str = "en_core_web_sm"):
        if not SPACY_AVAILABLE:
            raise ImportError("spacy is required for entity recognition")

        try:
            self.nlp = spacy.load(model_name)
        except OSError:
            print(f"spaCy model '{model_name}' not found. Please install it with:")
            print(f"python -m spacy download {model_name}")
            raise

    def recognize(self, text: str) -> List[Entity]:
        """
        Recognize named entities in text.

        Args:
            text: Text to analyze

        Returns:
            List of Entity objects
        """
        doc = self.nlp(text)
        entities = []

        for ent in doc.ents:
            entities.append(Entity(
                text=ent.text,
                label=ent.label_,
                start=ent.start_char,
                end=ent.end_char
            ))

        return entities

    def extract_keywords(self, text: str, top_n: int = 10) -> List[Tuple[str, float]]:
        """
        Extract keywords using noun chunks and named entities.

        Args:
            text: Text to analyze
            top_n: Number of top keywords to return

        Returns:
            List of (keyword, relevance_score) tuples
        """
        doc = self.nlp(text)
        keywords = {}

        # Extract noun chunks
        for chunk in doc.noun_chunks:
            # Filter out stop words and very short chunks
            if not chunk.root.is_stop and len(chunk.text) > 2:
                keywords[chunk.text.lower()] = keywords.get(chunk.text.lower(), 0) + 1

        # Add named entities with higher weight
        for ent in doc.ents:
            keywords[ent.text.lower()] = keywords.get(ent.text.lower(), 0) + 2

        # Sort by frequency and return top N
        sorted_keywords = sorted(keywords.items(), key=lambda x: x[1], reverse=True)
        return sorted_keywords[:top_n]

    def get_pos_tags(self, text: str) -> List[Tuple[str, str]]:
        """
        Get part-of-speech tags for text.

        Args:
            text: Text to analyze

        Returns:
            List of (word, pos_tag) tuples
        """
        doc = self.nlp(text)
        return [(token.text, token.pos_) for token in doc]


class NLPAnalyzer:
    """Complete NLP analyzer combining sentiment and entity recognition."""

    def __init__(self, spacy_model: str = "en_core_web_sm"):
        """
        Initialize NLP analyzer.

        Args:
            spacy_model: spaCy model to use
        """
        self.sentiment_analyzer = None
        self.entity_recognizer = None

        # Initialize sentiment analyzer if available
        if TEXTBLOB_AVAILABLE:
            try:
                self.sentiment_analyzer = SentimentAnalyzer()
            except Exception as e:
                print(f"Could not initialize sentiment analyzer: {e}")

        # Initialize entity recognizer if available
        if SPACY_AVAILABLE:
            try:
                self.entity_recognizer = EntityRecognizer(spacy_model)
            except Exception as e:
                print(f"Could not initialize entity recognizer: {e}")

    def analyze(self, text: str, include_keywords: bool = True) -> NLPAnalysisResult:
        """
        Perform complete NLP analysis on text.

        Args:
            text: Text to analyze
            include_keywords: Whether to extract keywords

        Returns:
            NLPAnalysisResult with sentiment, entities, and keywords
        """
        # Sentiment analysis
        sentiment = None
        if self.sentiment_analyzer:
            sentiment = self.sentiment_analyzer.analyze(text)
        else:
            sentiment = SentimentResult(polarity=0.0, subjectivity=0.5, label="neutral")

        # Entity recognition
        entities = []
        keywords = []
        language = "unknown"

        if self.entity_recognizer:
            entities = self.entity_recognizer.recognize(text)
            if include_keywords:
                keywords = self.entity_recognizer.extract_keywords(text)

            # Detect language
            doc = self.entity_recognizer.nlp(text[:100])  # Use first 100 chars
            language = doc.lang_

        return NLPAnalysisResult(
            sentiment=sentiment,
            entities=entities,
            keywords=keywords,
            language=language
        )

    def get_sentiment_summary(self, messages: List[str]) -> Dict[str, any]:
        """
        Get sentiment summary for a list of messages.

        Args:
            messages: List of message texts

        Returns:
            Dictionary with sentiment statistics
        """
        if not self.sentiment_analyzer:
            return {"error": "Sentiment analyzer not available"}

        sentiments = [self.sentiment_analyzer.analyze(msg) for msg in messages]

        positive = sum(1 for s in sentiments if s.label == "positive")
        negative = sum(1 for s in sentiments if s.label == "negative")
        neutral = sum(1 for s in sentiments if s.label == "neutral")

        avg_polarity = sum(s.polarity for s in sentiments) / len(sentiments) if sentiments else 0
        avg_subjectivity = sum(s.subjectivity for s in sentiments) / len(sentiments) if sentiments else 0

        return {
            "total_messages": len(messages),
            "positive": positive,
            "negative": negative,
            "neutral": neutral,
            "avg_polarity": avg_polarity,
            "avg_subjectivity": avg_subjectivity,
            "overall_sentiment": "positive" if avg_polarity > 0.1 else "negative" if avg_polarity < -0.1 else "neutral"
        }

    def get_entity_summary(self, text: str) -> Dict[str, List[str]]:
        """
        Get summary of entities grouped by type.

        Args:
            text: Text to analyze

        Returns:
            Dictionary mapping entity types to lists of entity texts
        """
        if not self.entity_recognizer:
            return {"error": "Entity recognizer not available"}

        entities = self.entity_recognizer.recognize(text)
        entity_dict = {}

        for entity in entities:
            if entity.label not in entity_dict:
                entity_dict[entity.label] = []
            if entity.text not in entity_dict[entity.label]:
                entity_dict[entity.label].append(entity.text)

        return entity_dict

    @property
    def sentiment_available(self) -> bool:
        """Check if sentiment analysis is available."""
        return self.sentiment_analyzer is not None

    @property
    def entity_recognition_available(self) -> bool:
        """Check if entity recognition is available."""
        return self.entity_recognizer is not None
