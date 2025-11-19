"""Logging and monitoring for the chatbot application."""
import os
import time
from typing import Dict, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
from collections import defaultdict

try:
    from loguru import logger
    LOGURU_AVAILABLE = True
except ImportError:
    LOGURU_AVAILABLE = False
    import logging
    logger = logging.getLogger(__name__)

try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, REGISTRY
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


@dataclass
class ChatMetrics:
    """Metrics for chat interactions."""
    total_messages: int = 0
    total_tokens: int = 0
    total_cost: float = 0.0
    avg_response_time: float = 0.0
    provider_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    model_usage: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    error_count: int = 0
    session_count: int = 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary."""
        return {
            "total_messages": self.total_messages,
            "total_tokens": self.total_tokens,
            "total_cost": self.total_cost,
            "avg_response_time": self.avg_response_time,
            "provider_usage": dict(self.provider_usage),
            "model_usage": dict(self.model_usage),
            "error_count": self.error_count,
            "session_count": self.session_count,
        }


class LogManager:
    """Centralized logging manager."""

    def __init__(self, log_file: str = "chatbot.log", log_level: str = "INFO"):
        """
        Initialize log manager.

        Args:
            log_file: Path to log file
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        """
        self.log_file = log_file

        if LOGURU_AVAILABLE:
            # Configure loguru
            logger.remove()  # Remove default handler
            logger.add(
                log_file,
                rotation="10 MB",
                retention="7 days",
                level=log_level,
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}"
            )
            logger.add(
                lambda msg: print(msg, end=""),
                level=log_level,
                format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>"
            )
        else:
            # Configure standard logging
            logging.basicConfig(
                level=getattr(logging, log_level),
                format='%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d - %(message)s',
                handlers=[
                    logging.FileHandler(log_file),
                    logging.StreamHandler()
                ]
            )

    def info(self, message: str, **kwargs):
        """Log info message."""
        logger.info(message, **kwargs)

    def debug(self, message: str, **kwargs):
        """Log debug message."""
        logger.debug(message, **kwargs)

    def warning(self, message: str, **kwargs):
        """Log warning message."""
        logger.warning(message, **kwargs)

    def error(self, message: str, **kwargs):
        """Log error message."""
        logger.error(message, **kwargs)

    def critical(self, message: str, **kwargs):
        """Log critical message."""
        logger.critical(message, **kwargs)

    def log_chat_message(self, role: str, provider: str, model: str, tokens: int, cost: float):
        """Log a chat message with metadata."""
        logger.info(
            f"Chat message | Role: {role} | Provider: {provider} | Model: {model} | "
            f"Tokens: {tokens} | Cost: ${cost:.6f}"
        )

    def log_error_with_context(self, error: Exception, context: Dict[str, Any]):
        """Log error with additional context."""
        logger.error(f"Error: {str(error)} | Context: {context}")


class PrometheusMetrics:
    """Prometheus metrics collector."""

    def __init__(self):
        if not PROMETHEUS_AVAILABLE:
            raise ImportError("prometheus_client is required for Prometheus metrics")

        # Define metrics
        self.message_counter = Counter(
            'chatbot_messages_total',
            'Total number of chat messages',
            ['provider', 'model', 'role']
        )

        self.token_counter = Counter(
            'chatbot_tokens_total',
            'Total number of tokens used',
            ['provider', 'model', 'type']
        )

        self.cost_counter = Counter(
            'chatbot_cost_total',
            'Total cost in USD',
            ['provider', 'model']
        )

        self.response_time = Histogram(
            'chatbot_response_time_seconds',
            'Response time in seconds',
            ['provider', 'model']
        )

        self.active_sessions = Gauge(
            'chatbot_active_sessions',
            'Number of active chat sessions'
        )

        self.error_counter = Counter(
            'chatbot_errors_total',
            'Total number of errors',
            ['error_type']
        )

    def record_message(self, provider: str, model: str, role: str):
        """Record a chat message."""
        self.message_counter.labels(provider=provider, model=model, role=role).inc()

    def record_tokens(self, provider: str, model: str, input_tokens: int, output_tokens: int):
        """Record token usage."""
        self.token_counter.labels(provider=provider, model=model, type='input').inc(input_tokens)
        self.token_counter.labels(provider=provider, model=model, type='output').inc(output_tokens)

    def record_cost(self, provider: str, model: str, cost: float):
        """Record cost."""
        self.cost_counter.labels(provider=provider, model=model).inc(cost)

    def record_response_time(self, provider: str, model: str, duration: float):
        """Record response time."""
        self.response_time.labels(provider=provider, model=model).observe(duration)

    def set_active_sessions(self, count: int):
        """Set number of active sessions."""
        self.active_sessions.set(count)

    def record_error(self, error_type: str):
        """Record an error."""
        self.error_counter.labels(error_type=error_type).inc()

    def get_metrics(self) -> bytes:
        """Get metrics in Prometheus format."""
        return generate_latest(REGISTRY)


class MetricsCollector:
    """Collects and manages application metrics."""

    def __init__(self, enable_prometheus: bool = False):
        """
        Initialize metrics collector.

        Args:
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.metrics = ChatMetrics()
        self.prometheus = None
        self.response_times = []

        if enable_prometheus and PROMETHEUS_AVAILABLE:
            try:
                self.prometheus = PrometheusMetrics()
            except Exception as e:
                print(f"Could not initialize Prometheus metrics: {e}")

    def record_message(
        self,
        provider: str,
        model: str,
        role: str,
        tokens: int,
        cost: float,
        response_time: Optional[float] = None
    ):
        """
        Record a chat message with all metrics.

        Args:
            provider: AI provider name
            model: Model name
            role: Message role (user/assistant)
            tokens: Number of tokens
            cost: Cost in USD
            response_time: Response time in seconds
        """
        self.metrics.total_messages += 1
        self.metrics.total_tokens += tokens
        self.metrics.total_cost += cost
        self.metrics.provider_usage[provider] += 1
        self.metrics.model_usage[model] += 1

        if response_time is not None:
            self.response_times.append(response_time)
            if self.response_times:
                self.metrics.avg_response_time = sum(self.response_times) / len(self.response_times)

        # Record in Prometheus if available
        if self.prometheus:
            self.prometheus.record_message(provider, model, role)
            self.prometheus.record_tokens(provider, model, tokens, 0)  # Simplified
            self.prometheus.record_cost(provider, model, cost)
            if response_time:
                self.prometheus.record_response_time(provider, model, response_time)

    def record_error(self, error_type: str = "unknown"):
        """Record an error."""
        self.metrics.error_count += 1
        if self.prometheus:
            self.prometheus.record_error(error_type)

    def increment_session_count(self):
        """Increment session counter."""
        self.metrics.session_count += 1
        if self.prometheus:
            self.prometheus.set_active_sessions(self.metrics.session_count)

    def get_metrics(self) -> ChatMetrics:
        """Get current metrics."""
        return self.metrics

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics.to_dict()

    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics = ChatMetrics()
        self.response_times = []


class PerformanceTimer:
    """Context manager for timing operations."""

    def __init__(self, name: str, log_manager: Optional[LogManager] = None):
        """
        Initialize performance timer.

        Args:
            name: Name of the operation
            log_manager: Optional log manager to log results
        """
        self.name = name
        self.log_manager = log_manager
        self.start_time = None
        self.duration = None

    def __enter__(self):
        """Start timer."""
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop timer and log result."""
        self.duration = time.time() - self.start_time
        if self.log_manager:
            self.log_manager.debug(f"{self.name} completed in {self.duration:.3f}s")


class MonitoringManager:
    """Centralized monitoring manager combining logging and metrics."""

    def __init__(
        self,
        log_file: str = "chatbot.log",
        log_level: str = "INFO",
        enable_prometheus: bool = False
    ):
        """
        Initialize monitoring manager.

        Args:
            log_file: Path to log file
            log_level: Logging level
            enable_prometheus: Whether to enable Prometheus metrics
        """
        self.logger = LogManager(log_file, log_level)
        self.metrics = MetricsCollector(enable_prometheus)

    def log_and_record_message(
        self,
        provider: str,
        model: str,
        role: str,
        tokens: int,
        cost: float,
        response_time: Optional[float] = None
    ):
        """Log and record a chat message."""
        self.logger.log_chat_message(role, provider, model, tokens, cost)
        self.metrics.record_message(provider, model, role, tokens, cost, response_time)

    def log_and_record_error(self, error: Exception, error_type: str = "unknown", context: Optional[Dict] = None):
        """Log and record an error."""
        self.logger.log_error_with_context(error, context or {})
        self.metrics.record_error(error_type)

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        return self.metrics.get_summary()

    def timer(self, name: str) -> PerformanceTimer:
        """Create a performance timer."""
        return PerformanceTimer(name, self.logger)
