"""Configuration validation for application settings."""

import logging
from typing import Dict, List, Any, Optional, Set, Type, Union
from dataclasses import dataclass
from enum import Enum
import re
import os

from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from .config import Config

logger = logging.getLogger(__name__)


class ValidationLevel(Enum):
    """Validation level for configuration settings."""

    ERROR = "error"  # Required setting, fail if invalid
    WARNING = "warning"  # Optional setting, warn if invalid
    INFO = "info"  # Informational validation only


@dataclass
class ValidationRule:
    """Validation rule for configuration setting."""

    name: str  # Setting name
    level: ValidationLevel  # Validation level
    type: Type  # Expected type
    required: bool = True  # Whether setting is required
    default: Any = None  # Default value if not set
    min_value: Optional[Union[int, float]] = None  # Minimum value
    max_value: Optional[Union[int, float]] = None  # Maximum value
    min_length: Optional[int] = None  # Minimum length
    max_length: Optional[int] = None  # Maximum length
    pattern: Optional[str] = None  # Regex pattern
    allowed_values: Optional[List[Any]] = None  # List of allowed values
    dependencies: Optional[List[str]] = None  # Required dependencies
    conflicts: Optional[List[str]] = None  # Conflicting settings
    custom_validator: Optional[callable] = None  # Custom validation function


@dataclass
class ValidationResult:
    """Result of configuration validation."""

    name: str  # Setting name
    level: ValidationLevel  # Validation level
    valid: bool  # Whether setting is valid
    value: Any  # Current value
    message: str  # Validation message
    rule: ValidationRule  # Applied rule


class ConfigValidator:
    """Configuration validator implementation."""

    def __init__(
        self,
        config: Config,
        metrics_manager: Optional[MetricsManager] = None,
    ) -> None:
        """Initialize config validator.

        Args:
            config: Application configuration
            metrics_manager: Optional metrics manager
        """
        self.config = config
        self.metrics = metrics_manager
        self._rules: Dict[str, ValidationRule] = {}
        self._results: List[ValidationResult] = []
        self._initialized = False

        # Register default rules
        self._register_default_rules()

    def _register_default_rules(self) -> None:
        """Register default validation rules."""
        # API settings
        self.add_rule(
            ValidationRule(
                name="API_DIR",
                level=ValidationLevel.ERROR,
                type=str,
                required=True,
                custom_validator=self._validate_directory,
            )
        )

        # Cache settings
        self.add_rule(
            ValidationRule(
                name="CACHE_TTL",
                level=ValidationLevel.WARNING,
                type=int,
                required=False,
                default=3600,
                min_value=0,
            )
        )

        # Model settings
        self.add_rule(
            ValidationRule(
                name="MODEL_NAME",
                level=ValidationLevel.ERROR,
                type=str,
                required=True,
                allowed_values=[
                    "all-MiniLM-L6-v2",
                    "all-mpnet-base-v2",
                    "multi-qa-mpnet-base-dot-v1",
                ],
            )
        )

        # Pinecone settings
        self.add_rule(
            ValidationRule(
                name="PINECONE_API_KEY",
                level=ValidationLevel.ERROR,
                type=str,
                required=True,
                min_length=32,
            )
        )
        self.add_rule(
            ValidationRule(
                name="PINECONE_ENV",
                level=ValidationLevel.ERROR,
                type=str,
                required=True,
            )
        )

        # ZeroMQ settings
        self.add_rule(
            ValidationRule(
                name="ZMQ_PUB_PORT",
                level=ValidationLevel.ERROR,
                type=int,
                required=True,
                min_value=1024,
                max_value=65535,
            )
        )
        self.add_rule(
            ValidationRule(
                name="ZMQ_SUB_PORT",
                level=ValidationLevel.ERROR,
                type=int,
                required=True,
                min_value=1024,
                max_value=65535,
                conflicts=["ZMQ_PUB_PORT"],
            )
        )

        # Service discovery settings
        self.add_rule(
            ValidationRule(
                name="DISCOVERY_HOST",
                level=ValidationLevel.WARNING,
                type=str,
                required=False,
                default="localhost",
            )
        )
        self.add_rule(
            ValidationRule(
                name="DISCOVERY_PORT",
                level=ValidationLevel.WARNING,
                type=int,
                required=False,
                default=8500,
                min_value=1024,
                max_value=65535,
            )
        )

        # Circuit breaker settings
        self.add_rule(
            ValidationRule(
                name="CIRCUIT_FAILURE_THRESHOLD",
                level=ValidationLevel.WARNING,
                type=int,
                required=False,
                default=5,
                min_value=1,
            )
        )
        self.add_rule(
            ValidationRule(
                name="CIRCUIT_RESET_TIMEOUT",
                level=ValidationLevel.WARNING,
                type=int,
                required=False,
                default=60,
                min_value=1,
            )
        )

        # Rate limiting settings
        self.add_rule(
            ValidationRule(
                name="RATE_LIMIT_REQUESTS",
                level=ValidationLevel.WARNING,
                type=int,
                required=False,
                default=100,
                min_value=1,
            )
        )
        self.add_rule(
            ValidationRule(
                name="RATE_LIMIT_WINDOW",
                level=ValidationLevel.WARNING,
                type=int,
                required=False,
                default=60,
                min_value=1,
            )
        )

    def add_rule(self, rule: ValidationRule) -> None:
        """Add validation rule.

        Args:
            rule: Validation rule
        """
        self._rules[rule.name] = rule

    def validate(self) -> List[ValidationResult]:
        """Validate configuration settings.

        Returns:
            List of validation results
        """
        self._results = []

        # Validate each rule
        for rule in self._rules.values():
            result = self._validate_rule(rule)
            self._results.append(result)

            # Update metrics
            if self.metrics:
                self.metrics.increment(
                    "config_validation",
                    1,
                    {
                        "setting": rule.name,
                        "level": rule.level.value,
                        "valid": str(result.valid),
                    },
                )

        return self._results

    def _validate_rule(self, rule: ValidationRule) -> ValidationResult:
        """Validate configuration setting.

        Args:
            rule: Validation rule

        Returns:
            Validation result
        """
        # Get setting value
        value = os.environ.get(rule.name, rule.default)

        # Check if required
        if rule.required and value is None:
            return ValidationResult(
                name=rule.name,
                level=rule.level,
                valid=False,
                value=value,
                message=f"Required setting {rule.name} is not set",
                rule=rule,
            )

        # Skip validation if not required and not set
        if not rule.required and value is None:
            return ValidationResult(
                name=rule.name,
                level=rule.level,
                valid=True,
                value=value,
                message=f"Optional setting {rule.name} is not set",
                rule=rule,
            )

        # Validate type
        try:
            value = rule.type(value)
        except (ValueError, TypeError):
            return ValidationResult(
                name=rule.name,
                level=rule.level,
                valid=False,
                value=value,
                message=f"Setting {rule.name} must be of type {rule.type.__name__}",
                rule=rule,
            )

        # Validate numeric range
        if isinstance(value, (int, float)):
            if rule.min_value is not None and value < rule.min_value:
                return ValidationResult(
                    name=rule.name,
                    level=rule.level,
                    valid=False,
                    value=value,
                    message=f"Setting {rule.name} must be >= {rule.min_value}",
                    rule=rule,
                )
            if rule.max_value is not None and value > rule.max_value:
                return ValidationResult(
                    name=rule.name,
                    level=rule.level,
                    valid=False,
                    value=value,
                    message=f"Setting {rule.name} must be <= {rule.max_value}",
                    rule=rule,
                )

        # Validate string length
        if isinstance(value, str):
            if rule.min_length is not None and len(value) < rule.min_length:
                return ValidationResult(
                    name=rule.name,
                    level=rule.level,
                    valid=False,
                    value=value,
                    message=f"Setting {rule.name} must be at least {rule.min_length} characters",
                    rule=rule,
                )
            if rule.max_length is not None and len(value) > rule.max_length:
                return ValidationResult(
                    name=rule.name,
                    level=rule.level,
                    valid=False,
                    value=value,
                    message=f"Setting {rule.name} must be at most {rule.max_length} characters",
                    rule=rule,
                )

        # Validate pattern
        if rule.pattern is not None and isinstance(value, str):
            if not re.match(rule.pattern, value):
                return ValidationResult(
                    name=rule.name,
                    level=rule.level,
                    valid=False,
                    value=value,
                    message=f"Setting {rule.name} must match pattern {rule.pattern}",
                    rule=rule,
                )

        # Validate allowed values
        if rule.allowed_values is not None and value not in rule.allowed_values:
            return ValidationResult(
                name=rule.name,
                level=rule.level,
                valid=False,
                value=value,
                message=f"Setting {rule.name} must be one of {rule.allowed_values}",
                rule=rule,
            )

        # Validate dependencies
        if rule.dependencies is not None:
            for dependency in rule.dependencies:
                if dependency not in os.environ:
                    return ValidationResult(
                        name=rule.name,
                        level=rule.level,
                        valid=False,
                        value=value,
                        message=f"Setting {rule.name} requires {dependency} to be set",
                        rule=rule,
                    )

        # Validate conflicts
        if rule.conflicts is not None:
            for conflict in rule.conflicts:
                if conflict in os.environ:
                    return ValidationResult(
                        name=rule.name,
                        level=rule.level,
                        valid=False,
                        value=value,
                        message=f"Setting {rule.name} conflicts with {conflict}",
                        rule=rule,
                    )

        # Run custom validator
        if rule.custom_validator is not None:
            try:
                valid, message = rule.custom_validator(value)
                if not valid:
                    return ValidationResult(
                        name=rule.name,
                        level=rule.level,
                        valid=False,
                        value=value,
                        message=message,
                        rule=rule,
                    )
            except Exception as e:
                return ValidationResult(
                    name=rule.name,
                    level=rule.level,
                    valid=False,
                    value=value,
                    message=f"Custom validation failed: {e}",
                    rule=rule,
                )

        # All validations passed
        return ValidationResult(
            name=rule.name,
            level=rule.level,
            valid=True,
            value=value,
            message=f"Setting {rule.name} is valid",
            rule=rule,
        )

    def _validate_directory(self, path: str) -> tuple[bool, str]:
        """Validate directory path.

        Args:
            path: Directory path

        Returns:
            Tuple of (valid, message)
        """
        if not os.path.exists(path):
            return False, f"Directory {path} does not exist"
        if not os.path.isdir(path):
            return False, f"Path {path} is not a directory"
        if not os.access(path, os.R_OK):
            return False, f"Directory {path} is not readable"
        return True, f"Directory {path} is valid"

    def get_errors(self) -> List[ValidationResult]:
        """Get validation errors.

        Returns:
            List of error results
        """
        return [
            result for result in self._results
            if not result.valid and result.level == ValidationLevel.ERROR
        ]

    def get_warnings(self) -> List[ValidationResult]:
        """Get validation warnings.

        Returns:
            List of warning results
        """
        return [
            result for result in self._results
            if not result.valid and result.level == ValidationLevel.WARNING
        ]

    def get_info(self) -> List[ValidationResult]:
        """Get validation info.

        Returns:
            List of info results
        """
        return [
            result for result in self._results
            if not result.valid and result.level == ValidationLevel.INFO
        ]


# Create validator instance
config_validator = ConfigValidator(Config(), MetricsManager())

__all__ = [
    "ValidationLevel",
    "ValidationRule",
    "ValidationResult",
    "ConfigValidator",
    "config_validator",
] 