"""
Configuration Package for Plexure API Search

This package provides configuration management functionality for the Plexure API Search system.
It handles loading, validating, and providing access to configuration settings from various
sources including environment variables and configuration files.

Key Components:
1. Configuration Loading:
   - Environment variables
   - Configuration files
   - Default values
   - Validation rules

2. Configuration Management:
   - Setting access
   - Value validation
   - Type conversion
   - Default handling

3. Environment Support:
   - Development settings
   - Staging settings
   - Production settings
   - Testing settings

The package supports:
- Multiple configuration sources
- Environment-specific settings
- Type validation
- Default values
- Configuration updates
- Setting persistence

Example Usage:
    from plexure_api_search.config import config

    # Access settings
    model_name = config.model_name
    batch_size = config.batch_size

    # Check environment
    is_prod = config.environment == "production"

    # Update settings
    config.update({
        "batch_size": 64,
        "cache_enabled": True
    })

Package Structure:
- __init__.py: Package initialization and exports
- settings.py: Configuration definitions
- validation.py: Setting validation
"""

from .settings import config

__all__ = ["config"]
