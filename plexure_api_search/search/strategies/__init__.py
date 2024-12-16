"""Search strategies initialization."""

from typing import Dict, List, Type

from .base import BaseSearchStrategy, CompositeSearchStrategy, SearchConfig, StrategyFactory
from .cross_encoder import CrossEncoderStrategy
from .hyde import HyDESearchStrategy
from .multi_vector import MultiVectorStrategy
from .self_querying import SelfQueryingStrategy
from .sparse_dense import SparseDenseStrategy

# Default strategy weights
DEFAULT_WEIGHTS = {
    "hyde": 0.15,
    "self_querying": 0.20,
    "multi_vector": 0.15,
    "cross_encoder": 0.25,
    "sparse_dense": 0.25,
}

def create_default_strategy() -> CompositeSearchStrategy:
    """Create default composite strategy with all registered strategies.

    Returns:
        Composite search strategy
    """
    # Create configs with default weights
    configs: Dict[str, SearchConfig] = {}
    for name, weight in DEFAULT_WEIGHTS.items():
        configs[name] = SearchConfig(
            strategy_name=name,
            enabled=True,
            weight=weight,
        )
    
    # Create composite strategy
    return StrategyFactory.create_composite(
        strategies=list(DEFAULT_WEIGHTS.keys()),
        configs=configs,
    )

__all__ = [
    "BaseSearchStrategy",
    "CompositeSearchStrategy",
    "SearchConfig",
    "StrategyFactory",
    "HyDESearchStrategy",
    "MultiVectorStrategy",
    "SelfQueryingStrategy",
    "CrossEncoderStrategy",
    "SparseDenseStrategy",
    "create_default_strategy",
] 