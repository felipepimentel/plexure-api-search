"""Dependency injection container."""

from dependency_injector import containers, providers
from dependency_injector.wiring import inject, provider

from ..config import Config, config_instance
from ..embedding.batch import BatchProcessor
from ..embedding.embeddings import EmbeddingManager
from ..integrations import (
    FAISSPreprocessor,
    PineconeClient,
    VectorManager,
)
from ..integrations.pinecone_pool import PineconePool
from ..monitoring.events import Publisher, Subscriber
from ..monitoring.metrics import MetricsManager
from ..search.searcher import APISearcher
from ..search.strategies import create_default_strategy
from ..search.vector_ops import (
    cosine_similarity,
    euclidean_distance,
    dot_product,
    top_k_indices,
    normalize_vectors,
    vector_mean,
    vector_quantize,
    vector_dequantize,
)
from ..services.circuit_breaker import CircuitBreakerService
from ..services.discovery import ServiceRegistry
from ..services.events import PublisherService, SubscriberService
from ..services.models import ModelService
from ..services.vector_store import VectorStoreService
from ..utils.cache import DiskCache


class Container(containers.DeclarativeContainer):
    """Dependency injection container."""

    # Configuration
    config = providers.Singleton(Config)

    # Core Services
    vector_manager = providers.Singleton(
        VectorManager,
        config=config,
    )

    pinecone_pool = providers.Singleton(
        PineconePool,
        config=config,
        publisher=publisher,
        metrics_manager=metrics_manager,
        circuit_breaker=circuit_breaker,
    )

    pinecone_client = providers.Singleton(
        PineconeClient,
        config=config,
        pool=pinecone_pool,
    )

    faiss_preprocessor = providers.Singleton(
        FAISSPreprocessor,
        config=config,
    )

    # Event System
    publisher = providers.Singleton(
        PublisherService,
        config=config,
        metrics_manager=metrics_manager,
    )

    subscriber = providers.Singleton(
        SubscriberService,
        config=config,
        metrics_manager=metrics_manager,
    )

    # Service Infrastructure
    service_registry = providers.Singleton(
        ServiceRegistry,
        config=config,
        publisher=publisher,
        metrics_manager=metrics_manager,
    )

    circuit_breaker = providers.Singleton(
        CircuitBreakerService,
        config=config,
        publisher=publisher,
        metrics_manager=metrics_manager,
    )

    # Model Management
    model_service = providers.Singleton(
        ModelService,
        config=config,
        publisher=publisher,
        metrics_manager=metrics_manager,
    )

    # Vector Store
    vector_store = providers.Singleton(
        VectorStoreService,
        config=config,
        publisher=publisher,
        metrics_manager=metrics_manager,
        store=vector_manager,
    )

    # Batch Processing
    batch_processor = providers.Singleton(
        BatchProcessor,
        config=config,
        publisher=publisher,
        metrics_manager=metrics_manager,
        model_service=model_service,
    )

    # Embedding and Search
    embedding_manager = providers.Singleton(
        EmbeddingManager,
        config=config,
        model_service=model_service,
        batch_processor=batch_processor,
        publisher=publisher,
    )

    search_strategy = providers.Singleton(
        create_default_strategy,
    )

    api_searcher = providers.Singleton(
        APISearcher,
        config=config,
        vector_store=vector_store,
        embedding_manager=embedding_manager,
        search_strategy=search_strategy,
        publisher=publisher,
    )

    # Monitoring
    metrics_manager = providers.Singleton(
        MetricsManager,
        config=config,
    )

    # Caching
    disk_cache = providers.Factory(
        DiskCache,
        config=config,
    )

    # Vector Operations
    vector_ops = providers.Dict({
        "cosine_similarity": providers.Object(cosine_similarity),
        "euclidean_distance": providers.Object(euclidean_distance),
        "dot_product": providers.Object(dot_product),
        "top_k_indices": providers.Object(top_k_indices),
        "normalize_vectors": providers.Object(normalize_vectors),
        "vector_mean": providers.Object(vector_mean),
        "vector_quantize": providers.Object(vector_quantize),
        "vector_dequantize": providers.Object(vector_dequantize),
    })


# Create global container instance
container = Container()

# Wire the container
container.wire(
    modules=[
        "plexure_api_search.cli.commands",
        "plexure_api_search.search",
        "plexure_api_search.indexing",
        "plexure_api_search.embedding",
        "plexure_api_search.monitoring",
        "plexure_api_search.services",
        "plexure_api_search.integrations",
    ]
)

__all__ = ["container", "Container"] 