"""Batch processing for embeddings."""

import asyncio
import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from ..config import Config
from ..monitoring.events import Event, EventType
from ..monitoring.metrics import MetricsManager
from ..services.base import BaseService, ServiceException
from ..services.events import PublisherService
from ..services.models import ModelService

logger = logging.getLogger(__name__)


class BatchProcessor(BaseService[Dict[str, Any]]):
    """Batch processor for embeddings."""

    def __init__(
        self,
        config: Config,
        publisher: PublisherService,
        metrics_manager: MetricsManager,
        model_service: ModelService,
        batch_size: int = 32,
        max_batch_wait: float = 0.1,  # 100ms
    ) -> None:
        """Initialize batch processor.

        Args:
            config: Application configuration
            publisher: Event publisher
            metrics_manager: Metrics manager
            model_service: Model service
            batch_size: Maximum batch size
            max_batch_wait: Maximum wait time for batch in seconds
        """
        super().__init__(config, publisher, metrics_manager)
        self.model_service = model_service
        self.batch_size = batch_size
        self.max_batch_wait = max_batch_wait

        self.queue: asyncio.Queue[Tuple[str, asyncio.Future]] = asyncio.Queue()
        self.current_batch: List[Tuple[str, asyncio.Future]] = []
        self._processor_task: Optional[asyncio.Task] = None
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize processor resources."""
        if self._initialized:
            return

        try:
            # Start processor task
            self._processor_task = asyncio.create_task(self._process_loop())
            self._initialized = True

            self.publisher.publish(
                Event(
                    type=EventType.SERVICE_STARTED,
                    timestamp=datetime.now(),
                    component="batch_processor",
                    description="Batch processor initialized",
                    metadata={
                        "batch_size": self.batch_size,
                        "max_batch_wait": self.max_batch_wait,
                    },
                )
            )

        except Exception as e:
            logger.error(f"Failed to initialize batch processor: {e}")
            raise ServiceException(
                message="Failed to initialize batch processor",
                service_name="BatchProcessor",
                error_code="INIT_FAILED",
                details={"error": str(e)},
            )

    async def cleanup(self) -> None:
        """Clean up processor resources."""
        if self._processor_task:
            self._processor_task.cancel()
            try:
                await self._processor_task
            except asyncio.CancelledError:
                pass

        # Clear queue and reject pending requests
        while not self.queue.empty():
            text, future = await self.queue.get()
            if not future.done():
                future.set_exception(
                    ServiceException(
                        message="Batch processor shutting down",
                        service_name="BatchProcessor",
                        error_code="SHUTDOWN",
                    )
                )

        # Clear current batch
        for text, future in self.current_batch:
            if not future.done():
                future.set_exception(
                    ServiceException(
                        message="Batch processor shutting down",
                        service_name="BatchProcessor",
                        error_code="SHUTDOWN",
                    )
                )
        self.current_batch.clear()

        self._initialized = False
        self.publisher.publish(
            Event(
                type=EventType.SERVICE_STOPPED,
                timestamp=datetime.now(),
                component="batch_processor",
                description="Batch processor stopped",
            )
        )

    async def health_check(self) -> Dict[str, Any]:
        """Check processor health.

        Returns:
            Health check results
        """
        return {
            "service": "BatchProcessor",
            "initialized": self._initialized,
            "queue_size": self.queue.qsize(),
            "current_batch_size": len(self.current_batch),
        }

    async def encode(
        self,
        text: str,
        model_id: str = "bi_encoder",
        use_fallback: bool = True,
    ) -> torch.Tensor:
        """Encode text in batch.

        Args:
            text: Text to encode
            model_id: Model identifier
            use_fallback: Whether to try fallback models

        Returns:
            Encoded vector

        Raises:
            ServiceException: If encoding fails
        """
        if not self._initialized:
            await self.initialize()

        try:
            # Create future for result
            future: asyncio.Future = asyncio.Future()
            await self.queue.put((text, future))

            # Wait for result
            try:
                result = await future
                return result
            except Exception as e:
                logger.error(f"Batch encoding failed: {e}")
                raise ServiceException(
                    message="Batch encoding failed",
                    service_name="BatchProcessor",
                    error_code="ENCODE_FAILED",
                    details={"error": str(e)},
                )

        except Exception as e:
            logger.error(f"Failed to queue text for encoding: {e}")
            raise ServiceException(
                message="Failed to queue text for encoding",
                service_name="BatchProcessor",
                error_code="QUEUE_FAILED",
                details={"error": str(e)},
            )

    async def _process_loop(self) -> None:
        """Process batches of texts."""
        while True:
            try:
                # Start new batch
                start_time = datetime.now()
                self.current_batch.clear()

                # Get first item
                text, future = await self.queue.get()
                self.current_batch.append((text, future))

                # Try to fill batch
                try:
                    while len(self.current_batch) < self.batch_size:
                        # Check if we should wait for more items
                        elapsed = (datetime.now() - start_time).total_seconds()
                        if elapsed >= self.max_batch_wait:
                            break

                        # Try to get next item
                        try:
                            text, future = await asyncio.wait_for(
                                self.queue.get(),
                                timeout=self.max_batch_wait - elapsed,
                            )
                            self.current_batch.append((text, future))
                        except asyncio.TimeoutError:
                            break

                except asyncio.CancelledError:
                    break

                # Process batch
                if self.current_batch:
                    await self._process_batch()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Batch processing failed: {e}")
                # Reject current batch
                for text, future in self.current_batch:
                    if not future.done():
                        future.set_exception(e)
                await asyncio.sleep(1)  # Retry after error

    async def _process_batch(self) -> None:
        """Process current batch."""
        try:
            # Get model
            model = await self.model_service.get_model("bi_encoder")

            # Prepare batch
            texts = [text for text, _ in self.current_batch]
            batch_size = len(texts)

            # Encode batch
            start_time = datetime.now()
            embeddings = model.encode(
                texts,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                show_progress_bar=False,
            )

            # Update metrics
            duration = (datetime.now() - start_time).total_seconds()
            self.metrics.observe(
                "batch_duration",
                duration,
                {"batch_size": batch_size},
            )
            self.metrics.increment(
                "texts_encoded",
                batch_size,
            )

            # Set results
            for i, (text, future) in enumerate(self.current_batch):
                if not future.done():
                    future.set_result(embeddings[i])

        except Exception as e:
            logger.error(f"Failed to process batch: {e}")
            self.metrics.increment("batch_errors", 1)
            # Reject batch
            for text, future in self.current_batch:
                if not future.done():
                    future.set_exception(e)


# Create service instance
batch_processor = BatchProcessor(
    Config(),
    PublisherService(Config(), MetricsManager()),
    MetricsManager(),
    ModelService(Config(), PublisherService(Config(), MetricsManager()), MetricsManager()),
)

__all__ = ["BatchProcessor", "batch_processor"] 