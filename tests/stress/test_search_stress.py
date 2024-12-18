"""Stress tests for search functionality."""

import pytest
import asyncio
import time
import statistics
from typing import List, Dict, Any
import json
import yaml
import os
from pathlib import Path
import numpy as np
import psutil
import logging
from concurrent.futures import ThreadPoolExecutor

from plexure_api_search.search import searcher
from plexure_api_search.indexing import indexer
from plexure_api_search.config import Config
from plexure_api_search.monitoring import metrics

# Test configuration
STRESS_TEST_DURATION = 3600  # 1 hour
MAX_CONCURRENT_USERS = 100
RAMP_UP_TIME = 300  # 5 minutes
COOL_DOWN_TIME = 300  # 5 minutes
MAX_REQUESTS_PER_SECOND = 50
FAILURE_THRESHOLD = 0.01  # 1% error rate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StressTestUser:
    """Simulated user for stress testing."""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.queries = [
            "create resource",
            "get data",
            "update record",
            "delete item",
            "list all",
            "search by tag",
            "filter results",
            "sort items",
            "paginate data",
            "export results"
        ]
        self.stats = {
            "requests": 0,
            "errors": 0,
            "latencies": [],
        }
    
    async def run(self, duration: int, request_rate: float):
        """Run user simulation.
        
        Args:
            duration: Test duration in seconds
            request_rate: Requests per second
        """
        start_time = time.time()
        while time.time() - start_time < duration:
            try:
                # Random query with context
                query = np.random.choice(self.queries)
                context = {
                    "user_id": self.user_id,
                    "timestamp": time.time(),
                    "client": "stress_test"
                }
                
                # Measure latency
                query_start = time.time()
                results = await searcher.search(query, context=context)
                latency = (time.time() - query_start) * 1000
                
                # Update stats
                self.stats["requests"] += 1
                self.stats["latencies"].append(latency)
                
                # Random delay based on request rate
                await asyncio.sleep(1 / request_rate)
                
            except Exception as e:
                logger.error(f"Search failed for user {self.user_id}: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(1)  # Back off on error

@pytest.fixture
async def setup_stress_test(tmp_path: Path):
    """Set up stress test environment."""
    # Create test API files
    api_dir = tmp_path / "apis"
    api_dir.mkdir()
    
    # Generate 10000 test APIs
    apis = []
    for i in range(10000):
        api = {
            "path": f"/api/v1/resource{i}",
            "method": "GET" if i % 2 == 0 else "POST",
            "description": f"Test API endpoint {i}",
            "parameters": [
                {
                    "name": "param1",
                    "type": "string",
                    "required": True
                },
                {
                    "name": "param2",
                    "type": "integer",
                    "required": False
                }
            ],
            "responses": {
                "200": {"description": "Success"},
                "400": {"description": "Bad request"},
                "500": {"description": "Server error"}
            },
            "tags": [f"tag{i%10}", "stress_test"]
        }
        apis.append(api)
    
    # Split into files
    for i in range(0, len(apis), 1000):
        batch = apis[i:i+1000]
        api_file = api_dir / f"apis_{i//1000}.yaml"
        with open(api_file, "w") as f:
            yaml.dump(batch, f)
    
    # Configure environment
    os.environ["API_DIR"] = str(api_dir)
    os.environ["PINECONE_API_KEY"] = "test_key"
    os.environ["PINECONE_ENV"] = "test"
    
    # Initialize with stress test configuration
    config = Config()
    config.search.cache_ttl = 60
    config.search.batch_size = 100
    config.vector_store.pool_size = 20
    config.monitoring.metrics_interval = 1
    
    await indexer.initialize(config)
    await searcher.initialize(config)
    
    # Index test data
    await indexer.index_apis()
    
    yield
    
    # Cleanup
    await indexer.cleanup()
    await searcher.cleanup()

@pytest.mark.asyncio
async def test_sustained_load(setup_stress_test):
    """Test system under sustained heavy load."""
    logger.info("Starting sustained load test")
    
    # Create users
    users = [
        StressTestUser(f"user_{i}")
        for i in range(MAX_CONCURRENT_USERS)
    ]
    
    # Run stress test
    start_time = time.time()
    user_tasks = []
    
    # Ramp up
    logger.info("Starting ramp up")
    for i, user in enumerate(users):
        delay = i * (RAMP_UP_TIME / len(users))
        task = asyncio.create_task(
            user.run(
                STRESS_TEST_DURATION - delay,
                MAX_REQUESTS_PER_SECOND / len(users)
            )
        )
        user_tasks.append(task)
        await asyncio.sleep(delay)
    
    # Wait for completion
    logger.info("All users started, waiting for completion")
    await asyncio.gather(*user_tasks)
    
    # Calculate statistics
    total_requests = sum(u.stats["requests"] for u in users)
    total_errors = sum(u.stats["errors"] for u in users)
    all_latencies = []
    for user in users:
        all_latencies.extend(user.stats["latencies"])
    
    avg_latency = statistics.mean(all_latencies)
    p95_latency = np.percentile(all_latencies, 95)
    p99_latency = np.percentile(all_latencies, 99)
    error_rate = total_errors / total_requests if total_requests > 0 else 1.0
    
    # Log results
    logger.info(f"Stress test completed:")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Error rate: {error_rate:.2%}")
    logger.info(f"Average latency: {avg_latency:.2f}ms")
    logger.info(f"P95 latency: {p95_latency:.2f}ms")
    logger.info(f"P99 latency: {p99_latency:.2f}ms")
    
    # Assert performance
    assert error_rate < FAILURE_THRESHOLD
    assert p95_latency < 1000  # 1 second
    assert p99_latency < 2000  # 2 seconds

@pytest.mark.asyncio
async def test_spike_load(setup_stress_test):
    """Test system under sudden spike in load."""
    logger.info("Starting spike load test")
    
    # Create baseline load
    baseline_users = [
        StressTestUser(f"baseline_user_{i}")
        for i in range(10)
    ]
    
    # Create spike users
    spike_users = [
        StressTestUser(f"spike_user_{i}")
        for i in range(90)
    ]
    
    # Run baseline load
    baseline_tasks = [
        asyncio.create_task(
            user.run(
                STRESS_TEST_DURATION,
                MAX_REQUESTS_PER_SECOND / 10
            )
        )
        for user in baseline_users
    ]
    
    # Wait for stable baseline
    logger.info("Establishing baseline load")
    await asyncio.sleep(60)
    
    # Create spike
    logger.info("Creating load spike")
    spike_tasks = [
        asyncio.create_task(
            user.run(
                60,  # 1 minute spike
                MAX_REQUESTS_PER_SECOND
            )
        )
        for user in spike_users
    ]
    
    # Wait for spike completion
    await asyncio.gather(*spike_tasks)
    
    # Wait for recovery
    logger.info("Spike complete, monitoring recovery")
    await asyncio.sleep(60)
    
    # Cancel baseline tasks
    for task in baseline_tasks:
        task.cancel()
    
    # Calculate statistics
    all_users = baseline_users + spike_users
    total_requests = sum(u.stats["requests"] for u in all_users)
    total_errors = sum(u.stats["errors"] for u in all_users)
    all_latencies = []
    for user in all_users:
        all_latencies.extend(user.stats["latencies"])
    
    error_rate = total_errors / total_requests if total_requests > 0 else 1.0
    p99_latency = np.percentile(all_latencies, 99)
    
    # Assert performance
    assert error_rate < FAILURE_THRESHOLD * 2  # Allow higher error rate during spike
    assert p99_latency < 5000  # 5 seconds during spike

@pytest.mark.asyncio
async def test_resource_limits(setup_stress_test):
    """Test system behavior near resource limits."""
    logger.info("Starting resource limits test")
    
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create maximum users
    users = [
        StressTestUser(f"resource_user_{i}")
        for i in range(MAX_CONCURRENT_USERS)
    ]
    
    # Run at maximum capacity
    tasks = [
        asyncio.create_task(
            user.run(
                300,  # 5 minutes
                MAX_REQUESTS_PER_SECOND / len(users)
            )
        )
        for user in users
    ]
    
    # Monitor resources
    memory_samples = []
    cpu_samples = []
    
    start_time = time.time()
    while time.time() - start_time < 300:
        memory_samples.append(process.memory_info().rss)
        cpu_samples.append(process.cpu_percent())
        await asyncio.sleep(1)
    
    # Cancel tasks
    for task in tasks:
        task.cancel()
    
    # Calculate resource usage
    max_memory = max(memory_samples)
    avg_cpu = statistics.mean(cpu_samples)
    memory_growth = (max_memory - initial_memory) / initial_memory
    
    logger.info(f"Resource usage:")
    logger.info(f"Memory growth: {memory_growth:.2%}")
    logger.info(f"Average CPU: {avg_cpu:.2f}%")
    
    # Assert resource usage
    assert memory_growth < 2.0  # Less than 200% growth
    assert avg_cpu < 90  # Below 90% CPU

@pytest.mark.asyncio
async def test_error_injection(setup_stress_test):
    """Test system resilience with injected errors."""
    logger.info("Starting error injection test")
    
    # Create users
    users = [
        StressTestUser(f"error_user_{i}")
        for i in range(20)
    ]
    
    # Error injection configuration
    error_types = [
        ValueError("Invalid query"),
        TimeoutError("Search timeout"),
        ConnectionError("Network error"),
        MemoryError("Out of memory"),
    ]
    
    async def error_injection():
        """Inject random errors."""
        while True:
            # Random error injection
            if np.random.random() < 0.1:  # 10% error rate
                error = np.random.choice(error_types)
                logger.info(f"Injecting error: {error}")
                raise error
            await asyncio.sleep(1)
    
    # Run test with error injection
    tasks = [
        asyncio.create_task(
            user.run(
                300,  # 5 minutes
                MAX_REQUESTS_PER_SECOND / len(users)
            )
        )
        for user in users
    ]
    
    # Add error injection task
    error_task = asyncio.create_task(error_injection())
    
    # Wait for completion
    try:
        await asyncio.gather(*tasks)
    except Exception as e:
        logger.error(f"Test error: {e}")
    finally:
        error_task.cancel()
    
    # Calculate error statistics
    total_requests = sum(u.stats["requests"] for u in users)
    total_errors = sum(u.stats["errors"] for u in users)
    error_rate = total_errors / total_requests if total_requests > 0 else 1.0
    
    logger.info(f"Error injection results:")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Error rate: {error_rate:.2%}")
    
    # Assert error handling
    assert error_rate < FAILURE_THRESHOLD * 3  # Allow higher error rate during injection

@pytest.mark.asyncio
async def test_recovery_time(setup_stress_test):
    """Test system recovery time after failures."""
    logger.info("Starting recovery time test")
    
    # Create users
    users = [
        StressTestUser(f"recovery_user_{i}")
        for i in range(20)
    ]
    
    # Run normal load
    tasks = [
        asyncio.create_task(
            user.run(
                60,  # 1 minute baseline
                MAX_REQUESTS_PER_SECOND / len(users)
            )
        )
        for user in users
    ]
    
    await asyncio.gather(*tasks)
    
    # Simulate system failure
    logger.info("Simulating system failure")
    await searcher.cleanup()
    
    # Measure recovery time
    start_time = time.time()
    await searcher.initialize(Config())
    recovery_time = time.time() - start_time
    
    # Run post-recovery load
    tasks = [
        asyncio.create_task(
            user.run(
                60,  # 1 minute post-recovery
                MAX_REQUESTS_PER_SECOND / len(users)
            )
        )
        for user in users
    ]
    
    await asyncio.gather(*tasks)
    
    logger.info(f"Recovery time: {recovery_time:.2f} seconds")
    
    # Assert recovery
    assert recovery_time < 30  # Recovery within 30 seconds
    
    # Calculate post-recovery performance
    all_latencies = []
    for user in users:
        all_latencies.extend(user.stats["latencies"])
    
    p95_latency = np.percentile(all_latencies, 95)
    assert p95_latency < 1000  # Normal performance after recovery 