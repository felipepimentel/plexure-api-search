"""Performance tests for search functionality."""

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
from concurrent.futures import ThreadPoolExecutor

from plexure_api_search.search import searcher
from plexure_api_search.indexing import indexer
from plexure_api_search.config import Config
from plexure_api_search.monitoring import metrics

# Test configuration
LOAD_TEST_DURATION = 60  # seconds
LOAD_TEST_USERS = 10
REQUESTS_PER_SECOND = 10
LATENCY_THRESHOLD = 200  # milliseconds
MEMORY_THRESHOLD = 512  # MB
CPU_THRESHOLD = 80  # percent

# Generate test data
def generate_test_apis(num_apis: int) -> List[Dict[str, Any]]:
    """Generate test API contracts."""
    apis = []
    for i in range(num_apis):
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
                "200": {
                    "description": "Success response"
                },
                "400": {
                    "description": "Bad request"
                }
            },
            "tags": [
                f"tag{i % 5}",
                "test"
            ]
        }
        apis.append(api)
    return apis

@pytest.fixture
async def setup_performance_test(tmp_path: Path):
    """Set up performance test environment."""
    # Create test API files
    api_dir = tmp_path / "apis"
    api_dir.mkdir()
    
    # Generate 1000 test APIs
    apis = generate_test_apis(1000)
    
    # Split into multiple files
    for i in range(0, len(apis), 100):
        batch = apis[i:i+100]
        api_file = api_dir / f"apis_{i//100}.yaml"
        with open(api_file, "w") as f:
            yaml.dump(batch, f)
    
    # Configure test environment
    os.environ["API_DIR"] = str(api_dir)
    os.environ["PINECONE_API_KEY"] = "test_key"
    os.environ["PINECONE_ENV"] = "test"
    
    # Initialize with performance configuration
    config = Config()
    config.search.cache_ttl = 300
    config.search.batch_size = 100
    config.vector_store.pool_size = 10
    
    await indexer.initialize(config)
    await searcher.initialize(config)
    
    # Index test data
    await indexer.index_apis()
    
    yield
    
    # Cleanup
    await indexer.cleanup()
    await searcher.cleanup()

class LoadTestUser:
    """Simulated user for load testing."""
    
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
        self.latencies = []
    
    async def run(self, duration: int):
        """Run user simulation."""
        start_time = time.time()
        while time.time() - start_time < duration:
            query = np.random.choice(self.queries)
            
            # Measure latency
            query_start = time.time()
            try:
                results = await searcher.search(
                    query,
                    context={"user_id": self.user_id}
                )
                latency = (time.time() - query_start) * 1000
                self.latencies.append(latency)
            except Exception as e:
                print(f"Search failed for user {self.user_id}: {e}")
            
            # Random delay between requests
            await asyncio.sleep(1 / REQUESTS_PER_SECOND)

@pytest.mark.asyncio
async def test_search_latency(setup_performance_test):
    """Test search latency under load."""
    # Warm up cache
    for query in LoadTestUser("warmup").queries:
        await searcher.search(query)
    
    # Measure latencies
    latencies = []
    for _ in range(100):
        start_time = time.time()
        results = await searcher.search("test query")
        latency = (time.time() - start_time) * 1000
        latencies.append(latency)
    
    # Calculate statistics
    avg_latency = statistics.mean(latencies)
    p95_latency = np.percentile(latencies, 95)
    p99_latency = np.percentile(latencies, 99)
    
    # Assert performance
    assert avg_latency < LATENCY_THRESHOLD
    assert p95_latency < LATENCY_THRESHOLD * 1.5
    assert p99_latency < LATENCY_THRESHOLD * 2.0

@pytest.mark.asyncio
async def test_concurrent_performance(setup_performance_test):
    """Test performance under concurrent load."""
    # Create concurrent searches
    num_concurrent = 50
    queries = ["test query"] * num_concurrent
    
    # Measure concurrent execution
    start_time = time.time()
    results = await asyncio.gather(*[
        searcher.search(query)
        for query in queries
    ])
    total_time = time.time() - start_time
    
    # Calculate throughput
    throughput = num_concurrent / total_time
    
    # Assert performance
    assert len(results) == num_concurrent
    assert throughput >= REQUESTS_PER_SECOND

@pytest.mark.asyncio
async def test_memory_usage(setup_performance_test):
    """Test memory usage under load."""
    import psutil
    process = psutil.Process()
    
    # Measure baseline memory
    baseline_memory = process.memory_info().rss / 1024 / 1024
    
    # Run intensive searches
    for _ in range(1000):
        await searcher.search("test query")
    
    # Measure peak memory
    peak_memory = process.memory_info().rss / 1024 / 1024
    memory_increase = peak_memory - baseline_memory
    
    # Assert performance
    assert memory_increase < MEMORY_THRESHOLD

@pytest.mark.asyncio
async def test_cpu_usage(setup_performance_test):
    """Test CPU usage under load."""
    import psutil
    process = psutil.Process()
    
    # Measure CPU usage during load
    cpu_percentages = []
    start_time = time.time()
    
    while time.time() - start_time < 10:
        # Run searches
        await asyncio.gather(*[
            searcher.search("test query")
            for _ in range(10)
        ])
        
        # Measure CPU
        cpu_percentages.append(process.cpu_percent())
        await asyncio.sleep(0.1)
    
    avg_cpu = statistics.mean(cpu_percentages)
    
    # Assert performance
    assert avg_cpu < CPU_THRESHOLD

@pytest.mark.asyncio
async def test_load_performance(setup_performance_test):
    """Test system performance under sustained load."""
    # Create test users
    users = [
        LoadTestUser(f"user_{i}")
        for i in range(LOAD_TEST_USERS)
    ]
    
    # Run load test
    await asyncio.gather(*[
        user.run(LOAD_TEST_DURATION)
        for user in users
    ])
    
    # Collect results
    all_latencies = []
    for user in users:
        all_latencies.extend(user.latencies)
    
    # Calculate statistics
    avg_latency = statistics.mean(all_latencies)
    p95_latency = np.percentile(all_latencies, 95)
    p99_latency = np.percentile(all_latencies, 99)
    
    # Get monitoring metrics
    monitoring = await searcher.get_monitoring()
    
    # Assert performance
    assert avg_latency < LATENCY_THRESHOLD
    assert p95_latency < LATENCY_THRESHOLD * 1.5
    assert p99_latency < LATENCY_THRESHOLD * 2.0
    assert monitoring["error_rate"] < 0.01

@pytest.mark.asyncio
async def test_cache_performance(setup_performance_test):
    """Test cache performance."""
    # Measure uncached performance
    uncached_latencies = []
    for query in LoadTestUser("cache_test").queries:
        start_time = time.time()
        await searcher.search(query)
        latency = (time.time() - start_time) * 1000
        uncached_latencies.append(latency)
    
    # Measure cached performance
    cached_latencies = []
    for query in LoadTestUser("cache_test").queries:
        start_time = time.time()
        await searcher.search(query)
        latency = (time.time() - start_time) * 1000
        cached_latencies.append(latency)
    
    # Calculate improvement
    avg_uncached = statistics.mean(uncached_latencies)
    avg_cached = statistics.mean(cached_latencies)
    improvement = (avg_uncached - avg_cached) / avg_uncached * 100
    
    # Assert performance
    assert improvement > 50  # At least 50% faster with cache

@pytest.mark.asyncio
async def test_search_scalability(setup_performance_test):
    """Test search scalability with increasing load."""
    latencies = []
    
    # Test with increasing concurrent requests
    for num_concurrent in [1, 10, 25, 50, 100]:
        start_time = time.time()
        await asyncio.gather(*[
            searcher.search("test query")
            for _ in range(num_concurrent)
        ])
        total_time = time.time() - start_time
        latency = total_time / num_concurrent * 1000
        latencies.append(latency)
    
    # Calculate scalability
    baseline = latencies[0]
    scaling_factor = max(latencies) / baseline
    
    # Assert performance
    assert scaling_factor < 3  # Should scale sub-linearly

@pytest.mark.asyncio
async def test_resource_cleanup(setup_performance_test):
    """Test resource cleanup under load."""
    import psutil
    process = psutil.Process()
    
    # Run intensive operations
    for _ in range(10):
        users = [
            LoadTestUser(f"cleanup_test_{i}")
            for i in range(5)
        ]
        await asyncio.gather(*[
            user.run(5)
            for user in users
        ])
        
        # Force cleanup
        await searcher.cleanup()
        
        # Check resource usage
        memory = process.memory_info().rss / 1024 / 1024
        assert memory < MEMORY_THRESHOLD
    
    # Verify final state
    assert await searcher.health_check()["status"] == "healthy"

@pytest.mark.asyncio
async def test_error_recovery(setup_performance_test):
    """Test performance during error recovery."""
    # Simulate errors
    error_count = 0
    total_requests = 1000
    
    for _ in range(total_requests):
        try:
            if np.random.random() < 0.1:  # 10% error rate
                raise Exception("Simulated error")
            await searcher.search("test query")
        except Exception:
            error_count += 1
            await asyncio.sleep(0.1)  # Brief recovery time
    
    # Check error rate
    error_rate = error_count / total_requests
    
    # Assert performance
    assert error_rate < 0.15  # Should handle errors gracefully

@pytest.mark.asyncio
async def test_long_running_stability(setup_performance_test):
    """Test system stability over long running period."""
    # Run for 5 minutes
    duration = 300
    start_time = time.time()
    metrics_samples = []
    
    while time.time() - start_time < duration:
        # Run searches
        await asyncio.gather(*[
            searcher.search("test query")
            for _ in range(10)
        ])
        
        # Collect metrics
        metrics_samples.append(await searcher.get_monitoring())
        await asyncio.sleep(1)
    
    # Analyze stability
    error_rates = [m["error_rate"] for m in metrics_samples]
    latencies = [m["average_latency"] for m in metrics_samples]
    
    # Assert performance
    assert max(error_rates) < 0.01
    assert statistics.stdev(latencies) < LATENCY_THRESHOLD * 0.1 