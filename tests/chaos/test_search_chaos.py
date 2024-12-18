"""Chaos testing for search functionality."""

import pytest
import asyncio
import time
import random
import signal
import os
import psutil
import logging
import docker
from typing import List, Dict, Any
from pathlib import Path
from contextlib import contextmanager

from plexure_api_search.search import searcher
from plexure_api_search.indexing import indexer
from plexure_api_search.config import Config
from plexure_api_search.monitoring import metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Chaos configuration
CHAOS_TEST_DURATION = 1800  # 30 minutes
CHAOS_INTERVAL = 60  # 1 minute between chaos events
FAILURE_THRESHOLD = 0.05  # 5% error rate

class ChaosEvent:
    """Base class for chaos events."""
    
    async def execute(self):
        """Execute chaos event."""
        raise NotImplementedError
    
    async def cleanup(self):
        """Clean up after chaos event."""
        raise NotImplementedError

class CPUStress(ChaosEvent):
    """Simulate CPU stress."""
    
    def __init__(self, duration: int = 30):
        self.duration = duration
        self.processes = []
    
    async def execute(self):
        """Execute CPU stress."""
        logger.info("Starting CPU stress")
        
        # Create CPU-intensive processes
        for _ in range(os.cpu_count()):
            process = psutil.Popen([
                "python", "-c",
                "while True: pass"
            ])
            self.processes.append(process)
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up CPU stress."""
        logger.info("Cleaning up CPU stress")
        
        for process in self.processes:
            process.terminate()
            process.wait()
        self.processes = []

class MemoryStress(ChaosEvent):
    """Simulate memory pressure."""
    
    def __init__(self, size_mb: int = 1024, duration: int = 30):
        self.size_mb = size_mb
        self.duration = duration
        self.data = []
    
    async def execute(self):
        """Execute memory stress."""
        logger.info(f"Starting memory stress ({self.size_mb}MB)")
        
        # Allocate memory
        chunk_size = 1024 * 1024  # 1MB
        for _ in range(self.size_mb):
            self.data.append(bytearray(chunk_size))
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up memory stress."""
        logger.info("Cleaning up memory stress")
        self.data = []

class NetworkLatency(ChaosEvent):
    """Simulate network latency."""
    
    def __init__(self, latency_ms: int = 100, duration: int = 30):
        self.latency_ms = latency_ms
        self.duration = duration
    
    async def execute(self):
        """Execute network latency."""
        logger.info(f"Adding network latency ({self.latency_ms}ms)")
        
        # Add network delay using tc
        os.system(
            f"tc qdisc add dev lo root netem delay {self.latency_ms}ms"
        )
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up network latency."""
        logger.info("Cleaning up network latency")
        os.system("tc qdisc del dev lo root")

class PacketLoss(ChaosEvent):
    """Simulate packet loss."""
    
    def __init__(self, loss_percent: int = 10, duration: int = 30):
        self.loss_percent = loss_percent
        self.duration = duration
    
    async def execute(self):
        """Execute packet loss."""
        logger.info(f"Adding packet loss ({self.loss_percent}%)")
        
        # Add packet loss using tc
        os.system(
            f"tc qdisc add dev lo root netem loss {self.loss_percent}%"
        )
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up packet loss."""
        logger.info("Cleaning up packet loss")
        os.system("tc qdisc del dev lo root")

class ProcessKiller(ChaosEvent):
    """Simulate process failures."""
    
    def __init__(self, process_name: str, duration: int = 1):
        self.process_name = process_name
        self.duration = duration
        self.killed_pid = None
    
    async def execute(self):
        """Execute process kill."""
        logger.info(f"Killing process {self.process_name}")
        
        for proc in psutil.process_iter(['pid', 'name']):
            if self.process_name in proc.info['name']:
                self.killed_pid = proc.info['pid']
                os.kill(self.killed_pid, signal.SIGTERM)
                break
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up process kill."""
        logger.info("Process kill cleanup not needed")

class DiskStress(ChaosEvent):
    """Simulate disk stress."""
    
    def __init__(self, size_gb: int = 1, duration: int = 30):
        self.size_gb = size_gb
        self.duration = duration
        self.test_file = "chaos_disk_test"
    
    async def execute(self):
        """Execute disk stress."""
        logger.info(f"Starting disk stress ({self.size_gb}GB)")
        
        # Write large file
        with open(self.test_file, "wb") as f:
            f.write(os.urandom(self.size_gb * 1024 * 1024 * 1024))
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up disk stress."""
        logger.info("Cleaning up disk stress")
        if os.path.exists(self.test_file):
            os.remove(self.test_file)

class ContainerChaos(ChaosEvent):
    """Simulate container chaos."""
    
    def __init__(self, container_name: str, action: str, duration: int = 30):
        self.container_name = container_name
        self.action = action
        self.duration = duration
        self.client = docker.from_env()
    
    async def execute(self):
        """Execute container chaos."""
        logger.info(f"Container chaos: {self.action} {self.container_name}")
        
        container = self.client.containers.get(self.container_name)
        if self.action == "stop":
            container.stop()
        elif self.action == "pause":
            container.pause()
        elif self.action == "restart":
            container.restart()
        
        await asyncio.sleep(self.duration)
    
    async def cleanup(self):
        """Clean up container chaos."""
        logger.info("Cleaning up container chaos")
        
        container = self.client.containers.get(self.container_name)
        if self.action == "stop":
            container.start()
        elif self.action == "pause":
            container.unpause()

@pytest.fixture
async def setup_chaos_test(tmp_path: Path):
    """Set up chaos test environment."""
    # Configure test environment
    config = Config()
    config.search.cache_ttl = 60
    config.search.batch_size = 100
    config.vector_store.pool_size = 10
    
    await indexer.initialize(config)
    await searcher.initialize(config)
    
    yield
    
    await indexer.cleanup()
    await searcher.cleanup()

class ChaosOrchestrator:
    """Orchestrate chaos testing."""
    
    def __init__(self):
        self.events = [
            CPUStress(duration=30),
            MemoryStress(size_mb=1024, duration=30),
            NetworkLatency(latency_ms=200, duration=30),
            PacketLoss(loss_percent=20, duration=30),
            ProcessKiller("python", duration=1),
            DiskStress(size_gb=1, duration=30),
        ]
        
        if os.getenv("DOCKER_ENABLED"):
            self.events.extend([
                ContainerChaos("pinecone", "pause", duration=30),
                ContainerChaos("redis", "restart", duration=30),
            ])
    
    async def run_chaos(self, duration: int):
        """Run chaos testing.
        
        Args:
            duration: Test duration in seconds
        """
        start_time = time.time()
        
        while time.time() - start_time < duration:
            # Select random chaos event
            event = random.choice(self.events)
            
            try:
                # Execute chaos
                await event.execute()
                
                # Wait for system to stabilize
                await asyncio.sleep(10)
                
            except Exception as e:
                logger.error(f"Chaos event failed: {e}")
            
            finally:
                # Clean up
                await event.cleanup()
            
            # Wait before next chaos event
            await asyncio.sleep(CHAOS_INTERVAL)

@pytest.mark.asyncio
async def test_chaos_resilience(setup_chaos_test):
    """Test system resilience under chaos conditions."""
    logger.info("Starting chaos testing")
    
    # Create orchestrator
    orchestrator = ChaosOrchestrator()
    
    # Start chaos testing
    chaos_task = asyncio.create_task(
        orchestrator.run_chaos(CHAOS_TEST_DURATION)
    )
    
    # Run test queries
    total_requests = 0
    total_errors = 0
    latencies = []
    
    start_time = time.time()
    while time.time() - start_time < CHAOS_TEST_DURATION:
        try:
            # Perform search
            query_start = time.time()
            await searcher.search("test query")
            latency = (time.time() - query_start) * 1000
            latencies.append(latency)
            total_requests += 1
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            total_errors += 1
            total_requests += 1
        
        await asyncio.sleep(1)
    
    # Cancel chaos
    chaos_task.cancel()
    try:
        await chaos_task
    except asyncio.CancelledError:
        pass
    
    # Calculate statistics
    error_rate = total_errors / total_requests if total_requests > 0 else 1.0
    avg_latency = sum(latencies) / len(latencies) if latencies else float('inf')
    
    # Log results
    logger.info("Chaos test completed:")
    logger.info(f"Total requests: {total_requests}")
    logger.info(f"Total errors: {total_errors}")
    logger.info(f"Error rate: {error_rate:.2%}")
    logger.info(f"Average latency: {avg_latency:.2f}ms")
    
    # Assert resilience
    assert error_rate < FAILURE_THRESHOLD
    assert avg_latency < 5000  # 5 seconds

@pytest.mark.asyncio
async def test_recovery_patterns(setup_chaos_test):
    """Test system recovery patterns after chaos events."""
    logger.info("Testing recovery patterns")
    
    # Test each chaos event separately
    orchestrator = ChaosOrchestrator()
    for event in orchestrator.events:
        logger.info(f"Testing recovery from {event.__class__.__name__}")
        
        # Execute chaos event
        await event.execute()
        
        # Measure recovery
        start_time = time.time()
        recovered = False
        
        while time.time() - start_time < 60:  # 1 minute timeout
            try:
                await searcher.search("test query")
                recovered = True
                break
            except Exception:
                await asyncio.sleep(1)
        
        # Clean up
        await event.cleanup()
        
        # Assert recovery
        assert recovered, f"System did not recover from {event.__class__.__name__}"
        
        # Wait between events
        await asyncio.sleep(10)

@pytest.mark.asyncio
async def test_cascading_failures(setup_chaos_test):
    """Test system behavior under cascading failures."""
    logger.info("Testing cascading failures")
    
    # Create multiple simultaneous chaos events
    events = [
        CPUStress(duration=30),
        MemoryStress(size_mb=512, duration=30),
        NetworkLatency(latency_ms=100, duration=30)
    ]
    
    # Execute events in sequence with overlap
    for event in events:
        asyncio.create_task(event.execute())
        await asyncio.sleep(10)  # Partial overlap
    
    # Monitor system behavior
    total_requests = 0
    total_errors = 0
    
    for _ in range(60):  # Monitor for 1 minute
        try:
            await searcher.search("test query")
            total_requests += 1
        except Exception:
            total_errors += 1
            total_requests += 1
        await asyncio.sleep(1)
    
    # Clean up
    for event in events:
        await event.cleanup()
    
    # Calculate error rate
    error_rate = total_errors / total_requests if total_requests > 0 else 1.0
    
    # Assert resilience
    assert error_rate < FAILURE_THRESHOLD * 2  # Allow higher error rate for cascading failures

@pytest.mark.asyncio
async def test_partial_failures(setup_chaos_test):
    """Test system behavior under partial failures."""
    logger.info("Testing partial failures")
    
    # Simulate partial system degradation
    async with asyncio.TaskGroup() as group:
        # Add moderate CPU load
        group.create_task(
            CPUStress(duration=60).execute()
        )
        
        # Add moderate network latency
        group.create_task(
            NetworkLatency(latency_ms=50, duration=60).execute()
        )
        
        # Run test queries
        results = []
        for _ in range(10):
            try:
                result = await searcher.search("test query")
                results.append(len(result))
            except Exception as e:
                logger.error(f"Search failed: {e}")
            await asyncio.sleep(1)
    
    # Assert degraded performance
    assert len(results) > 0
    assert all(count > 0 for count in results)  # Should still return results

@pytest.mark.asyncio
async def test_resource_exhaustion(setup_chaos_test):
    """Test system behavior under resource exhaustion."""
    logger.info("Testing resource exhaustion")
    
    # Monitor resource usage during test
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Create resource pressure
    memory_stress = MemoryStress(size_mb=1024, duration=60)
    cpu_stress = CPUStress(duration=60)
    
    async with asyncio.TaskGroup() as group:
        # Add resource stress
        group.create_task(memory_stress.execute())
        group.create_task(cpu_stress.execute())
        
        # Run test queries
        for _ in range(60):
            try:
                await searcher.search("test query")
            except Exception as e:
                logger.error(f"Search failed: {e}")
            await asyncio.sleep(1)
    
    # Check resource usage
    final_memory = process.memory_info().rss
    memory_growth = (final_memory - initial_memory) / initial_memory
    
    # Assert resource management
    assert memory_growth < 3.0  # Less than 300% growth 