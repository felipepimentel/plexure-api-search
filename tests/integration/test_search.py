"""Integration tests for search functionality."""

import pytest
import asyncio
from typing import List, Dict, Any
import json
import yaml
import os
from pathlib import Path

from plexure_api_search.search import searcher
from plexure_api_search.indexing import indexer
from plexure_api_search.config import Config
from plexure_api_search.monitoring import metrics

# Test data
TEST_APIS = [
    {
        "path": "/users",
        "method": "POST",
        "description": "Create a new user account",
        "parameters": [
            {
                "name": "username",
                "type": "string",
                "required": True
            },
            {
                "name": "email",
                "type": "string",
                "required": True
            }
        ],
        "responses": {
            "201": {
                "description": "User created successfully"
            }
        }
    },
    {
        "path": "/users/{user_id}",
        "method": "GET",
        "description": "Get user details by ID",
        "parameters": [
            {
                "name": "user_id",
                "type": "string",
                "required": True,
                "in": "path"
            }
        ],
        "responses": {
            "200": {
                "description": "User details retrieved"
            }
        }
    }
]

@pytest.fixture
async def setup_test_data(tmp_path: Path):
    """Set up test data and index."""
    # Create test API files
    api_dir = tmp_path / "apis"
    api_dir.mkdir()
    
    api_file = api_dir / "test.yaml"
    with open(api_file, "w") as f:
        yaml.dump(TEST_APIS, f)
    
    # Configure test environment
    os.environ["API_DIR"] = str(api_dir)
    os.environ["PINECONE_API_KEY"] = "test_key"
    os.environ["PINECONE_ENV"] = "test"
    
    # Initialize components
    config = Config()
    await indexer.initialize(config)
    await searcher.initialize(config)
    
    # Index test data
    await indexer.index_apis()
    
    yield
    
    # Cleanup
    await indexer.cleanup()
    await searcher.cleanup()

@pytest.mark.asyncio
async def test_basic_search(setup_test_data):
    """Test basic search functionality."""
    # Search for user creation
    results = await searcher.search("create user account")
    
    assert len(results) > 0
    assert any(
        r["path"] == "/users" and r["method"] == "POST"
        for r in results
    )
    assert results[0]["score"] >= 0.5

@pytest.mark.asyncio
async def test_filtered_search(setup_test_data):
    """Test search with filters."""
    # Search with method filter
    results = await searcher.search(
        "user",
        filters={"method": "GET"}
    )
    
    assert len(results) > 0
    assert all(r["method"] == "GET" for r in results)
    assert any(r["path"] == "/users/{user_id}" for r in results)

@pytest.mark.asyncio
async def test_semantic_search(setup_test_data):
    """Test semantic search capabilities."""
    # Search with semantic variations
    queries = [
        "create new account",
        "sign up user",
        "add new user",
        "user registration"
    ]
    
    for query in queries:
        results = await searcher.search(query)
        assert len(results) > 0
        assert any(
            r["path"] == "/users" and r["method"] == "POST"
            for r in results
        )

@pytest.mark.asyncio
async def test_search_ranking(setup_test_data):
    """Test search result ranking."""
    # Search with multiple relevant results
    results = await searcher.search("user")
    
    assert len(results) >= 2
    assert all(
        results[i]["score"] >= results[i+1]["score"]
        for i in range(len(results)-1)
    )

@pytest.mark.asyncio
async def test_search_metrics(setup_test_data):
    """Test search metrics collection."""
    # Reset metrics
    metrics.reset()
    
    # Perform searches
    await searcher.search("test query 1")
    await searcher.search("test query 2")
    
    # Check metrics
    stats = await metrics.get_stats()
    assert stats["search_requests"] == 2
    assert stats["search_latency_avg"] > 0

@pytest.mark.asyncio
async def test_error_handling(setup_test_data):
    """Test search error handling."""
    # Test with invalid filters
    with pytest.raises(ValueError):
        await searcher.search(
            "test",
            filters={"invalid": "filter"}
        )
    
    # Test with empty query
    with pytest.raises(ValueError):
        await searcher.search("")

@pytest.mark.asyncio
async def test_concurrent_searches(setup_test_data):
    """Test concurrent search operations."""
    # Prepare concurrent searches
    queries = [
        "user",
        "create",
        "account",
        "details"
    ]
    
    # Run searches concurrently
    results = await asyncio.gather(*[
        searcher.search(query)
        for query in queries
    ])
    
    assert len(results) == len(queries)
    assert all(isinstance(r, list) for r in results)

@pytest.mark.asyncio
async def test_search_caching(setup_test_data):
    """Test search result caching."""
    # First search
    query = "test cache"
    start_time = asyncio.get_event_loop().time()
    results1 = await searcher.search(query)
    first_duration = asyncio.get_event_loop().time() - start_time
    
    # Second search (should be cached)
    start_time = asyncio.get_event_loop().time()
    results2 = await searcher.search(query)
    second_duration = asyncio.get_event_loop().time() - start_time
    
    assert results1 == results2
    assert second_duration < first_duration

@pytest.mark.asyncio
async def test_search_personalization(setup_test_data):
    """Test search personalization."""
    # Search with user context
    results = await searcher.search(
        "user",
        context={
            "user_id": "test_user",
            "preferences": {
                "method": "GET"
            }
        }
    )
    
    assert len(results) > 0
    assert results[0]["method"] == "GET"

@pytest.mark.asyncio
async def test_search_feedback(setup_test_data):
    """Test search feedback loop."""
    # Perform search
    results = await searcher.search("user")
    
    # Submit feedback
    await searcher.record_feedback(
        query="user",
        result_id=results[0]["id"],
        relevant=True
    )
    
    # Check feedback metrics
    stats = await metrics.get_stats()
    assert stats["feedback_count"] > 0

@pytest.mark.asyncio
async def test_search_analytics(setup_test_data):
    """Test search analytics collection."""
    # Perform searches
    queries = [
        "user account",
        "create user",
        "user details"
    ]
    
    for query in queries:
        await searcher.search(query)
    
    # Get analytics
    analytics = await searcher.get_analytics()
    assert analytics["total_searches"] == len(queries)
    assert "user" in analytics["top_terms"]

@pytest.mark.asyncio
async def test_search_monitoring(setup_test_data):
    """Test search monitoring."""
    # Check initial health
    health = await searcher.health_check()
    assert health["status"] == "healthy"
    
    # Perform searches
    for _ in range(5):
        await searcher.search("test")
    
    # Check metrics
    monitoring = await searcher.get_monitoring()
    assert monitoring["requests_per_minute"] > 0
    assert monitoring["average_latency"] > 0

@pytest.mark.asyncio
async def test_search_cleanup(setup_test_data):
    """Test search cleanup."""
    # Perform searches
    await searcher.search("test")
    
    # Cleanup
    await searcher.cleanup()
    
    # Verify cleanup
    with pytest.raises(RuntimeError):
        await searcher.search("test") 