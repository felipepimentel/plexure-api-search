import pytest
from plexure_api_search.integrations.llm.openrouter_client import OpenRouterClient

def test_clean_json_content_valid():
    client = OpenRouterClient()
    valid_json = '''{
        "detailed_description": "Test description",
        "use_cases": ["case1", "case2"],
        "best_practices": ["practice1"],
        "related_endpoints": []
    }'''
    
    result = client.clean_json_content(valid_json)
    assert result["detailed_description"] == "Test description"
    assert len(result["use_cases"]) == 2

def test_clean_json_content_invalid_quotes():
    client = OpenRouterClient()
    invalid_json = '''{
        'detailed_description': 'Test description',
        'use_cases': ['case1', 'case2'],
        'best_practices': ['practice1'],
        'related_endpoints': []
    }'''
    
    result = client.clean_json_content(invalid_json)
    assert result["detailed_description"] == "Test description"

def test_clean_json_content_missing_quotes():
    client = OpenRouterClient()
    invalid_json = '''{
        detailed_description: "Test description",
        use_cases: ["case1", "case2"],
        best_practices: ["practice1"],
        related_endpoints: []
    }'''
    
    result = client.clean_json_content(invalid_json)
    assert result["detailed_description"] == "Test description"

def test_enrich_endpoint_content():
    client = OpenRouterClient()
    content = {
        "detailed_description": "Test API",
        "use_cases": ["test"],
        "best_practices": ["practice"],
        "related_endpoints": []
    }
    
    result = client.enrich_endpoint_content(content)
    assert result["detailed_description"] == "Test API"

def test_invalid_content_returns_minimal_structure():
    client = OpenRouterClient()
    invalid_content = "{invalid json}"
    
    result = client.clean_json_content(invalid_content)
    assert "detailed_description" in result
    assert isinstance(result["use_cases"], list) 