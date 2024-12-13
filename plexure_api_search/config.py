"""Configuration module for the API search engine."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API directory configuration
API_DIR = os.getenv("API_DIR", "assets/apis")

# Pinecone configuration
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_INDEX = os.getenv("PINECONE_INDEX_NAME")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")

# Construct Pinecone environment
PINECONE_ENVIRONMENT = (
    f"{PINECONE_REGION}"  # Pinecone environment is just the region for AWS
)

# OpenRouter configuration
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Model configuration
MODEL_NAME = os.getenv("MODEL_NAME", "all-MiniLM-L6-v2")
EMBEDDING_DIMENSION = 384  # Matches the model's output dimension

# Cache configuration
CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Metrics configuration
METRICS_DIR = Path(".metrics")
METRICS_DIR.mkdir(parents=True, exist_ok=True)

# Health check configuration
HEALTH_DIR = Path(".health")
HEALTH_DIR.mkdir(parents=True, exist_ok=True)

# Validate required configuration
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is required")
if not PINECONE_INDEX:
    raise ValueError("PINECONE_INDEX_NAME is required")

# Print configuration (for debugging)
print("\nPinecone Configuration:")
print(f"Index: {PINECONE_INDEX}")
print(f"Region: {PINECONE_REGION}")
print(f"Environment: {PINECONE_ENVIRONMENT}")
print(f"Cloud: {PINECONE_CLOUD}")

__all__ = [
    "API_DIR",
    "PINECONE_API_KEY",
    "PINECONE_ENVIRONMENT",
    "PINECONE_INDEX",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "OPENROUTER_API_KEY",
    "MODEL_NAME",
    "EMBEDDING_DIMENSION",
    "CACHE_DIR",
    "METRICS_DIR",
    "HEALTH_DIR",
]
