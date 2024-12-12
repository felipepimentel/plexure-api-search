"""Constants used throughout the application."""

import os

# File and directory paths
DEFAULT_API_DIR = "assets/apis"
INDEX_FILE = "api_index.pkl"

# API endpoints
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# Search parameters
DEFAULT_MIN_SCORE = 0.5
DEFAULT_TOP_K = 5

# Headers
DEFAULT_HEADERS = {
    "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
    "HTTP-Referer": "https://github.com/cursor-ai",
    "Content-Type": "application/json",
}
