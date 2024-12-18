"""Security tests for search functionality."""

import pytest
import asyncio
import time
import jwt
import json
import logging
import ssl
import aiohttp
import socket
import subprocess
from typing import List, Dict, Any
from pathlib import Path
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64
import os
import secrets

from plexure_api_search.search import searcher
from plexure_api_search.indexing import indexer
from plexure_api_search.config import Config
from plexure_api_search.monitoring import metrics

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security configuration
JWT_SECRET = secrets.token_hex(32)
ENCRYPTION_KEY = Fernet.generate_key()
SSL_CERT = "test_cert.pem"
SSL_KEY = "test_key.pem"

class SecurityTestConfig:
    """Configuration for security tests."""
    
    def __init__(self):
        self.jwt_secret = JWT_SECRET
        self.encryption_key = ENCRYPTION_KEY
        self.ssl_cert = SSL_CERT
        self.ssl_key = SSL_KEY
        self.allowed_origins = ["https://trusted-domain.com"]
        self.rate_limit = 100  # requests per minute
        self.max_payload_size = 1024 * 1024  # 1MB
        self.password_min_length = 12
        self.require_mfa = True

def generate_test_cert():
    """Generate self-signed certificate for testing."""
    subprocess.run([
        "openssl", "req", "-x509", "-newkey", "rsa:4096",
        "-keyout", SSL_KEY,
        "-out", SSL_CERT,
        "-days", "1",
        "-nodes",
        "-subj", "/CN=localhost"
    ])

@pytest.fixture
async def setup_security_test(tmp_path: Path):
    """Set up security test environment."""
    # Generate test certificate
    generate_test_cert()
    
    # Configure test environment
    config = Config()
    config.security = SecurityTestConfig()
    
    await indexer.initialize(config)
    await searcher.initialize(config)
    
    yield
    
    # Cleanup
    await indexer.cleanup()
    await searcher.cleanup()
    
    # Remove test certificates
    os.remove(SSL_CERT)
    os.remove(SSL_KEY)

def generate_jwt(claims: Dict[str, Any]) -> str:
    """Generate JWT token.
    
    Args:
        claims: Token claims
        
    Returns:
        JWT token
    """
    return jwt.encode(claims, JWT_SECRET, algorithm="HS256")

def encrypt_data(data: str) -> str:
    """Encrypt sensitive data.
    
    Args:
        data: Data to encrypt
        
    Returns:
        Encrypted data
    """
    f = Fernet(ENCRYPTION_KEY)
    return f.encrypt(data.encode()).decode()

@pytest.mark.asyncio
async def test_authentication(setup_security_test):
    """Test authentication security."""
    logger.info("Testing authentication")
    
    # Test without token
    with pytest.raises(Exception):
        await searcher.search(
            "test query",
            auth_token=None
        )
    
    # Test with invalid token
    with pytest.raises(jwt.InvalidTokenError):
        await searcher.search(
            "test query",
            auth_token="invalid_token"
        )
    
    # Test with expired token
    expired_token = generate_jwt({
        "exp": int(time.time()) - 3600,
        "sub": "test_user"
    })
    with pytest.raises(jwt.ExpiredSignatureError):
        await searcher.search(
            "test query",
            auth_token=expired_token
        )
    
    # Test with valid token
    valid_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "test_user",
        "scope": "search:read"
    })
    results = await searcher.search(
        "test query",
        auth_token=valid_token
    )
    assert len(results) >= 0

@pytest.mark.asyncio
async def test_authorization(setup_security_test):
    """Test authorization security."""
    logger.info("Testing authorization")
    
    # Test with insufficient permissions
    limited_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "test_user",
        "scope": "search:read"
    })
    with pytest.raises(Exception):
        await searcher.index_api(
            {"path": "/test"},
            auth_token=limited_token
        )
    
    # Test with required permissions
    admin_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "admin_user",
        "scope": "search:read search:write"
    })
    await searcher.index_api(
        {"path": "/test"},
        auth_token=admin_token
    )

@pytest.mark.asyncio
async def test_input_validation(setup_security_test):
    """Test input validation security."""
    logger.info("Testing input validation")
    
    valid_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "test_user",
        "scope": "search:read"
    })
    
    # Test SQL injection
    with pytest.raises(ValueError):
        await searcher.search(
            "test' OR '1'='1",
            auth_token=valid_token
        )
    
    # Test XSS
    with pytest.raises(ValueError):
        await searcher.search(
            "<script>alert('xss')</script>",
            auth_token=valid_token
        )
    
    # Test command injection
    with pytest.raises(ValueError):
        await searcher.search(
            "test; rm -rf /",
            auth_token=valid_token
        )
    
    # Test oversized payload
    large_query = "x" * (1024 * 1024 + 1)  # > 1MB
    with pytest.raises(ValueError):
        await searcher.search(
            large_query,
            auth_token=valid_token
        )

@pytest.mark.asyncio
async def test_rate_limiting(setup_security_test):
    """Test rate limiting security."""
    logger.info("Testing rate limiting")
    
    valid_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "test_user",
        "scope": "search:read"
    })
    
    # Test rate limiting
    for _ in range(100):  # At limit
        await searcher.search(
            "test query",
            auth_token=valid_token
        )
    
    # Should be rate limited
    with pytest.raises(Exception):
        await searcher.search(
            "test query",
            auth_token=valid_token
        )
    
    # Wait for rate limit reset
    await asyncio.sleep(60)
    
    # Should work again
    await searcher.search(
        "test query",
        auth_token=valid_token
    )

@pytest.mark.asyncio
async def test_encryption(setup_security_test):
    """Test data encryption security."""
    logger.info("Testing encryption")
    
    # Test sensitive data handling
    sensitive_data = "password123"
    encrypted = encrypt_data(sensitive_data)
    
    # Verify encrypted data is not plaintext
    assert sensitive_data not in encrypted
    
    # Test encryption in transit
    async with aiohttp.ClientSession() as session:
        # Test unencrypted connection
        with pytest.raises(Exception):
            async with session.get("http://localhost:8000/search") as response:
                assert response.status == 400  # Should require HTTPS
        
        # Test encrypted connection
        ssl_context = ssl.create_default_context(
            cafile=SSL_CERT
        )
        async with session.get(
            "https://localhost:8000/search",
            ssl=ssl_context
        ) as response:
            assert response.status == 401  # Should require auth

@pytest.mark.asyncio
async def test_cors_security(setup_security_test):
    """Test CORS security."""
    logger.info("Testing CORS security")
    
    async with aiohttp.ClientSession() as session:
        # Test allowed origin
        headers = {
            "Origin": "https://trusted-domain.com"
        }
        async with session.get(
            "https://localhost:8000/search",
            headers=headers,
            ssl=False
        ) as response:
            assert "Access-Control-Allow-Origin" in response.headers
        
        # Test disallowed origin
        headers = {
            "Origin": "https://malicious-domain.com"
        }
        async with session.get(
            "https://localhost:8000/search",
            headers=headers,
            ssl=False
        ) as response:
            assert "Access-Control-Allow-Origin" not in response.headers

@pytest.mark.asyncio
async def test_session_security(setup_security_test):
    """Test session security."""
    logger.info("Testing session security")
    
    # Test session fixation
    old_session = "abc123"
    with pytest.raises(Exception):
        await searcher.search(
            "test query",
            session_id=old_session
        )
    
    # Test session timeout
    valid_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "test_user"
    })
    await searcher.search(
        "test query",
        auth_token=valid_token
    )
    
    # Wait for session timeout
    await asyncio.sleep(1800)  # 30 minutes
    
    with pytest.raises(Exception):
        await searcher.search(
            "test query",
            auth_token=valid_token
        )

@pytest.mark.asyncio
async def test_audit_logging(setup_security_test):
    """Test security audit logging."""
    logger.info("Testing audit logging")
    
    # Clear existing logs
    metrics.reset()
    
    # Generate security events
    valid_token = generate_jwt({
        "exp": int(time.time()) + 3600,
        "sub": "test_user"
    })
    
    # Test successful auth
    await searcher.search(
        "test query",
        auth_token=valid_token
    )
    
    # Test failed auth
    with pytest.raises(Exception):
        await searcher.search(
            "test query",
            auth_token="invalid"
        )
    
    # Check audit logs
    audit_logs = await metrics.get_audit_logs()
    assert len(audit_logs) >= 2
    assert any(log["event"] == "auth_success" for log in audit_logs)
    assert any(log["event"] == "auth_failure" for log in audit_logs)

@pytest.mark.asyncio
async def test_dependency_security(setup_security_test):
    """Test dependency security."""
    logger.info("Testing dependency security")
    
    # Check for known vulnerabilities
    result = subprocess.run(
        ["safety", "check"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "Known vulnerabilities found"
    
    # Check dependency licenses
    result = subprocess.run(
        ["liccheck"],
        capture_output=True,
        text=True
    )
    assert result.returncode == 0, "License compliance issues found"

@pytest.mark.asyncio
async def test_secrets_management(setup_security_test):
    """Test secrets management security."""
    logger.info("Testing secrets management")
    
    # Test environment variable protection
    assert "JWT_SECRET" not in os.environ
    assert "ENCRYPTION_KEY" not in os.environ
    
    # Test secure key generation
    key = secrets.token_bytes(32)
    assert len(key) >= 32
    
    # Test secure password hashing
    password = "test_password"
    salt = os.urandom(16)
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=32,
        salt=salt,
        iterations=100000,
    )
    key = base64.urlsafe_b64encode(kdf.derive(password.encode()))
    assert len(key) >= 32

@pytest.mark.asyncio
async def test_network_security(setup_security_test):
    """Test network security."""
    logger.info("Testing network security")
    
    # Test port scanning protection
    with pytest.raises(Exception):
        socket.create_connection(("localhost", 22))
    
    # Test firewall rules
    result = subprocess.run(
        ["sudo", "iptables", "-L"],
        capture_output=True,
        text=True
    )
    assert "DROP" in result.stdout
    
    # Test secure headers
    async with aiohttp.ClientSession() as session:
        async with session.get(
            "https://localhost:8000/search",
            ssl=False
        ) as response:
            headers = response.headers
            assert headers.get("X-Content-Type-Options") == "nosniff"
            assert headers.get("X-Frame-Options") == "DENY"
            assert headers.get("X-XSS-Protection") == "1; mode=block" 