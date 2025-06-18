"""Pytest configuration and shared fixtures."""

import pytest
import os
from unittest.mock import patch, Mock

def pytest_configure(config):
    """Configure pytest."""
    # Set test environment variables
    os.environ['COHERE_API_KEY'] = os.getenv('COHERE_API_KEY')

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure test API key is set
    if not os.getenv('COHERE_API_KEY'):
        os.environ['COHERE_API_KEY'] = 'test-key-for-testing'