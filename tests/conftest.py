"""Pytest configuration and shared fixtures."""

import pytest
import os
from unittest.mock import patch, Mock
from assistant import Assistant


def pytest_configure(config):
    """Configure pytest."""
    # Set test environment variables
    os.environ['COHERE_API_KEY'] = str(os.getenv('COHERE_API_KEY'))

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment."""
    # Ensure test API key is set
    if not os.getenv('COHERE_API_KEY'):
        os.environ['COHERE_API_KEY'] = 'test-key-for-testing'


@pytest.fixture
def mock_cohere_client():
    """Mock Cohere client for testing."""
    with patch('cohere.Client') as mock_client:
        mock_instance = Mock()
        mock_client.return_value = mock_instance

        # Mock chat response
        mock_chat_response = Mock()
        mock_chat_response.text = "This is a test response from the AI assistant."
        mock_instance.chat.return_value = mock_chat_response

        # Mock summarize response
        mock_summary_response = Mock()
        mock_summary_response.summary = "This is a test summary."
        mock_instance.summarize.return_value = mock_summary_response

        yield mock_instance


@pytest.fixture
def ai_assistant(mock_cohere_client):
    """Create assistant instance with mocked client."""
    with patch.dict(os.environ, {'COHERE_API_KEY': 'test-key'}):
        return Assistant()
