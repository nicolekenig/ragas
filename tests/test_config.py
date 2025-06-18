"""Test configuration and fixtures."""

import os
import pytest
from unittest.mock import Mock, patch
from assistant import Assistant

# Test data for RAGAS evaluation
TEST_QUESTIONS = [
    "What is artificial intelligence?",
    "How does machine learning work?",
    "Explain natural language processing",
    "What are the benefits of AI?",
    "How do neural networks function?"
]

TEST_CONTEXTS = [
    "Artificial intelligence (AI) is a branch of computer science that aims to create intelligent machines capable of performing tasks that typically require human intelligence.",
    "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed for every task.",
    "Natural language processing (NLP) is a field of AI that focuses on the interaction between computers and humans using natural language.",
    "AI benefits include automation of repetitive tasks, improved decision-making, enhanced productivity, and solving complex problems faster.",
    "Neural networks are computing systems inspired by biological neural networks, consisting of interconnected nodes that process information."
]

TEST_GROUND_TRUTHS = [
    "AI is computer science focused on creating intelligent machines that can perform human-like tasks.",
    "Machine learning allows computers to learn from data without explicit programming.",
    "NLP enables computers to understand and process human language.",
    "AI provides automation, better decisions, increased productivity, and faster problem-solving.",
    "Neural networks are AI systems modeled after brain networks with interconnected processing nodes."
]


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
def assistant(mock_cohere_client):
    """Create assistant instance with mocked client."""
    with patch.dict(os.environ, {'COHERE_API_KEY': 'test-key'}):
        return Assistant()
