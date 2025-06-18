"""Basic unit tests for the assistant functionality."""

import pytest
import json
import tempfile
import os
from unittest.mock import patch, Mock
from assistant import Assistant

class TestSimpleAssistant:
    """Test cases for SimpleAssistant class."""

    def test_initialization_with_api_key(self, mock_cohere_client):
        """Test assistant initialization with API key."""
        with patch.dict(os.environ, {'COHERE_API_KEY': str(os.getenv('COHERE_API_KEY'))}):
            assistant = Assistant()
            assert assistant.client is not None
            assert assistant.history == []

    def test_initialization_without_api_key(self):
        """Test assistant initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Set COHERE_API_KEY environment variable"):
                Assistant()

    def test_chat_basic_functionality(self, assistant):
        """Test basic chat functionality."""
        message = "Hello, how are you?"
        response = assistant.chat(message)

        assert isinstance(response, str)
        assert len(assistant.history) == 2  # User + Assistant messages
        assert assistant.history[0]["role"] == "USER"
        assert assistant.history[0]["message"] == message
        assert assistant.history[1]["role"] == "CHATBOT"

    def test_chat_history_management(self, assistant):
        """Test chat history is properly managed."""
        # Send multiple messages to test history management
        for i in range(10):
            assistant.chat(f"Test message {i}")

        # History should not exceed MAX_HISTORY * 2
        assert len(assistant.history) <= 12  # 6 conversations * 2 messages each

    def test_summarize_functionality(self, assistant):
        """Test text summarization."""
        text = "This is a long text that needs to be summarized for testing purposes."
        summary = assistant.summarize(text)

        assert isinstance(summary, str)
        assert len(summary) > 0

    def test_clear_history(self, assistant):
        """Test clearing conversation history."""
        assistant.chat("Test message")
        assert len(assistant.history) > 0

        result = assistant.clear_history()
        assert assistant.history == []
        assert result == "History cleared!"

    def test_save_chat_functionality(self, assistant):
        """Test saving chat to file."""
        assistant.chat("Test message for saving")

        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name

        try:
            result = assistant.save_chat(tmp_filename)
            assert "Saved to" in result

            # Verify file contents
            with open(tmp_filename, 'r') as f:
                saved_history = json.load(f)

            assert len(saved_history) == len(assistant.history)
            assert saved_history[0]["message"] == "Test message for saving"

        finally:
            os.unlink(tmp_filename)

    def test_error_handling_chat(self, assistant):
        """Test error handling in chat method."""
        # Mock the client to raise an exception
        assistant.client.chat.side_effect = Exception("API Error")

        response = assistant.chat("Test message")
        assert response.startswith("Error:")

    def test_error_handling_summarize(self, assistant):
        """Test error handling in summarize method."""
        assistant.client.summarize.side_effect = Exception("Summarize Error")

        response = assistant.summarize("Test text")
        assert response.startswith("Error:")
