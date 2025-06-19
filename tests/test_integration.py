"""Integration tests for the complete assistant system."""

import pytest
import tempfile
import os
from unittest.mock import patch
from assistant import Assistant


class TestIntegration:
    """Integration tests for the complete system."""

    def test_complete_conversation_flow(self, ai_assistant):
        """Test a complete conversation flow."""
        # Start conversation
        response1 = ai_assistant.chat("Hello, can you help me understand AI?")
        assert isinstance(response1, str)
        assert len(response1) > 0

        # Follow-up question
        response2 = ai_assistant.chat("Can you give me more details about machine learning?")
        assert isinstance(response2, str)
        assert len(response2) > 0

        # Summarize a text
        text = "Machine learning is a method of data analysis that automates analytical model building."
        summary = ai_assistant.summarize(text)
        assert isinstance(summary, str)
        assert len(summary) > 0

        # Save conversation
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp_file:
            tmp_filename = tmp_file.name

        try:
            save_result = ai_assistant.save_chat(tmp_filename)
            assert "Saved to" in save_result

            # Verify saved file exists and has content
            assert os.path.exists(tmp_filename)
            assert os.path.getsize(tmp_filename) > 0

        finally:
            os.unlink(tmp_filename)

        # Clear history
        clear_result = ai_assistant.clear_history()
        assert clear_result == "History cleared!"
        assert len(ai_assistant.history) == 0

    def test_error_recovery(self, ai_assistant):
        """Test system recovery from errors."""
        # Test with invalid input
        response = ai_assistant.chat("")
        assert isinstance(response, str)

        # Test with very long input
        long_text = "test " * 1000
        response = ai_assistant.chat(long_text)
        assert isinstance(response, str)

        # Test summarization with empty text
        summary = ai_assistant.summarize("")
        assert isinstance(summary, str)