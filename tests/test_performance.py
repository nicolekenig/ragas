"""Performance and load testing for the assistant."""

import time
from unittest.mock import patch

import psutil
import pytest
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed
from assistant import Assistant


class TestPerformance:
    """Performance tests for the assistant."""

    def test_response_time(self, assistant):
        """Test response time for single query."""
        start_time = time.time()
        response = assistant.chat("What is machine learning?")
        end_time = time.time()

        response_time = end_time - start_time

        # Response should be within reasonable time (< 10 seconds)
        assert response_time < 10.0
        assert isinstance(response, str)
        assert len(response) > 0

        print(f"Response time: {response_time:.2f} seconds")

    def test_memory_usage(self, assistant):
        """Test memory usage with multiple conversations."""
        import psutil


import os

process = psutil.Process(os.getpid())
initial_memory = process.memory_info().rss / 1024 / 1024  # MB

# Have multiple conversations
for i in range(20):
    assistant.chat(f"Test conversation {i}")

final_memory = process.memory_info().rss / 1024 / 1024  # MB
memory_increase = final_memory - initial_memory

# Memory increase should be reasonable (< 100MB)
assert memory_increase < 100
print(f"Memory increase: {memory_increase:.2f} MB")


def test_concurrent_requests(self, mock_cohere_client):
    """Test handling concurrent requests."""

    def create_assistant_and_chat(message):
        with patch.dict(os.environ, {'COHERE_API_KEY': 'test-key'}):
            assistant = Assistant()
            return assistant.chat(message)

    messages = [f"Concurrent test {i}" for i in range(5)]

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(create_assistant_and_chat, msg) for msg in messages]
        results = [future.result() for future in as_completed(futures)]

    end_time = time.time()
    total_time = end_time - start_time

    assert len(results) == 5
    assert all(isinstance(result, str) for result in results)
    assert total_time < 30  # Should complete within 30 seconds

    print(f"Concurrent requests completed in: {total_time:.2f} seconds")