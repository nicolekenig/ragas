"""Cohere AI Assistant."""
import os

import cohere
import json
from datetime import datetime
from config import COHERE_API_KEY, MODEL, MAX_TOKENS, TEMPERATURE, MAX_HISTORY


class Assistant:
    def __init__(self, api_key=None):
        if api_key is None:
            api_key  = os.getenv('COHERE_API_KEY')

        if not api_key:
            raise ValueError("Set COHERE_API_KEY environment variable")

        self.api_key = api_key

        self.client = cohere.Client(self.api_key)
        self.history = []

    def chat(self, message):
        """Send message and get response."""
        try:
            # Add user message to history
            self.history.append({"role": "USER", "message": message})

            # Keep only recent history
            recent_history = self.history[-MAX_HISTORY:]

            # Get response from Cohere
            response = self.client.chat(
                model=MODEL,
                message=message,
                chat_history=recent_history[:-1],  # Exclude current message
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS
            )

            # Add assistant response to history
            self.history.append({"role": "CHATBOT", "message": response.text})

            return response.text

        except Exception as e:
            return f"Error: {str(e)}"

    def summarize(self, text):
        """Summarize text."""
        try:
            response = self.client.summarize(
                text=text,
                model=MODEL,
                length='medium'
            )
            return response.summary
        except Exception as e:
            return f"Error: {str(e)}"

    def clear_history(self):
        """Clear conversation history."""
        self.history = []
        return "History cleared!"

    def save_chat(self, filename=None):
        """Save conversation."""
        if not filename:
            filename = f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        try:
            with open(filename, 'w') as f:
                json.dump(self.history, f, indent=2)
            return f"Saved to {filename}"
        except Exception as e:
            return f"Error saving: {str(e)}"
