"""Configuration for Cohere AI Assistant."""
import os

# API Settings
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
MODEL = 'command-r-plus'
MAX_TOKENS = 500
TEMPERATURE = 0.7

# Chat Settings
MAX_HISTORY = 6
