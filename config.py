"""Configuration for Cohere AI Assistant."""
import os
"""Configuration settings for the Cohere AI Assistant project."""
import os
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv
from langchain_cohere import ChatCohere, CohereEmbeddings
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# API Settings
COHERE_API_KEY = os.getenv('COHERE_API_KEY')
MODEL = 'command-r-plus'
MAX_TOKENS = 500
TEMPERATURE = 0.7

# Chat Settings
MAX_HISTORY = 6

""" Test data for RAGAS evaluation """

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

# Load environment variables
load_dotenv()

# Base paths
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# API Configuration
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("COHERE_API_KEY environment variable is not set")

# Model Configuration
COHERE_MODEL = os.getenv("COHERE_MODEL", "command-r")
COHERE_EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-english-v3.0")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.1"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "8000"))

# Rate Limiting Configuration (for Cohere trial API)
COHERE_TRIAL_RATE_LIMIT = 40  # calls per minute
DELAY_BETWEEN_CALLS = 2  # seconds
REQUEST_TIMEOUT = 60  # seconds

# Logging Configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)


# RAGAS Configuration Functions
def get_cohere_llm():
    """Get configured Cohere LLM with rate limit handling."""
    return ChatCohere(
        model=COHERE_MODEL,
        temperature=TEMPERATURE,
        max_tokens=MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
        cohere_api_key=COHERE_API_KEY,
    )


def get_cohere_embeddings():
    """Get configured Cohere embeddings."""
    return CohereEmbeddings(
        model=COHERE_EMBED_MODEL,
        cohere_api_key=COHERE_API_KEY,
    )


def get_ragas_config() -> Dict[str, Any]:
    """Get RAGAS-compatible wrappers for Cohere models."""
    llm = get_cohere_llm()
    embeddings = get_cohere_embeddings()

    return {
        "llm": LangchainLLMWrapper(llm),
        "embeddings": LangchainEmbeddingsWrapper(embeddings)
    }


# Rate limit handling decorator
def handle_rate_limit_error(func):
    """Decorator to handle Cohere rate limit errors."""
    import time
    from functools import wraps

    @wraps(func)
    def wrapper(*args, **kwargs):
        max_retries = 3
        retry_delay = 5

        for attempt in range(max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "rate limit" in error_str.lower():
                    if attempt < max_retries - 1:
                        print(f"Rate limit hit, waiting {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                    else:
                        raise
                else:
                    raise

    return wrapper


# Test Configuration
TEST_CONFIG = {
    "batch_size": 3,  # Small batch size to avoid rate limits
    "delay_between_tests": DELAY_BETWEEN_CALLS,
    "raise_exceptions": False,  # Don't raise exceptions for individual failures
}


# Validation
def validate_config():
    """Validate configuration settings."""
    if not COHERE_API_KEY:
        raise ValueError("COHERE_API_KEY must be set")

    if MAX_TOKENS > 8000:
        print(f"Warning: MAX_TOKENS ({MAX_TOKENS}) exceeds Cohere's limit. Setting to 8000.")
        globals()['MAX_TOKENS'] = 8000

    return True


# Run validation on import
validate_config()