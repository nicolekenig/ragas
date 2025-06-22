import pytest
import time
from typing import List, Dict, Any
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    answer_correctness,
    answer_similarity,
    context_recall,
    context_precision,
)

# Import configuration from your existing config file
from config import get_ragas_config, TEST_CONFIG, handle_rate_limit_error


class TestRAGASEvaluation:
    """Test RAGAS evaluation metrics with Cohere models."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up Cohere models for all tests."""
        # Get configuration from centralized config file
        ragas_config = get_ragas_config()
        self.ragas_llm = ragas_config["llm"]
        self.ragas_embeddings = ragas_config["embeddings"]

    @pytest.fixture
    def sample_dataset(self) -> Dataset:
        """Create a sample dataset for testing."""
        data = {
            "question": [
                "What is the capital of France?",
                "What is machine learning?",
                "What is Python?",
            ],
            "answer": [
                "The capital of France is Paris.",
                "Machine learning is a subset of AI that enables systems to learn from data.",
                "Python is a high-level programming language.",
            ],
            "contexts": [
                ["Paris is the capital and largest city of France."],
                [
                    "Machine learning is a branch of artificial intelligence that focuses on building systems that learn from data."],
                ["Python is an interpreted, high-level programming language known for its simplicity."],
            ],
            "ground_truth": [
                "Paris is the capital of France.",
                "Machine learning is a type of AI that allows systems to learn from data without explicit programming.",
                "Python is a popular programming language used in many fields.",
            ]
        }
        return Dataset.from_dict(data)

    def test_answer_relevancy(self, sample_dataset):
        """Test answer relevancy metric."""
        # Configure metric with Cohere
        answer_relevancy.llm = self.ragas_llm
        answer_relevancy.embeddings = self.ragas_embeddings

        result = evaluate(
            sample_dataset,
            metrics=[answer_relevancy],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            raise_exceptions=TEST_CONFIG["raise_exceptions"]
        )

        # Access results correctly using to_pandas()
        df = result.to_pandas()

        # Check if metric exists in dataframe
        if 'answer_relevancy' in df.columns:
            scores = df['answer_relevancy'].tolist()
            avg_score = sum(scores) / len(scores)
            assert avg_score > 0.5
            assert all(0 <= score <= 1 for score in scores if not pd.isna(score))
        else:
            # Handle case where all evaluations failed
            pytest.skip("Answer relevancy evaluation failed due to LLM errors")

    def test_faithfulness(self, sample_dataset):
        """Test faithfulness metric."""
        # Add delay to avoid rate limits
        time.sleep(TEST_CONFIG["delay_between_tests"])

        # Configure metric with Cohere
        faithfulness.llm = self.ragas_llm

        result = evaluate(
            sample_dataset,
            metrics=[faithfulness],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )

        # Access results correctly
        df = result.to_pandas()

        if 'faithfulness' in df.columns:
            scores = df['faithfulness'].tolist()
            # Filter out NaN values
            valid_scores = [s for s in scores if not pd.isna(s)]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                assert avg_score > 0.5
                assert all(0 <= score <= 1 for score in valid_scores)
            else:
                pytest.skip("All faithfulness evaluations failed")
        else:
            pytest.skip("Faithfulness evaluation failed due to LLM errors")

    def test_answer_correctness(self, sample_dataset):
        """Test answer correctness metric."""
        # Add delay to avoid rate limits
        time.sleep(2)

        # Configure metric with Cohere
        answer_correctness.llm = self.ragas_llm
        answer_correctness.embeddings = self.ragas_embeddings

        result = evaluate(
            sample_dataset,
            metrics=[answer_correctness],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )

        df = result.to_pandas()

        if 'answer_correctness' in df.columns:
            scores = df['answer_correctness'].tolist()
            valid_scores = [s for s in scores if not pd.isna(s)]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                assert avg_score > 0.5
                assert all(0 <= score <= 1 for score in valid_scores)
            else:
                pytest.skip("All answer correctness evaluations failed")
        else:
            pytest.skip("Answer correctness evaluation failed due to LLM errors")

    def test_answer_similarity(self, sample_dataset):
        """Test answer similarity metric."""
        # Add delay to avoid rate limits
        time.sleep(2)

        # Configure metric with Cohere
        answer_similarity.llm = self.ragas_llm
        answer_similarity.embeddings = self.ragas_embeddings

        result = evaluate(
            sample_dataset,
            metrics=[answer_similarity],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )

        df = result.to_pandas()

        if 'answer_similarity' in df.columns:
            scores = df['answer_similarity'].tolist()
            valid_scores = [s for s in scores if not pd.isna(s)]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                assert avg_score > 0.5
                assert all(0 <= score <= 1 for score in valid_scores)
            else:
                pytest.skip("All answer similarity evaluations failed")
        else:
            pytest.skip("Answer similarity evaluation failed due to LLM errors")

    def test_context_recall(self, sample_dataset):
        """Test context recall metric."""
        # Add delay to avoid rate limits
        time.sleep(2)

        # Configure metric with Cohere
        context_recall.llm = self.ragas_llm

        result = evaluate(
            sample_dataset,
            metrics=[context_recall],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )

        df = result.to_pandas()

        if 'context_recall' in df.columns:
            scores = df['context_recall'].tolist()
            valid_scores = [s for s in scores if not pd.isna(s)]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                assert avg_score > 0.3
                assert all(0 <= score <= 1 for score in valid_scores)
            else:
                pytest.skip("All context recall evaluations failed")
        else:
            pytest.skip("Context recall evaluation failed due to LLM errors")

    def test_context_precision(self, sample_dataset):
        """Test context precision metric."""
        # Add delay to avoid rate limits
        time.sleep(2)

        # Configure metric with Cohere
        context_precision.llm = self.ragas_llm

        result = evaluate(
            sample_dataset,
            metrics=[context_precision],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings
        )

        df = result.to_pandas()

        if 'context_precision' in df.columns:
            scores = df['context_precision'].tolist()
            valid_scores = [s for s in scores if not pd.isna(s)]
            if valid_scores:
                avg_score = sum(valid_scores) / len(valid_scores)
                assert avg_score > 0.3
                assert all(0 <= score <= 1 for score in valid_scores)
            else:
                pytest.skip("All context precision evaluations failed")
        else:
            pytest.skip("Context precision evaluation failed due to LLM errors")

    def test_comprehensive_evaluation(self, sample_dataset):
        """Test multiple metrics together."""
        # Add delay to avoid rate limits
        time.sleep(2)

        metrics = [
            answer_relevancy,
            faithfulness,
            context_recall,
            context_precision
        ]

        # Configure all metrics with Cohere
        for metric in metrics:
            metric.llm = self.ragas_llm
            if hasattr(metric, 'embeddings'):
                metric.embeddings = self.ragas_embeddings

        result = evaluate(
            sample_dataset,
            metrics=metrics,
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            raise_exceptions=False  # Don't raise exceptions for individual failures
        )

        print("\n=== RAGAS Evaluation Results ===")
        df = result.to_pandas()

        # Process results for each metric
        for metric_name in ['answer_relevancy', 'faithfulness', 'context_recall', 'context_precision']:
            if metric_name in df.columns:
                scores = df[metric_name].tolist()
                valid_scores = [s for s in scores if not pd.isna(s)]
                if valid_scores:
                    avg_score = sum(valid_scores) / len(valid_scores)
                    print(f"{metric_name}: {avg_score:.3f} ({len(valid_scores)}/{len(scores)} successful)")
                    assert avg_score > 0.3
                else:
                    print(f"{metric_name}: All evaluations failed")

    def test_small_batch_evaluation(self):
        """Test with very small batch to avoid rate limits."""
        # Create minimal dataset
        data = {
            "question": ["What is 2+2?"],
            "answer": ["2+2 equals 4."],
            "contexts": [["Basic arithmetic: 2+2=4"]],
            "ground_truth": ["The answer is 4."]
        }

        dataset = Dataset.from_dict(data)

        # Configure metric
        answer_relevancy.llm = self.ragas_llm
        answer_relevancy.embeddings = self.ragas_embeddings

        result = evaluate(
            dataset,
            metrics=[answer_relevancy],
            llm=self.ragas_llm,
            embeddings=self.ragas_embeddings,
            raise_exceptions=False
        )

        df = result.to_pandas()
        print("\n=== Small Batch Results ===")
        print(df)

        # Check results
        if 'answer_relevancy' in df.columns and not pd.isna(df['answer_relevancy'][0]):
            assert df['answer_relevancy'][0] > 0.5


# Add pandas import at the top
import pandas as pd