"""RAGAS-based evaluation tests for the AI assistant."""

import pytest
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    faithfulness,
    context_recall,
    context_precision,
    answer_correctness,
    answer_similarity
)
from config import TEST_QUESTIONS, TEST_CONTEXTS, TEST_GROUND_TRUTHS


class TestRAGASEvaluation:
    """RAGAS evaluation tests for the assistant."""

    @pytest.fixture
    def evaluation_dataset(self, ai_assistant):
        """Create evaluation dataset with assistant responses."""
        responses = []
        for question in TEST_QUESTIONS:
            response = ai_assistant.chat(question)
            responses.append(response)

        # Create dataset for RAGAS evaluation
        data = {
            'question': TEST_QUESTIONS,
            'contexts': [[context] for context in TEST_CONTEXTS],  # RAGAS expects list of contexts
            'answer': responses,
            'ground_truth': TEST_GROUND_TRUTHS
        }

        return Dataset.from_dict(data)

    def test_answer_relevancy(self, evaluation_dataset):
        """Test answer relevancy using RAGAS."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[answer_relevancy]
        )

        # Answer relevancy should be reasonably high (> 0.5)
        assert result['answer_relevancy'] > 0.5
        print(f"Answer Relevancy Score: {result['answer_relevancy']:.3f}")

    def test_faithfulness(self, evaluation_dataset):
        """Test faithfulness of answers using RAGAS."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[faithfulness]
        )

        # Faithfulness should be high (> 0.7)
        assert result['faithfulness'] > 0.7
        print(f"Faithfulness Score: {result['faithfulness']:.3f}")

    def test_answer_correctness(self, evaluation_dataset):
        """Test answer correctness using RAGAS."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[answer_correctness]
        )

        # Answer correctness should be reasonable (> 0.6)
        assert result['answer_correctness'] > 0.6
        print(f"Answer Correctness Score: {result['answer_correctness']:.3f}")

    def test_answer_similarity(self, evaluation_dataset):
        """Test answer similarity using RAGAS."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[answer_similarity]
        )

        # Answer similarity should be reasonable (> 0.6)
        assert result['answer_similarity'] > 0.6
        print(f"Answer Similarity Score: {result['answer_similarity']:.3f}")

    def test_comprehensive_evaluation(self, evaluation_dataset):
        """Run comprehensive RAGAS evaluation."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                answer_correctness,
                answer_similarity
            ]
        )

        # Print all scores
        print("\n=== RAGAS Evaluation Results ===")
        for metric, score in result.items():
            print(f"{metric}: {score:.3f}")

        # Create summary report
        report = {
            'overall_score': sum(result.values()) / len(result),
            'metrics': result,
            'status': 'PASS' if sum(result.values()) / len(result) > 0.6 else 'FAIL'
        }

        assert report['overall_score'] > 0.6, f"Overall score too low: {report['overall_score']:.3f}"

        return report
