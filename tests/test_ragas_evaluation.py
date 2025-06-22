"""RAGAS-based evaluation tests for the AI assistant using Cohere."""

import pytest
import os
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
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_cohere import ChatCohere
from langchain_cohere.embeddings import CohereEmbeddings
from config import TEST_QUESTIONS, TEST_CONTEXTS, TEST_GROUND_TRUTHS


class TestRAGASEvaluation:
    """RAGAS evaluation tests for the assistant using Cohere."""

    @pytest.fixture
    def cohere_llm(self):
        """Create Cohere LLM for RAGAS."""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            pytest.skip("COHERE_API_KEY not set")

        cohere_chat = ChatCohere(
            cohere_api_key=api_key,
            model="command-r-plus",
            temperature=0
        )
        return LangchainLLMWrapper(cohere_chat)

    @pytest.fixture
    def cohere_embeddings(self):
        """Create Cohere embeddings for RAGAS."""
        api_key = os.getenv("COHERE_API_KEY")
        if not api_key:
            pytest.skip("COHERE_API_KEY not set")

        cohere_emb = CohereEmbeddings(
            cohere_api_key=api_key,
            model="embed-english-v3.0"
        )
        return LangchainEmbeddingsWrapper(cohere_emb)

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

    def test_answer_relevancy(self, evaluation_dataset, cohere_llm):
        """Test answer relevancy using RAGAS with Cohere."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[answer_relevancy],
            llm=cohere_llm
        )

        # Answer relevancy should be reasonably high (> 0.5)
        assert result['answer_relevancy'] > 0.5
        print(f"Answer Relevancy Score: {result['answer_relevancy']:.3f}")

    def test_faithfulness(self, evaluation_dataset, cohere_llm):
        """Test faithfulness of answers using RAGAS with Cohere."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[faithfulness],
            llm=cohere_llm
        )

        # Faithfulness should be high (> 0.7)
        assert result['faithfulness'] > 0.7
        print(f"Faithfulness Score: {result['faithfulness']:.3f}")

    def test_answer_correctness(self, evaluation_dataset, cohere_llm):
        """Test answer correctness using RAGAS with Cohere."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[answer_correctness],
            llm=cohere_llm
        )

        # Answer correctness should be reasonable (> 0.6)
        assert result['answer_correctness'] > 0.6
        print(f"Answer Correctness Score: {result['answer_correctness']:.3f}")

    def test_answer_similarity(self, evaluation_dataset, cohere_embeddings):
        """Test answer similarity using RAGAS with Cohere embeddings."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[answer_similarity],
            embeddings=cohere_embeddings
        )

        # Answer similarity should be reasonable (> 0.6)
        assert result['answer_similarity'] > 0.6
        print(f"Answer Similarity Score: {result['answer_similarity']:.3f}")

    def test_context_recall(self, evaluation_dataset, cohere_llm):
        """Test context recall using RAGAS with Cohere."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[context_recall],
            llm=cohere_llm
        )

        # Context recall should be reasonable (> 0.5)
        assert result['context_recall'] > 0.5
        print(f"Context Recall Score: {result['context_recall']:.3f}")

    def test_context_precision(self, evaluation_dataset, cohere_llm):
        """Test context precision using RAGAS with Cohere."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[context_precision],
            llm=cohere_llm
        )

        # Context precision should be reasonable (> 0.5)
        assert result['context_precision'] > 0.5
        print(f"Context Precision Score: {result['context_precision']:.3f}")

    def test_comprehensive_evaluation(self, evaluation_dataset, cohere_llm, cohere_embeddings):
        """Run comprehensive RAGAS evaluation with Cohere."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                answer_correctness,
                answer_similarity
            ],
            llm=cohere_llm,
            embeddings=cohere_embeddings
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

    def test_full_comprehensive_evaluation(self, evaluation_dataset, cohere_llm, cohere_embeddings):
        """Run full comprehensive RAGAS evaluation including context metrics."""
        result = evaluate(
            dataset=evaluation_dataset,
            metrics=[
                answer_relevancy,
                faithfulness,
                context_recall,
                context_precision,
                answer_correctness,
                answer_similarity
            ],
            llm=cohere_llm,
            embeddings=cohere_embeddings
        )

        # Print all scores
        print("\n=== Full RAGAS Evaluation Results ===")
        for metric, score in result.items():
            print(f"{metric}: {score:.3f}")

        # Create detailed summary report
        report = {
            'overall_score': sum(result.values()) / len(result),
            'retrieval_metrics': {
                'context_recall': result.get('context_recall', 0),
                'context_precision': result.get('context_precision', 0)
            },
            'generation_metrics': {
                'answer_relevancy': result.get('answer_relevancy', 0),
                'faithfulness': result.get('faithfulness', 0),
                'answer_correctness': result.get('answer_correctness', 0),
                'answer_similarity': result.get('answer_similarity', 0)
            },
            'metrics': result,
            'status': 'PASS' if sum(result.values()) / len(result) > 0.6 else 'FAIL'
        }

        print(f"\nOverall Score: {report['overall_score']:.3f}")
        print(f"Status: {report['status']}")

        assert report['overall_score'] > 0.6, f"Overall score too low: {report['overall_score']:.3f}"

        return report