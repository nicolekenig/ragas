# Tests Cohere AI Assistant

An automation tests for AI assistant using pytest

## 🧪 Test Categories
### 1. Basic Unit Tests(`test_assistant_basic.py`)

- API key validation
- Chat functionality
- History management
- Summarization
- File saving/loading
- Error handling

### 2. RAGAS AI Evaluation (`test_ragas_evaluation.py`)

- Answer Relevancy: How relevant are responses to questions
- Faithfulness: How faithful answers are to context
- Answer Correctness: Factual accuracy of responses
- Answer Similarity: Semantic similarity to ground truth
- Comprehensive scoring with pass/fail thresholds

### 3. Performance Tests (`test_performance.py`)

- Response time measurement
- Memory usage monitoring
- Concurrent request handling
- Load testing

### 4. Integration Tests (`test_integration.py`)

- Complete conversation flows
- Error recovery
- System resilience

## 🚀 How to Run Tests:
### 1. Install Test Dependencies:
```bash
pip install -r requirements_test.txt
```
### 2. Set Your API Key:
```
export COHERE_API_KEY="your-actual-api-key"
```
### 3. Run All Tests:
```bash
python run_tests.py
```
### 4. Run Specific Test Categories:
```
# Basic functionality
pytest test_assistant_basic.py -v

# RAGAS evaluation (requires real API)
pytest test_ragas_evaluation.py -v -s

# Performance tests
pytest test_performance.py -v

# Integration tests
pytest test_integration.py -v
```

## 📊 RAGAS Evaluation Features:
The RAGAS tests evaluate your AI assistant on:
- Answer Relevancy: 0.5+ required
- Faithfulness: 0.7+ required
- Answer Correctness: 0.6+ required
- Answer Similarity: 0.6+ required
- Overall Score: 0.6+ for passing

## 🎯 Key Benefits:
1. Comprehensive Coverage: Tests all aspects of your assistant
2. AI-Specific Metrics: RAGAS provides LLM evaluation metrics
3. Performance Monitoring: Tracks response times and memory usage
4. Quality Assurance: Ensures consistent AI responses
5. Automated Reporting: Detailed test reports with scores

## 📈 Sample Test Output:
```
=== RAGAS Evaluation Results ===
answer_relevancy: 0.847
faithfulness: 0.923
answer_correctness: 0.756
answer_similarity: 0.812
Overall Score: 0.835 - PASS ✅
```