from evaluate import GeminiEvalModel, LLMTestCase, FaithfulnessMetric
from config import settings
from loguru import logger

def test_single_metric():
    print("Testing DeepEval + Gemini Integration...")
    try:
        eval_model = GeminiEvalModel(model_name=settings.llm_model)
        metric = FaithfulnessMetric(threshold=0.7, model=eval_model)
        
        test_case = LLMTestCase(
            input="What is the capital of France?",
            actual_output="The capital of France is Paris.",
            retrieval_context=["Paris is the capital and most populous city of France."]
        )
        
        metric.measure(test_case)
        print(f"Success! Faithfulness Score: {metric.score}")
        print(f"Reasoning: {metric.reason}")
    except Exception as e:
        print(f"Failed: {e}")
        logger.exception("Metric test failed")

if __name__ == "__main__":
    test_single_metric()
