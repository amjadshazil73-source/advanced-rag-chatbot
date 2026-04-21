import json
import random
from typing import List, Dict, Any
from google import genai
from loguru import logger
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric, ContextualRecallMetric, HallucinationMetric
from deepeval.test_case import LLMTestCase
from deepeval.models.base_model import DeepEvalBaseLLM
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)
from google.genai import errors
from config import settings
from rag_chain import RAGChain

# 1. Custom Gemini Wrapper for DeepEval
class GeminiEvalModel(DeepEvalBaseLLM):
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.client = genai.Client(api_key=settings.google_api_key)

    def load_model(self):
        return self.client

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=5, max=60),
        retry=retry_if_exception_type((errors.ClientError, errors.ServerError)),
        reraise=True
    )
    def generate(self, prompt: str) -> str:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
        )
        return response.text

    async def a_generate(self, prompt: str) -> str:
        return self.generate(prompt)

    def get_model_name(self):
        return self.model_name

# 2. Main Evaluation Script
@retry(
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=5, max=60),
    retry=retry_if_exception_type((errors.ClientError, errors.ServerError)),
    reraise=True
)
def generate_synthetic_test_set(rag: RAGChain, n: int = 10) -> List[Dict[str, str]]:
    """Generates a test set by picking random chunks and asking Gemini to create a QA pair."""
    logger.info(f"Generating synthetic test set of {n} questions...")
    test_set = []
    
    # Pick n random chunks from the loaded documents
    sample_chunks = random.sample(rag.all_docs, min(n, len(rag.all_docs)))
    
    for i, chunk in enumerate(sample_chunks):
        prompt = f"""Based on the following document excerpt, generate:
1. A clear question that can be answered using ONLY this excerpt.
2. The correct, factual answer (Ground Truth).

Excerpt:
{chunk.page_content}

Format your response as:
Question: <question>
Answer: <answer>"""
        
        response = rag.gemini.models.generate_content(
            model=settings.llm_model,
            contents=prompt
        )
        text = response.text
        lines = text.strip().split("\n")
        
        # Helper to extract content
        question = ""
        answer_parts = []
        for line in lines:
            if line.lower().startswith("question:"):
                question = line.replace("Question:", "").replace("question:", "").strip()
            elif line.lower().startswith("answer:"):
                answer_parts.append(line.replace("Answer:", "").replace("answer:", "").strip())
            elif answer_parts:
                answer_parts.append(line.strip())
        
        answer = " ".join(answer_parts).strip()
        
        if question and answer:
            test_set.append({
                "question": question,
                "ground_truth": answer,
                "reference_context": chunk.page_content
            })
            logger.info(f"Generated test case {i+1}/{n}")
        else:
            logger.warning(f"Could not parse QA pair for chunk {i+1}")

    if not test_set:
        raise ValueError("Could not generate any test cases. This usually means the API is hitting a limit or returning empty responses.")
            
    return test_set

def run_evaluation():
    import os
    import time
    # Initialize components
    rag = RAGChain()
    eval_model = GeminiEvalModel(model_name=settings.llm_model)
    
    # Metrics definition (Sequential mode)
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=eval_model),
        AnswerRelevancyMetric(threshold=0.7, model=eval_model),
        ContextualRecallMetric(threshold=0.7, model=eval_model),
        HallucinationMetric(threshold=0.7, model=eval_model)
    ]
    
    # 1. Test Set Management
    test_set_path = "test_set.json"
    n_cases = 5

    if os.path.exists(test_set_path):
        logger.info(f"Loading existing test set from {test_set_path}...")
        with open(test_set_path, "r") as f:
            test_set = json.load(f)
    else:
        test_set = generate_synthetic_test_set(rag, n=n_cases)
        with open(test_set_path, "w") as f:
            json.dump(test_set, f, indent=4)
        logger.info(f"Saved synthetic test set to {test_set_path}")

    # 2. Results Management (Resuming support)
    results_path = "eval_results.json"
    if os.path.exists(results_path):
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
            logger.info(f"Found {len(results)} existing results. Resuming evaluation...")
        except Exception:
            results = []
    else:
        results = []

    completed_questions = {res["test_case"]["question"] for res in results}

    # 3. Evaluation Loop
    print(f"\nStarting Evaluation Loop (Survivor Mode, {len(test_set)} total cases)...")
    for i, test_case in enumerate(test_set):
        question = test_case["question"]
        
        if question in completed_questions:
            logger.info(f"Skipping Case {i+1}/{len(test_set)} (Already evaluated)")
            continue
        
        # Aggressive Cooldown to satisfy 60s/minute limits
        if len(results) > 0:
            logger.info("Waiting 65 seconds for API quota to reset...")
            time.sleep(65)

        logger.info(f"Evaluating Case {i+1}/{len(test_set)}: {question[:50]}...")
        
        try:
            # Run RAG Pipeline
            rag_output = rag.ask(question, skip_expansion=True)
            
            # Create DeepEval Test Case
            actual_test_case = LLMTestCase(
                input=question,
                actual_output=rag_output["answer"],
                expected_output=test_case["ground_truth"],
                retrieval_context=[doc.page_content for doc in rag_output["source_chunks"]]
            )
            
            # Score individually with built-in retry for each metric
            scores = {}
            for metric in metrics:
                @retry(
                    stop=stop_after_attempt(3),
                    wait=wait_exponential(multiplier=2, min=10, max=60),
                    retry=retry_if_exception_type((errors.ClientError, errors.ServerError)),
                    reraise=True
                )
                def measure_with_retry():
                    metric.measure(actual_test_case)
                
                try:
                    measure_with_retry()
                    scores[metric.__class__.__name__] = metric.score
                    logger.debug(f"Metric {metric.__class__.__name__}: {metric.score}")
                except Exception as me:
                    logger.warning(f"Metric {metric.__class__.__name__} failed after retries: {me}")
                    scores[metric.__class__.__name__] = 0.0 # Fail gracefully for this metric
                
                time.sleep(5) # Brief pause between metrics
            
            results.append({
                "test_case": test_case,
                "actual_output": rag_output["answer"],
                "scores": scores,
                "model_used": rag_output["model_used"]
            })

            # SAVE INCREMENTALLY
            with open(results_path, "w") as f:
                json.dump(results, f, indent=4)
            logger.info(f"Saved result {len(results)} to {results_path}")

        except Exception as e:
            logger.error(f"Error evaluating test case {i+1}: {e}")
            if "RESOURCE_EXHAUSTED" in str(e):
                logger.warning("Quota hit! Stopping for now. Run evaluate.py again later to resume.")
                break

    print("\nEvaluation Session Ended. Run 'python eval_report.py' for current progress.")

if __name__ == "__main__":
    run_evaluation()
