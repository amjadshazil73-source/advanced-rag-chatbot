import json
import os
from tabulate import tabulate
from loguru import logger

def format_score(score):
    """Format scores with color codes for threshold identification."""
    if score >= 0.8:
        return f"\033[92m{score:.2f}\033[0m" # Green
    elif score >= 0.6:
        return f"\033[93m{score:.2f}\033[0m" # Yellow
    else:
        return f"\033[91m{score:.2f}\033[0m" # Red

def generate_report():
    if not os.path.exists("eval_results.json"):
        print("\n[!] No evaluation results found. Please run 'python evaluate.py' first.")
        return

    with open("eval_results.json", "r") as f:
        results = json.load(f)

    if not results:
        print("\n[!] Results file is empty.")
        return

    # Prepare table data
    table_data = []
    avg_scores = {
        "FaithfulnessMetric": 0,
        "AnswerRelevancyMetric": 0,
        "ContextualRecallMetric": 0,
        "HallucinationMetric": 0
    }

    for i, res in enumerate(results):
        q = res["test_case"]["question"]
        if len(q) > 40: q = q[:37] + "..."
        
        scores = res["scores"]
        row = [i+1, q]
        
        for k in avg_scores.keys():
            val = scores.get(k, 0)
            avg_scores[k] += val
            row.append(format_score(val))
        
        table_data.append(row)

    # Calculate final averages
    headers = ["#", "Question", "Faithful", "Rel.", "Recall", "Halluc."]
    final_averages = []
    for k, v in avg_scores.items():
        avg = v / len(results)
        final_averages.append(format_score(avg))

    print("\n" + "="*80)
    print(" RAG EVALUATION REPORT: Phase 4 Advanced Metrics ")
    print("="*80 + "\n")
    
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    print("\n" + "="*80)
    print(f" FINAL AGGREGATE SCORES (Avg of {len(results)} cases) ")
    print("="*80)
    print(f" Faithfulness:    {final_averages[0]}  (Target: 0.8+)")
    print(f" Relevancy:       {final_averages[1]}  (Target: 0.8+)")
    print(f" Context Recall:  {final_averages[2]}  (Target: 0.7+)")
    print(f" Hallucination:   {final_averages[3]}  (Target: 0.0-0.2)")
    print("="*80 + "\n")

    print("\n--- WHAT THESE METRICS MEAN ---")
    print("\n1. Faithfulness: Ensures the answer is derived ONLY from the retrieved context.")
    print("   Low score? Your LLM is using outside knowledge (hallucinating).")
    print("\n2. Answer Relevancy: Ensures the answer actually addresses the question.")
    print("   Low score? The LLM might be rambling or giving generic answers.")
    print("\n3. Contextual Recall: Measures if retrieval found the correct page.")
    print("   Low score? Your embeddings, hybrid search, or reranker need tuning.")
    print("\n4. Hallucination: Detects contradictions between the answer and document.")
    print("   High score? The AI is making up specific facts (dangerous for production!).")

if __name__ == "__main__":
    generate_report()
