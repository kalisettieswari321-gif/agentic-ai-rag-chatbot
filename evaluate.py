
import json
import time

from src import config
from src.graph import build_rag_graph

# "in"  = answerable from the eBook
# "out" = should be refused
DATASET = [
    # In-scope (first 5 come from tests_sample_queries.py)
    ("in", "What is Agentic AI?"),
    ("in", "What are the key components of Agentic AI?"),
    ("in", "What are the types and categories of agents?"),
    ("in", "What is the role of planning in agentic systems?"),
    ("in", "What are some applications of Agentic AI?"),
    ("in", "How does Agentic AI differ from traditional AI?"),
    ("in", "What is the role of memory in agentic systems?"),
    ("in", "What are the benefits of Agentic AI?"),
    ("in", "What are the challenges or risks of Agentic AI?"),
    ("in", "How do LLMs differ from AI agents?"),
    ("in", "What is multi-agent collaboration?"),
    ("in", "What is the role of reasoning in AI agents?"),
    ("in", "How do AI agents make decisions?"),
    ("in", "What are the future trends of Agentic AI?"),
    ("in", "How can organizations adopt Agentic AI?"),
    # Out-of-scope, easy (unrelated topics)
    ("out", "What is the capital of France?"),
    ("out", "Who won the FIFA World Cup in 2018?"),
    ("out", "Give me a recipe for chicken biryani."),
    ("out", "Who is the current president of the United States?"),
    ("out", "What is the boiling point of water?"),
    ("out", "Write a Python function to reverse a list."),
    ("out", "What is the tallest mountain in the world?"),
    # Out-of-scope, hard (AI-related but probably not in the eBook)
    ("out", "How do I fine-tune GPT-4 on my own data?"),
    ("out", "What is the price of Pinecone's paid plan?"),
    ("out", "Explain the mathematics of the transformer attention mechanism."),
    ("out", "Who founded OpenAI?"),
    ("out", "Which GPU is best for training neural networks?"),
]


def invoke_with_backoff(graph, question, attempts=3, wait=60):
    """Run the graph; on a Groq rate limit (429), wait and retry."""
    for attempt in range(1, attempts + 1):
        try:
            return graph.invoke({"question": question})
        except Exception as exc:
            msg = str(exc).lower()
            is_rate_limit = "429" in msg or "rate limit" in msg
            if is_rate_limit and attempt < attempts:
                print(f"   (rate limit, waiting {wait}s, retry {attempt}/{attempts - 1})")
                time.sleep(wait)
                continue
            raise


def main():
    graph = build_rag_graph()
    rows = []

    print(f"{'LBL':<4}{'BEST':<8}{'RESULT':<10}{'CHECK':<7}QUESTION")
    print("-" * 80)

    for label, question in DATASET:
        try:
            result = invoke_with_backoff(graph, question)
            scores = result.get("retrieval_scores", [])
            best = round(max(scores), 3) if scores else 0.0
            refused = result["answer"].strip() == config.REFUSAL_MESSAGE
            correct = (label == "in" and not refused) or (label == "out" and refused)
            error = None
        except Exception as exc:
            best, refused, correct, error = None, None, None, str(exc)[:100]

        rows.append({
            "label": label,
            "question": question,
            "best_retrieval_score": best,
            "refused": refused,
            "correct": correct,
            "error": error,
        })

        if error:
            print(f"{label:<4}{'-':<8}{'ERROR':<10}{'-':<7}{question}  [{error}]")
        else:
            print(
                f"{label:<4}{best:<8.3f}"
                f"{'REFUSED' if refused else 'ANSWERED':<10}"
                f"{'OK' if correct else 'WRONG':<7}{question}"
            )
        time.sleep(8)

    valid = [r for r in rows if r["error"] is None]
    ins = [r for r in valid if r["label"] == "in"]
    outs = [r for r in valid if r["label"] == "out"]

    print("\n=== SUMMARY ===")
    print(f"Errors: {len(rows) - len(valid)}")
    print(f"Overall correct: {sum(r['correct'] for r in valid)}/{len(valid)}")
    print(f"In-scope answered: {sum(not r['refused'] for r in ins)}/{len(ins)}")
    print(f"Out-of-scope refused: {sum(r['refused'] for r in outs)}/{len(outs)}")
    print(f"Lowest in-scope best score:   {min(r['best_retrieval_score'] for r in ins):.3f}")
    print(f"Highest out-of-scope best score: {max(r['best_retrieval_score'] for r in outs):.3f}")

    print("\n=== THRESHOLD SWEEP (retrieval score only) ===")
    print(f"{'THRESHOLD':<11}{'IN-SCOPE KEPT':<16}{'OUT-OF-SCOPE REJECTED'}")
    for i in range(8, 19):
        t = i * 0.05
        kept = sum(r["best_retrieval_score"] >= t for r in ins)
        rejected = sum(r["best_retrieval_score"] < t for r in outs)
        print(f"{t:<11.2f}{f'{kept}/{len(ins)}':<16}{rejected}/{len(outs)}")

    with open("eval_results.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved eval_results.json")


if __name__ == "__main__":
    main()
