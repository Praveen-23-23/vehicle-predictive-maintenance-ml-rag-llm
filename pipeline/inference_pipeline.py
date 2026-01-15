from models.random_forest import load_and_predict
from rag.vector_store import build_retriever
from llm.decision_engine import decide_and_explain

def run_pipeline():
    # Load data and predictions
    data = load_and_predict("data/predictive_maintenance.csv")

    # Build retriever once
    retriever = build_retriever()

    # Sample a realistic operating point from historical data
    sample = data.sample(1).iloc[0]

    # Let LLM interpret risk (non-rule-based)
    risk, explanation = decide_and_explain(sample, retriever)

    return risk, explanation
