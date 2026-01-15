from llm.llm_client import call_llm

_llm_cache = {}

def decide_and_explain(sample_row, retriever):
    # Soft cache key (NOT rule-based)
    prob_key = round(sample_row["Failure_Probability"], 2)

    if prob_key in _llm_cache:
        return _llm_cache[prob_key]

    docs = retriever.invoke("vehicle failure risk interpretation")
    context = docs[0].page_content

    prompt = f"""
You are an automotive maintenance expert.

Vehicle sensor data:
- Failure probability (from ML model): {sample_row['Failure_Probability']}
- Tool wear: {sample_row['Tool wear [min]']}
- Torque: {sample_row['Torque [Nm]']}
- Process temperature: {sample_row['Process temperature [K]']}

Maintenance knowledge:
{context}

Instructions:
- The failure probability is a statistical estimate, not a rule.
- Higher probability generally indicates higher risk, but context matters.
- Consider sensor values and maintenance knowledge together.

Task:
1. Decide the risk level: LOW, MEDIUM, or HIGH
2. Explain the reasoning in a professional maintenance report style

Respond in the following format ONLY:

RISK: <LOW | MEDIUM | HIGH>


EXPLANATION:

<detailed explanation paragraph>
"""

    response = call_llm(prompt)

    risk = "UNKNOWN"
    for line in response.splitlines():
        if line.upper().startswith("RISK"):
            risk = line.split(":")[-1].strip().upper()

    result = (risk, response)
    _llm_cache[prob_key] = result

    return result
