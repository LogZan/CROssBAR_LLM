from tools.utils import Logger

def score_answer(judge, question: str, expected: str, rationale: str, answer: str):
    Logger.info("Scoring answer using judge model")
    if not answer or answer.strip().lower() in {"n/a", "na"}:
        return {"pass": False, "reason": "Empty or N/A answer"}

    result = judge(question, expected, rationale, answer)
    return {
        "pass": result.get("pass", False),
        "reason": result.get("reason", "Unknown reason"),
        "rationale_match": result.get("rationale_match", False),
        "raw": result.get("raw", "")
    }