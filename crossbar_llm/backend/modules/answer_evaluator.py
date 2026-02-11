from tools.langchain_llm_qa_trial import RunPipeline
from tools.utils import Logger

class AnswerEvaluator:
    def __init__(self, model_name: str, temperature: float, max_tokens: int):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.pipeline = RunPipeline(model_name=model_name, verbose=False)

    def evaluate(self, question: str, expected: str, rationale: str, answer: str):
        Logger.info(f"Evaluating answer for question: {question}")
        raw_result = self.pipeline.run(question, expected, rationale, answer)
        return raw_result