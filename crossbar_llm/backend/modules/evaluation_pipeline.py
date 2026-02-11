from .test_dataset_loader import TestDatasetLoader
from .answer_evaluator import AnswerEvaluator
from .answer_scorer import score_answer

class EvaluationPipeline:
    def __init__(self, dataset_path: str, model_name: str, temperature: float, max_tokens: int):
        self.dataset_loader = TestDatasetLoader(dataset_path)
        self.answer_evaluator = AnswerEvaluator(model_name, temperature, max_tokens)

    def run(self):
        data = self.dataset_loader.get_data()
        results = []
        for item in data.get("questions", []):
            question = item.get("question")
            expected = item.get("expected")
            rationale = item.get("rationale")
            answer = item.get("answer")

            evaluation = self.answer_evaluator.evaluate(question, expected, rationale, answer)
            score = score_answer(evaluation, question, expected, rationale, answer)
            results.append({"question": question, "score": score})

        return results