from ..evaluation.evaluators import RerankingEvaluator
from .AbsTask import AbsTask


class AbsTaskReranking(AbsTask):
    """
    Abstract class for re-ranking experiments.
    Child-classes must implement the following properties:
    self.corpus = {'dev': Dict[id, str], 'test': Dict[id, str]}         #id => sentence
    self.queries = {'dev': Dict[id, str], 'test': Dict[id, str]}
    self.relevant_docs = {'dev': Dict[id, set], 'test': Dict[id, set]}
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def evaluate(self, model, split="test", **kwargs):
        if not self.data_loaded:
            self.load_data()

        data_split = self.dataset[split]

        evaluator = RerankingEvaluator(data_split, **kwargs)
        scores = evaluator(model)

        return dict(scores)
