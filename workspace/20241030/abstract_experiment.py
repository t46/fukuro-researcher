from abc import ABC, abstractmethod
from typing import TypeVar, List, Callable

Data = TypeVar('Data')    # データ型
Output = TypeVar('Output')  # 出力型
Score = TypeVar('Score')    # スコア型

class Algorithm(ABC):
    @abstractmethod
    def __call__(self, data: Data) -> Output:
        pass

class Metric(ABC):
    @abstractmethod
    def __call__(self, outputs: List[Output]) -> Score:
        pass

# Metric と comparison_func の組み合わせとしての Evaluation を定義する。つまり、Outputs から Score を計算し、その Score を比較することで、どのアルゴリズムが最も良いかを判断する。
class Evaluation:
    def __init__(self, metric: Metric, comparison_func: Callable[[List[Score]], bool]):
        self.metric = metric
        self.comparison_func = comparison_func

    def __call__(self, outputs: List[Output]) -> bool:
        scores = [self.metric(outputs) for outputs in outputs]
        experiment_result = self.comparison_func(scores)
        return experiment_result


class Experiment:
    def __init__(self, data: Data, algorithms: List[Algorithm], evaluation: Evaluation):
        self.data = data
        self.algorithms = algorithms
        self.evaluation = evaluation

    def __call__(self) -> bool:
        outputs = [algorithm(self.data) for algorithm in self.algorithms]
        experiment_result = self.evaluation(outputs)
        return experiment_result