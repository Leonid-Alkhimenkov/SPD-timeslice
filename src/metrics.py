import gin

from torchmetrics import Metric
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    accuracy_score, f1_score, precision_score, recall_score,
)

class BaseScoreMetric(Metric):
    def __init__(self, score_func, **kwargs):
        super().__init__()
        self.cumulative_score = []
        self.score_func = score_func
        self.score_func_kwargs = kwargs

    def update(self, *args, **kwargs):
        self.cumulative_score.append(self.score_func(*args, **kwargs, **self.score_func_kwargs))

    def compute(self):
        mean_score = sum(self.cumulative_score) / len(self.cumulative_score)
        self.cumulative_score = []
        return mean_score

@gin.configurable
class SilhouetteScoreMetric(BaseScoreMetric):
    def __init__(self):
        super().__init__(silhouette_score)

@gin.configurable
class DaviesBouldinScoreMetric(BaseScoreMetric):
    def __init__(self):
        super().__init__(davies_bouldin_score)

@gin.configurable
class CalinskiHarabaszScoreMetric(BaseScoreMetric):
    def __init__(self):
        super().__init__(calinski_harabasz_score)

@gin.configurable
class AccuracyScoreMetric(BaseScoreMetric):
    def __init__(self):
        super().__init__(accuracy_score)

@gin.configurable
class F1ScoreMetric(BaseScoreMetric):
    def __init__(self, average='macro'):
        super().__init__(f1_score, average=average, zero_division=0)

@gin.configurable
class PrecisionScoreMetric(BaseScoreMetric):
    def __init__(self, average='macro'):
        super().__init__(precision_score, average=average, zero_division=0)

@gin.configurable
class RecallScoreMetric(BaseScoreMetric):
    def __init__(self, average='macro'):
        super().__init__(recall_score, average=average, zero_division=0)
