import numpy as np


class ClassificationMetrics:
    def __init__(self, y_true, x, model):
        y_pred = model.predict(x)
        self.y_true = y_true
        self.y_pred = y_pred

    def get_all_metrics(self):
        from sklearn.metrics import accuracy_score
        score = accuracy_score(self.y_true, self.y_pred)
        return score