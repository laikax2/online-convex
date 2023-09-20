import numpy as np


class ConvexMetrics:
    def __init__(self, gamma: float = 0.9, mu: float = 1, lambda_error: float = 0.05, store_metrics: bool = False):
        # For the first use, first combine and then update
        # By default 1 is low div (fast), 2 is high div (slow)
        # Convex Combination parameters
        self.gamma = gamma  # forgetting factor
        self.mu = mu  # step size
        self.lambda_error = lambda_error
        self.store_metrics = store_metrics

        # Initial values
        self.B = 0.5
        self.a = 0.0
        self.p = 0.5

        if self.store_metrics:
            # Store metrics
            self.B_list = []
            self.a_list = []
            self.p_list = []

    def combine(self, y1: float, y2: float):
        y = self.B * y1 + (1 - self.B) * y2
        return y

    def update(self, e: float, e1: float, e2: float):
        self.p = self.gamma * self.p + (1 - self.gamma) * np.power((e2 - e1), 2)
        # Get new a from updated p and e
        # Multiplying by 0.99 (or gamma) was added later to avoid a scaling to infinity
        self.a = 0.99 * self.a + (self.mu / self.p) * e * (e2 - e1) * self.lambda_error
        # Use new a to get new B which will be used for next prediction
        self.B = 1 / (1 + np.exp(-self.a))

        if self.store_metrics:
            # Store metrics
            self.B_list.append(self.B)
            self.a_list.append(self.a)
            self.p_list.append(self.p)


class ConvexCombination:
    def __init__(
        self,
        fast_learner,
        slow_learner,
        p_convex: dict = {},
        use_binary_error: bool = False,
        class_list: list = [0, 1],
    ):
        """
        Wrapper for two river learners (fast and slow) to use convex combination.
        Note there is no drift detection.

        fast_learner: fast learner
        slow_learner: slow learner
        p_convex = parameters for convex combination
        """
        # By default 1 is low div (fast), 2 is high div (slow)
        self.fast_learner = fast_learner
        self.slow_learner = slow_learner
        self.p_convex = p_convex
        self.use_binary_error = use_binary_error # Quizas actualizar cada 10 samples?
        self.class_list = class_list

        self.convex = ConvexMetrics(**self.p_convex)

    def predict_one(self, x: dict, include_individual: bool = False):
        if include_individual:
            y_pred_proba, y_fast_proba, y_slow_proba = self.predict_proba_one(x, include_individual=True)
            return (
                max(y_pred_proba, key=lambda k: y_pred_proba[k]),
                max(y_fast_proba, key=lambda k: y_fast_proba[k]),
                max(y_slow_proba, key=lambda k: y_slow_proba[k]),
            )
        else:
            y_pred_proba = self.predict_proba_one(x)
            return max(y_pred_proba, key=lambda k: y_pred_proba[k])

    def predict_proba_one(self, x: dict, include_individual: bool = False) -> dict:
        # Obtain the predictions
        y_fast_proba = self.fast_learner.predict_proba_one(x)
        y_slow_proba = self.slow_learner.predict_proba_one(x)
        # Obtain the weighted probability of every label
        y_pred_proba = {}
        for label in self.class_list:
            y_pred_proba[label] = self.convex.combine(y_fast_proba.get(label, 0), y_slow_proba.get(label, 0))

        if include_individual:
            y_fast_parsed = {}
            y_slow_parsed = {}
            for label in self.class_list:
                y_fast_parsed[label] = y_fast_proba.get(label, 0)
                y_slow_parsed[label] = y_slow_proba.get(label, 0)
            return y_pred_proba, y_fast_parsed, y_slow_parsed

        return y_pred_proba

    def learn_one(self, x: dict, y_true):
        # Update convex combination
        if self.use_binary_error:
            y_pred, y_fast, y_slow = self.predict_proba_one(x, include_individual=True)
            e = 0 if y_pred == y_true else 1
            e1 = 0 if y_fast == y_true else 1
            e2 = 0 if y_slow == y_true else 1
        else:
            y_pred, y_fast, y_slow = self.predict_proba_one(x, include_individual=True)
            e = 1 - y_pred.get(y_true, 0)
            e1 = 1 - y_fast.get(y_true, 0)
            e2 = 1 - y_slow.get(y_true, 0)
        self.convex.update(e, e1, e2)

        # Update ensembles
        self.fast_learner.learn_one(x, y_true)
        self.slow_learner.learn_one(x, y_true)
