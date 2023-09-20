import numpy as np

class ConvexCombination:
    def __init__(self, gamma=0.9, mu=1, lambda_error=0.05, B0=0.5, p0=0.5, a0=0):
        # For the first use, first combine and then update
        # By default 1 is low div, 2 is high div
        # Convex Combination parameters
        self.gamma = gamma # forgetting factor
        self.mu = mu # step size
        self.lambda_error = 0.05
        # Initial values
        self.B = [B0]
        self.p = [p0, p0]
        self.a = a0
    
    def combine(self, y1, y2):
        y = self.B[-1] * y1 + (1 - self.B[-1]) * y2
        return y
    
    def update(self, y1, y2, y_pred, y_true):
        e = np.mean(np.absolute(y_true - y_pred))
        e1 = np.mean(np.absolute(y_true - y1))
        e2 = np.mean(np.absolute(y_true - y2))
        self.p[1] = self.gamma * self.p[0] + (1-self.gamma) * np.power((e2 - e1), 2)
        # Get new a from updated p and e
        self.a = self.a + (self.mu / self.p[1]) * e * (e2 - e1) * self.lambda_error * (1 - self.lambda_error)
        # Use new a to get new B which will be used for next prediction
        sigm = 1 / (1 + np.exp(-self.a))
        self.B.append(sigm)
        self.p[0] = self.p[1]

class SimpleConvexWrapper:
    def __init__(self, p_convex, ensemble_method, p_low_ensemble, p_high_ensemble):
        """
        Wrapper for two ensembles (high and low div) to use convex combination.
        Note there is no drift detection.

        ensemble_method: ensemble class, usually OnlineBagging
        p_low_ensemble: dictionary with parameters for the low diversity ensembles
        p_high_ensemble: dictionary with parameters for the high diversity ensembles
        """
        self.p_convex = p_convex
        self.ensemble_method = ensemble_method
        self.p_low_ensemble = p_low_ensemble
        self.p_high_ensemble = p_high_ensemble
        self.low_ensemble, self.high_ensemble = self._init_ensembles()
        self.convex = ConvexCombination(**self.p_convex)

    def predict(self, X):
        y_pred = self.predict_proba(X).round()
        return y_pred

    def update(self, X, y_true):
        # Update ensembles
        self.low_ensemble.update(X, y_true)
        self.high_ensemble.update(X, y_true)
        # Update convex combination
        y1, y2 = self._predict_each_proba(X)
        y_pred = self.predict_proba(X)
        self.convex.update(y1, y2, y_pred, y_true)

    def _init_ensembles(self):
        low_ensemble = self.ensemble_method(**self.p_low_ensemble)
        high_ensemble = self.ensemble_method(**self.p_high_ensemble)
        return low_ensemble, high_ensemble

    def predict_proba(self,X):
        y_low_proba, y_high_proba = self._predict_each_proba(X)
        y_proba = self.convex.combine(y_low_proba, y_high_proba)
        return y_proba

    def _predict_each_proba(self, X):
        # Get the probability class is 1, meaning 0.2 means prediction is 0 and 0.8 means prediction is 1
        y_low_proba = self.low_ensemble.predict_proba(X)[:, 1]
        y_high_proba = self.high_ensemble.predict_proba(X)[:, 1]
        return y_low_proba, y_high_proba
