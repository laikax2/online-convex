from copy import deepcopy
import numpy as np

class PrequentialMetrics:
    def __init__(self):
        self.acc = 1
        self.var = 0
        self.std = 0
        self.t = 0
        self.t_drift = 0
    
    def update(self, y_pred, y_true, drift):
        number_of_time_steps = len(y_pred)  # number of time steps in the batch
        self.t += number_of_time_steps  # update the number of items seen
        good_predictions = np.sum(y_pred == y_true)
        batch_accuracy = good_predictions / number_of_time_steps

        if drift:
            self.acc = batch_accuracy
            self.var = self.acc * (1 - self.acc) / number_of_time_steps
            self.t_drift = self.t
        else:
            self.acc += (batch_accuracy - self.acc) / (self.t - self.t_drift + 1)
            self.var = self.acc * (1 - self.acc) / (self.t - self.t_drift + 1)
        self.std = np.sqrt(self.var)

class DDD:
    def __init__(self, drift_detector, ensemble_method, p_low_ensemble, p_high_ensemble, W=1, lambda_low_div=1):
        """ DDD algorithm based on article: L. L. Minku and X. Yao, "DDD: A New Ensemble Approach for Dealing with Concept Drift"

        drift_detector: drift detection class (DDM, EDDM...)
        ensemble_method: ensemble class, usually OnlineBagging
        W: multiplier constant for the weight of the old low diversity ensemble
        p_low_ensemble: dictionary with parameters for the low diversity ensembles
        p_high_ensemble: dictionary with parameters for the high diversity ensembles
        """
        self.drift_detector = drift_detector()
        self.ensemble_method = ensemble_method
        self.p_low_ensemble = p_low_ensemble
        self.p_high_ensemble = p_high_ensemble
        self.W = W
        self.lambda_low_div = lambda_low_div

        self.before_drift = True
        self.drift = False
        self.new_low_ensemble, self.new_high_ensemble = self._init_ensembles()
        self.old_low_ensemble = self.old_high_ensemble = None
        self.metrics_nl, self.metrics_nh, self.metrics_ol, self.metrics_oh = self._init_metrics()
        self.w_nl = self.w_ol = self.w_oh = 0
        self.y_pred = None

    def predict(self, X):
        # Before a drift is decected, only the low diversity ensemble is used for predictions
        if self.before_drift:
            y_pred = self.new_low_ensemble.predict(X)
        else:
            sum_acc = self.metrics_nl.acc + self.metrics_ol.acc * self.W + self.metrics_oh.acc
            self.w_nl = self.metrics_nl.acc / sum_acc
            self.w_ol = self.metrics_ol.acc * self.W / sum_acc
            self.w_oh = self.metrics_oh.acc / sum_acc
            y_pred = self._weighted_majority(X)
        self.y_pred = y_pred
        return y_pred

    def update(self, X, y_true):
        self.new_low_ensemble.update(X, y_true)
        self.new_high_ensemble.update(X, y_true)
        if not self.before_drift:
            self.old_low_ensemble.update(X, y_true)
            self.new_high_ensemble.update(X, y_true)

    def drift_detection(self, X, y_true, y_pred):
        self._update_metrics(X, y_true)
        for i in range(len(X)):
            if y_true[i] == y_pred[i]:
                prediction = 0
            else:
                prediction = 1
            self.drift_detector.add_element(prediction)

            if self.drift_detector.detected_change():
                self.drift = True
                
                if self.before_drift or (not self.before_drift and self.metrics_nl.acc > self.metrics_oh.acc):
                    self.old_low_ensemble = self.new_low_ensemble
                else:
                    self.old_low_ensemble = self.old_high_ensemble
                self.old_high_ensemble = self.new_high_ensemble
                self.old_high_ensemble.lambda_diversity = self.lambda_low_div
                self.new_low_ensemble, self.new_high_ensemble = self._init_ensembles()
                self.metrics_nl, self.metrics_nh, self.metrics_ol, self.metrics_oh = self._init_metrics()
                self.before_drift = False
                self._update_drift_mode()
                return True
        self.drift = False
        self._update_drift_mode()
        return False

    def _init_ensembles(self):
        new_low_ensemble = self.ensemble_method(**self.p_low_ensemble)
        new_high_ensemble = self.ensemble_method(**self.p_high_ensemble)
        return new_low_ensemble, new_high_ensemble

    @staticmethod
    def _init_metrics():
        metrics_nl = PrequentialMetrics()
        metrics_nh = PrequentialMetrics()
        metrics_ol = PrequentialMetrics()
        metrics_oh = PrequentialMetrics()
        return metrics_nl, metrics_nh, metrics_ol, metrics_oh

    def _weighted_majority(self, X):
        y_nl = self.new_low_ensemble.predict_proba(X)
        y_ol = self.old_low_ensemble.predict_proba(X)
        y_oh = self.old_high_ensemble.predict_proba(X)
        scores = self.w_nl * y_nl + self.w_ol * y_ol + self.w_oh * y_oh
        return self._scores_to_single_label(scores)

    @staticmethod
    def _scores_to_single_label(scores):
        if len(scores.shape) == 1:
            return (scores > 0).astype(np.int)
        else:
            return scores.argmax(axis=1)
    
    def _update_metrics(self, X, y_true):
        self.metrics_nl.update(self.new_low_ensemble.predict(X), y_true, self.drift)
        self.metrics_nh.update(self.new_high_ensemble.predict(X), y_true, self.drift)
        if not self.before_drift:
            self.metrics_ol.update(self.old_low_ensemble.predict(X), y_true, self.drift)
            self.metrics_oh.update(self.old_high_ensemble.predict(X), y_true, self.drift)
    
    def _update_drift_mode(self):
        if not self.before_drift:
            if self.metrics_nl.acc > self.metrics_oh.acc and self.metrics_nl.acc > self.metrics_ol.acc:
                self.before_drift = True
            elif self.metrics_oh.acc - self.metrics_oh.std > self.metrics_nl.acc + self.metrics_nl.std \
                    and self.metrics_oh.acc - self.metrics_oh.std > self.metrics_ol.acc + self.metrics_ol.std:
                self.new_low_ensemble = deepcopy(self.old_high_ensemble)
                self.metric_nl = deepcopy(self.metrics_oh)
                self.before_drift = True
