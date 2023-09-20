from copy import deepcopy

import numpy as np

from src.online_bagging import OnlineBagging


class PrequentialMetrics:
    def __init__(self):
        self.acc = 1
        self.var = 0
        self.std = 0
        self.t = 0
        self.t_drift = 0

    def update(self, y_pred, y_true, drift=False):
        if y_true == y_pred:
            prediction = 0
        else:
            prediction = 1

        self.t += 1
        if drift:
            self.acc = prediction
            self.var = self.acc * (1 - self.acc)
            self.t_drift = self.t
        else:
            self.acc += (prediction - self.acc) / (self.t - self.t_drift + 1)
            self.var = self.acc * (1 - self.acc) / (self.t - self.t_drift + 1)
        self.std = np.sqrt(self.var)


class DDD:
    def __init__(
        self, drift_detector, p_low_ensemble: dict, p_high_ensemble: dict, W: float = 1, class_list: list = [0, 1]
    ):
        """DDD algorithm based on article: L. L. Minku and X. Yao, "DDD: A New Ensemble Approach for Dealing with Concept Drift"

        drift_detector: drift detection class (DDM, EDDM...)
        ensemble_method: ensemble class, usually OnlineBagging
        W: multiplier constant for the weight of the old low diversity ensemble
        p_low_ensemble: dictionary with parameters for the low diversity ensembles
        p_high_ensemble: dictionary with parameters for the high diversity ensembles
        """
        self.drift_detector = drift_detector
        self.p_low_ensemble = p_low_ensemble
        self.p_high_ensemble = p_high_ensemble
        self.W = W
        self.class_list = class_list

        self.ensemble_method = OnlineBagging
        self.lambda_low_div = p_low_ensemble["lambda_diversity"]

        self.before_drift = True
        self.drift = False
        self.new_low_ensemble, self.new_high_ensemble = self._init_ensembles()
        self.old_low_ensemble = self.old_high_ensemble = None
        self.metrics_nl, self.metrics_nh, self.metrics_ol, self.metrics_oh = self._init_metrics()
        self.w_nl = self.w_ol = self.w_oh = 0
        self.y_pred = None

    def predict_one(self, x: dict):
        y_pred_proba = self.predict_proba_one(x)
        if len(y_pred_proba) == 0:
            return self.class_list[0]
        return max(y_pred_proba, key=lambda k: y_pred_proba[k])

    def predict_proba_one(self, x: dict) -> dict:
        if self.before_drift:
            y_pred_proba = self.new_low_ensemble.predict_proba_one(x)
        else:
            # Get the weights for every ensemble
            sum_acc = self.metrics_nl.acc + self.metrics_ol.acc * self.W + self.metrics_oh.acc
            if sum_acc < 0.0001:
                # Avoid division by 0 error later
                self.w_nl = 1 / 3
                self.w_ol = 1 / 3
                self.w_oh = 1 / 3
                sum_acc = 1
            else:
                self.w_nl = self.metrics_nl.acc / sum_acc
                self.w_ol = self.metrics_ol.acc * self.W / sum_acc
                self.w_oh = self.metrics_oh.acc / sum_acc
            # Obtain the predictions
            y_nl = self.new_low_ensemble.predict_proba_one(x)
            y_ol = self.old_low_ensemble.predict_proba_one(x)
            y_oh = self.old_high_ensemble.predict_proba_one(x)
            # Obtain the weighted probability of every label
            y_pred_proba = {}
            for label in self.class_list:
                y_pred_proba[label] = (
                    self.w_nl * y_nl.get(label, 0) + self.w_ol * y_ol.get(label, 0) + self.w_oh * y_oh.get(label, 0)
                )
        return y_pred_proba

    def learn_one(self, x: dict, y_true, y_pred=None):
        if y_pred is None:
            y_pred = self.predict_one(x)
        detected = self.drift_detection_one(x, y_true, y_pred)
        self.new_low_ensemble.learn_one(x, y_true)
        self.new_high_ensemble.learn_one(x, y_true)
        if not self.before_drift:
            self.old_low_ensemble.learn_one(x, y_true)
            self.old_high_ensemble.learn_one(x, y_true)
        return detected

    def drift_detection_one(self, x, y_true, y_pred):
        # Update metrics
        self.metrics_nl.update(self.new_low_ensemble.predict_one(x), y_true, self.drift)
        self.metrics_nh.update(self.new_high_ensemble.predict_one(x), y_true, self.drift)
        if not self.before_drift:
            self.metrics_ol.update(self.old_low_ensemble.predict_one(x), y_true, self.drift)
            self.metrics_oh.update(self.old_high_ensemble.predict_one(x), y_true, self.drift)

        if y_true == y_pred:
            prediction = 0
        else:
            prediction = 1
        self.drift_detector.update(prediction)

        # Run algorithm if concept drift is detected
        if self.drift_detector.drift_detected:
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
            return "drift"

        self.drift = False
        self._update_drift_mode()

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

    def _update_drift_mode(self):
        if not self.before_drift:
            if self.metrics_nl.acc > self.metrics_oh.acc and self.metrics_nl.acc > self.metrics_ol.acc:
                print("EXITED DRIFT WITH NEW ENSEMBLES")
                self.before_drift = True
            elif (
                self.metrics_oh.acc - self.metrics_oh.std > self.metrics_nl.acc + self.metrics_nl.std
                and self.metrics_oh.acc - self.metrics_oh.std > self.metrics_ol.acc + self.metrics_ol.std
            ):
                print("EXITED DRIFT WITH OLD ENSEMBLES")
                self.new_low_ensemble = deepcopy(self.old_high_ensemble)
                self.metric_nl = deepcopy(self.metrics_oh)
                self.before_drift = True
