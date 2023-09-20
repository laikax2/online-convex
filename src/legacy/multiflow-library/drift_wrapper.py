import numpy as np

class DriftWrapper:
    def __init__(self, drift_detector, ensemble_method):
        """ Wrapper for online bagging with a drift detector (DDM or EDDM)
        """
        self.drift_detector = drift_detector
        self.ensemble_method = ensemble_method

        self.in_warning = False
        self.X_warning = None
        self.y_warning = None
        self.y_pred = None
    
    def _reset(self):
        self.in_warning = False
        self.ensemble_method.reset()
        if self.X_warning is not None and self.y_warning is not None:
            self.ensemble_method.update(self.X_warning, self.y_warning)
        self.X_warning = None
        self.y_warning = None

    def predict(self, X):
        y_pred = self.ensemble_method.predict(X)
        self.y_pred = y_pred
        return y_pred

    def drift_detection(self, X, y_true, y_pred):
        for i in range(len(X)):
            if y_true[i] == y_pred[i]:
                prediction = 0
            else:
                prediction = 1
            self.drift_detector.add_element(prediction)
            if self.drift_detector.detected_change():
                self._reset()
                return True
        if self.drift_detector.detected_warning_zone():
            if not self.in_warning:
                self.X_warning = X.copy()
                self.y_warning = y_true.copy()
                self.in_warning = True
            else:
                self.X_warning = np.concatenate((self.X_warning, X))
                self.y_warning = np.concatenate((self.y_warning, y_true))
        elif self.in_warning:
            self.in_warning = False
            self.X_warning = None
            self.y_warning = None
        return False

    def update(self, X, y_true):
        self.ensemble_method.update(X, y_true)
