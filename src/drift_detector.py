class DriftDetectorWrapper:
    def __init__(self, classifier, drift_detector, train_in_background: bool = True):
        """Wrapper for an OnlineBagging classifier with a drift detector (DDM or EDDM).

        classifier: classifier instance that implements the BaseClassifier interface and has a reset method. Usually OnlineBagging.
        drift_detector: drift detector instance that implements the DriftDetector interface.
        """
        self.classifier = classifier
        self.drift_detector = drift_detector
        self.train_in_background = train_in_background

        self.in_warning = False
        self.X_warning = []
        self.y_warning = []

    def predict_one(self, x: dict):
        return self.classifier.predict_one(x)

    def predict_proba_one(self, x: dict):
        return self.classifier.predict_proba_one(x)

    def learn_one(self, x: dict, y_true, y_pred=None):
        if y_pred is None:
            y_pred = self.predict_one(x)

        if y_true == y_pred:
            prediction = 0
        else:
            prediction = 1
        self.drift_detector.update(prediction)

        # Check if drift was detected
        detected = ""
        if self.drift_detector.drift_detected:
            self.reset()
            detected = "drift"

        if self.train_in_background:
            # If entering warning zone, trigger warning flag
            if self.drift_detector.warning_detected and not self.in_warning:
                self.in_warning = True
                detected = "warning"

            if self.in_warning:
                # Check if we have exited warning zone
                if not self.drift_detector.warning_detected:
                    self.X_warning = []
                    self.y_warning = []
                    detected = "exited warning"
                else:
                    self.X_warning.append(x)
                    self.y_warning.append(y_true)

        # Learn instance and return true if drift was detected
        self.classifier.learn_one(x, y_true)
        return detected

    def reset(self):
        self.in_warning = False
        self.classifier.reset()
        for i in range(len(self.X_warning)):
            self.classifier.learn_one(self.X_warning[i], self.y_warning[i])
        self.X_warning = []
        self.y_warning = []
