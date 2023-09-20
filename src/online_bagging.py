import numpy as np


class OnlineBagging:
    def __init__(
        self,
        base_classifier_class,
        n_classifiers: int = 25,
        lambda_diversity: float = 1,
        p_classifiers: dict = {},
        class_list: list = [0, 1],
    ):
        """Ensemble with online bagging, it uses lambda_diversity to control its diveristy

        base_classifier_class: classifier class that implements the BaseClassifier interface
        n_classifiers: number of classifiers
        lambda_diversity: paremeter for forcing diversity when learning (higher means lower diversity)
        p_classifiers: dictionary with parameters for the classifiers initialization
        list_classes: list of classes to predict
        """
        self.base_classifier_class = base_classifier_class
        self.lambda_diversity = lambda_diversity
        self.n_classifiers = n_classifiers
        self.p_classifiers = p_classifiers
        self.classifier_list = [self.base_classifier_class(**self.p_classifiers) for _ in range(self.n_classifiers)]
        self.class_list = class_list

    def predict_one(self, x: dict):
        # Get all base classifiers predictions at once
        pred_arr = np.array([clf.predict_one(x) for clf in self.classifier_list])
        # Remove None predictions as they will crash numpy
        pred_arr = pred_arr[pred_arr != np.array(None)]
        # If all are None, return the first class
        if pred_arr.size == 0:
            return self.class_list[0]
        # Return most common value
        values, counts = np.unique(pred_arr, return_counts=True)
        return values[counts.argmax()]

    def predict_proba_one(self, x: dict):
        y_pred_proba = {}
        for clf in self.classifier_list:
            pred = clf.predict_proba_one(x)
            for key, value in pred.items():
                y_pred_proba[key] = y_pred_proba.get(key, 0) + value / self.n_classifiers
        return y_pred_proba

    def learn_one(self, x: dict, y):
        # For each base classifier draw k from the poisson distribution and learn k times
        for i in range(self.n_classifiers):
            k = np.random.poisson(self.lambda_diversity)
            for _ in range(k):
                self.classifier_list[i].learn_one(x, y)

    def reset(self):
        self.classifier_list = [self.base_classifier_class(**self.p_classifiers) for _ in range(self.n_classifiers)]
