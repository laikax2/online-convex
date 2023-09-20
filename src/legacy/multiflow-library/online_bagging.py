import numpy as np

class OnlineBagging:
    def __init__(self, base_estimator, n_estimators=25, lambda_diversity=1, p_estimators=None, list_classes=[0, 1]):
        """ Ensemble with online bagging, it uses lambda_diversity to control its diveristy

        base_estimator: sklearn or skmultiflow estimator with partial_fit, predict and predict_proba methods
        n_estimators: number of estimators
        lambda_diversity: paremeter for forcing diversity when learning (higher means lower diversity)
        p_estimators: dictionary with parameters for the estimators initialization
        list_classes: list of classes to predict
        """
        self.base_estimator = base_estimator
        self.lambda_diversity = lambda_diversity
        self.n_estimators = n_estimators
        self.p_estimators = p_estimators
        if self.p_estimators is not None:
            self.list_classifiers = [self.base_estimator(**self.p_estimators) for _ in range(self.n_estimators)]
        else:
            self.list_classifiers = [self.base_estimator() for _ in range(self.n_estimators)]
        self.list_classes = np.array(list_classes)
    
    def reset(self):
        if self.p_estimators is not None:
            self.list_classifiers = [self.base_estimator(**self.p_estimators) for _ in range(self.n_estimators)]
        else:
            self.list_classifiers = [self.base_estimator() for _ in range(self.n_estimators)]

    def predict(self, X):
        """ Predict binary class for the data

        X: np.ndarray of shape (n_samples, n_features)
        return: np.ndarray with the predictions (0 or 1) of size n_samples
        """
        # predict for each classifier
        predictions = np.array([clf.predict(X).tolist() for clf in self.list_classifiers])
        # count the number of times the class is predicted for each sample
        votes_by_class = []
        for c in self.list_classes:
            votes_by_class.append(np.sum(predictions==c, axis=0))

        return self.list_classes[np.argmax(votes_by_class, axis=0)]

    def predict_proba(self, X):
        """ Compute the probability of belonging to each class

        X: np.ndarray of shape (n_samples, n_features)
        return: np.ndarray of shape (n_samples, 2) with the probabilites
        """
        # create array with the probabilities of every sample for each classifier
        array_probas = np.zeros((len(X), len(self.list_classes), self.n_estimators))
        for i, clf in enumerate(self.list_classifiers):
            array_probas[:, :, i] = clf.predict_proba(X)
        # return the mean of the probabilities computed by each classifier
        return array_probas.mean(axis=2)


    def update(self, X, y):
        """ Update the ensemble of models using bagging with lambda_diversity

        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of size n_samples containing the correct classes (0 or 1)
        """
        for i in range(self.n_estimators):
            k = np.random.poisson(self.lambda_diversity, len(X)) # generate k from poisson distribution for each sample
            X_training = None
            y_training = None

            # Add samples with k > 0 to the training set and reduce k by 1, repeat until every k is smaller or equal to 0
            while np.sum(k > 0):
                pos = np.where(k > 0)
                if X_training is None and y_training is None:
                    X_training = X[pos]
                    y_training = y[pos]
                else:
                    X_pos = X[pos]
                    y_pos = y[pos]
                    if X_pos.shape[0] == 1:
                        X_training = np.concatenate((X_training, X[pos].reshape((1, X[pos].shape[1]))), axis=0)
                    else:
                        X_training = np.concatenate((X_training, X[pos]), axis=0)
                    y_training = np.vstack((y_training.reshape((-1, 1)), y_pos.reshape((-1, 1))))
                k -= 1

            if X_training is not None and y_training is not None:
                y_training = y_training.reshape((y_training.shape[0],))
                self.list_classifiers[i].partial_fit(X_training, y_training, self.list_classes)

    def update_without_diversity(self, X, y):
        """ Alternative update where every model is updated with all of the samples

        X: np.ndarray of shape (n_samples, n_features)
        y: np.ndarray of size n_samples containing the correct classes (0 or 1)
        """
        for i in range(self.n_estimators):
            self.list_classifiers[i].partial_fit(X, y, self.list_classes)
