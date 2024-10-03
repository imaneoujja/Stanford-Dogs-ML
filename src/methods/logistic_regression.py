import numpy as np
from ..utils import label_to_onehot, accuracy_fn , get_n_classes , gradient_logistic_multi

class LogisticRegression(object):
    """
    Logistic regression classifier.
    """

    def __init__(self, lr, max_iters=500 , task_kind = "classification"):
        """
        Initialize the new object and set its arguments.

        Arguments:
            lr (float): learning rate of the gradient descent
            max_iters (int): maximum number of iterations
            regularization_strength (float): strength of L2 regularization
        """
        self.lr = lr
        self.max_iters = max_iters
        self.task_kind = task_kind


    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """
        D = training_data.shape[1]
        C = get_n_classes(training_labels)
        self.weights = np.random.normal(0, 0.01, (D, C))

        iters = 0
        while iters < self.max_iters:
            gradient = gradient_logistic_multi(self, training_data, label_to_onehot(training_labels, C), self.weights)
            self.weights -= self.lr * gradient
            predictions = self.predict(training_data)
            if accuracy_fn(predictions, training_labels) == 100:
                break
            iters += 1

        return self.predict(training_data)
        

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """

        probabilities = np.exp(np.dot(test_data, self.weights)) / np.sum( np.exp(np.dot(test_data, self.weights)), axis=1, keepdims=True)
        return np.argmax(probabilities, axis=1)
        

