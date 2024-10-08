import numpy as np
import sys

class LinearRegression(object):
    """
        Linear regressor object.
        Note: This class will implement BOTH linear regression and ridge regression.
        Recall that linear regression is just ridge regression with lambda=0.
    """

    def __init__(self,  lmda, task_kind ="regression"):
        """
            Initialize the task_kind (see dummy_methods.py)
            and call set_arguments function of this class.
        """
        self.lmda = lmda
        self.task_kind = task_kind


    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): regression target of shape (N,regression_target_size)
            Returns:
                pred_labels (np.array): target of shape (N,regression_target_size)
        """
        ##
        ###
        #### YOUR CODE HERE!
        ###
        ##The formula for linear regression coefficients is: (XTX)−1   .XT   .y

        x = training_data
        y = training_labels

        firstPart = np.dot(x.T, x)
        secondPart = np.dot(x.T, y)

        termForRidge = self.lmda * np.identity(firstPart.shape[0])

        self.coefficientFormula = np.dot(np.linalg.inv(firstPart + termForRidge), secondPart)


        pred_regression_targets = np.dot(x, self.coefficientFormula)

        return pred_regression_targets


    def predict(self, test_data):
            """
                Runs prediction on the test data.
                
                Arguments:
                    test_data (np.array): test data of shape (N,D)
                Returns:
                    test_labels (np.array): labels of shape (N,regression_target_size)
            """
            ##
            ###
            #### YOUR CODE HERE!
            ###
            ##
            pred_regression_targets = np.dot(test_data, self.coefficientFormula)
    
            return pred_regression_targets 
