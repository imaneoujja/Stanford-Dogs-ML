import numpy as np

class KNN(object):
    """
        kNN classifier object.
    """

    def __init__(self, k=1, task_kind = "classification"):
        """
            Call set_arguments function of this class.
        """
        self.k = k
        self.task_kind =task_kind

    def fit(self, training_data, training_labels):
        """
            Trains the model, returns predicted labels for training data.
            Hint: Since KNN does not really have parameters to train, you can try saving the training_data
            and training_labels as part of the class. This way, when you call the "predict" function
            with the test_data, you will have already stored the training_data and training_labels
            in the object.

            Arguments:
                training_data (np.array): training data of shape (N,D)
                training_labels (np.array): labels of shape (N,)
            Returns:
                pred_labels (np.array): labels of shape (N,)
        """

        self.training_data = training_data # shape (NxD)
        self.training_labels = training_labels # shape (Nx1)
        pred_labels = self.predict(training_data)
        return pred_labels

    def predict(self, test_data):
        """
            Runs prediction on the test data.

            Arguments:
                test_data (np.array): test data of shape (N,D)
            Returns:
                test_labels (np.array): labels of shape (N,)
        """
        
        if self.task_kind == "classification":
            test_labels = np.apply_along_axis(self.kNN_clf_single_sample, 1, test_data, self.training_data, self.training_labels, self.k)
        else: # Regression
            test_labels = np.apply_along_axis(self.kNN_reg_single_sample, 1, test_data, self.training_data, self.training_labels, self.k)

        assert test_labels.shape[0] == test_data.shape[0]
        return test_labels
    

    def kNN_reg_single_sample(self, sample, training_data, training_centers, k):
        """Returns the prediction of a single sample.
    
        Inputs:
            sample: shape (D,)
            training_data : shape (NxD)
            training_centers: shape (N, 2)
            k: integer
        Outputs:
            predicted center of given sample of shape (1, 2)
        """
        # Compute distances
        distances = self.l1_dist(sample)
        
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k, distances)

        # Find the centers of the k nearest data samples
        k_nearest_centers = training_centers[nn_indices]
        assert k_nearest_centers.shape[1] == 2

        # Compute the prediction (continuous value) for the regression task
        prediction = np.average(k_nearest_centers, axis=0)
        
        return prediction

    def kNN_clf_single_sample(self, sample, training_features, training_labels, k):
        """Returns the label of a single unlabelled sample.
    
        Inputs:
            sample: shape (D,) 
            training_features: shape (NxD)
            training_labels: shape (N,) 
            k: integer
        Outputs:
            predicted label of given sample
        """
        # Compute distances
        distances = self.l1_dist(sample)
        
        # Find neighbors
        nn_indices = self.find_k_nearest_neighbors(k, distances)
        
        # Get neighbors' labels
        neighbor_labels = training_labels[nn_indices]
        
        # Pick the most common
        best_label = self.best_label(neighbor_labels)
        return best_label
        

    def find_k_nearest_neighbors(self, k, distances):
        """ Find the indices of the k smallest distances from a list of distances.
    
        Inputs:
            k: integer
            distances: shape (N,) 
        Outputs:
            indices of the k nearest neighbors: shape (k,)
        """
        all_indices = np.argsort(distances)
        indices = all_indices[:k]
        return indices

    def best_label(self, neighbor_labels):
        """Return the most frequent label in the neighbors'.
    
        Inputs:
            neighbor_labels: shape (N,) 
        Outputs:
            most frequent label
        """
        res = np.bincount(neighbor_labels)
        return np.argmax(res)

    def euclidean_dist(self, sample):
        """Compute the Euclidean distance between a single sample
        vector and all training_samples.
    
        Inputs:
            sample: shape (D,)
        Outputs:
            euclidean distances: shape (N,)
        """
        res = np.square(sample.T - self.training_data)
        return np.sqrt(np.sum(res, axis=1))

    def chi_square_dist(self, sample):
        """Compute the Chi-square distance between a single sample
        vector and all training_samples.
    
        Inputs:
            sample: shape (D,)
        Outputs:
            chi-square distances: shape (N,)
        """
        res = np.square(sample.T - self.training_data) / (sample.T + self.training_data)
        return np.sqrt(np.sum(res, axis=1))

    def l1_dist(self, sample):
        """Compute the L1 distance between a single sample
        vector and all training_samples.
    
        Inputs:
            sample: shape (D,)
        Outputs:
            L1 distances: shape (N,)
        """
        res = np.abs(sample.T - self.training_data)
        return np.sum(res, axis=1)
        