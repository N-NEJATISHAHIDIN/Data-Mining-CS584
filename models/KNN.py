import numpy as np
from scipy.spatial import distance
from collections import Counter
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import Normalizer



class KNN():
    def __init__(self, k):
        """
        Initializes the KNN classifier with the k.
        """
        self.k = k
        self.X = None
        self.Y = None
        self.dist = None
        
    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just 
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X = X
        self.Y = y
        
    def find_dist(self,  X_test):
        #cosin from scratch
        #normalize the X_train and X_test
        
        transformer = Normalizer().fit(self.X)  
        X_train_normalized = transformer.transform(self.X)
        
        transformer = Normalizer().fit(X_test) 
        X_test_normalized = transformer.transform(X_test)
        self.dist = X_train_normalized.dot(X_test_normalized.transpose())
        
        
        return self.dist
    
    def predict(self):
        
        #self.find_dist(X_test) 
        result = np.argsort(self.dist, axis=0)
        labels = self.Y[result[-(self.k):]]
        final_labels = stats.mode(labels,axis=0)[0]
        
        """
        Predict labels for test data using the computed distances.

        Inputs:
        - X_test: A numpy array of shape (num_test, D) containing test data consisting
             of num_test samples each of dimension D.

        Returns:
        - pred: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].  
        """
        return final_labels[0]
    


