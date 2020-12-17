import numpy as np
from collections import Counter
from scipy import stats
from tqdm import tqdm
from sklearn.preprocessing import Normalizer
from sklearn.metrics import mean_squared_error
from sklearn.metrics.pairwise import euclidean_distances





class KMeans():
    def __init__(self, X, k, F):
        """
        KMeans classifier with the k class.
        """
        self.k = k
        self.X = X
        self.Y = None
        self.dist = np.zeros((X.shape[0], k))
        self.Error = None
        self.centroid = np.zeros((k, X.shape[1] ))
        self.Featur = F
        
    def Random_centroid(self):
        """
        k centroid  with the KMean++ algorithm.
        """
        centroid_id = np.random.choice(self.X.shape[0], 1) 
        self.centroid[0] = self.X[centroid_id]
        for i in range(1, self.k):
            dist_temp =  euclidean_distances(self.X, self.centroid[0:i] )
            t = np.argmax(np.amin(dist_temp, axis = 1))
            self.centroid[i] = self.X[t]
        
    def Normalize(self):
        
        """
        Normalize the data.
        """
        
        transformer = Normalizer().fit(self.X)  
        self.X = transformer.transform(self.X)
        
    def Calculate_distance_and_cluster(self):
        
        """
        1) Calculate the distance between each centroid and all points.
        2) Calculate cluster of each point as a label Y.
        """

        self.dist= euclidean_distances(self.X, self.centroid )
        self.Y = np.argmin(self.dist, axis=1)
        
        
    def Calculate_centroid(self):  
        
        """
        Calculate the mean of each cluster as a centroid.
        """
        
        l=np.zeros((self.k,self.Featur))
        for i in range(self.k):
            l[i] = self.X[self.Y == i].mean(axis = 0)
        self.centroid = np.asarray(l)                    

        
    def Calculate_Error(self):
        
        """
        Calculate Error
        """
        l=np.zeros((self.k,))
        for i in range(self.k):
            l[i] = (np.min(self.dist[self.Y == i], axis=1)).sum()
        self.Error =np.asarray(l)
        
    def Main(self, Num_iter):
        
        """
        calculate the label of the data according to the Num_iter.
        """

        for i in tqdm(range(Num_iter)):
            self.Calculate_distance_and_cluster()
            self.Calculate_centroid()
            self.Calculate_Error()            
        self.Calculate_Error()
        
        return self.Error, self.Y, self.centroid