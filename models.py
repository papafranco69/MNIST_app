from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy import stats
import numpy as np
import pandas as pd
import joblib


"""
Notes to put on docs:



"""
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X, y = mnist["data"], mnist["target"]
# cast y to integers
y = y.astype(np.uint8)
X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))



"""
K-Nearest Neighbor

- User-defined parameters:
 - n_neighbors: Number of neighbors to use
 - weights: {"uniform", "distance"}
    "uniform" : uniform weights. All points in each neighborhood are weighted equally.
    "distance" : weight points by the inverse of their distance. in this case, 
    closer neighbors of a query point will have a greater influence than neighbors which are further away.

Using GridSearchCV we found the best combination of the parameters are {'n_neighbors': 4, 'weights': 'distance'}
Here is the code, this might takes up to 16 hours to run depending on your hardware

from sklearn.model_selection import GridSearchCV

param_grid = [{'weights':["uniform", "distance"], 'n_neighbors': [3, 4, 5]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3)
grid_search.fit(X_train, y_train)

"""

n_neighbors = 4
weights = 'distance'

knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
knn_clf.fit(X_train, y_train)
# Return the mean accuracy on the given test data and labels.
print("Test Score: ", knn_clf.score(X_test, y_test))




class KNN_scratch:
    """
    KNN from Scratch
    """
    def __init__(self, k=n_neighbors):
        self.k = k
        self.X_train = None
        self.y_train = None

    def euclidean(p, q):
        """
        Using this function to calculate the distance between each observation in the training data
        param p: np.array, first vector
        param q: np.array, second vector
        return float, distance
        """
        return np.sqrt(np.sum((p - q) ** 2))
    
    def fit(self, X, y):
        """
        Trains the model. 
        No training is required for KNN,
        so this saves the parameters to the constructor.

        param X: pd.Dataframe, features
        param y: pd.Series, target
        return None
        """

        self.X_train = X
        self.y_train = y
    
    def predict(self, X):
        """
        Predicts the class labels based on nearest neighbors
        param X: pd.Dataframe, features
        return: np.array, predicted class labels
        """

        predictions = []
        for p in X:
            distances = [self.euclidean(p, q) for q in self.X_train]
            sorted_k = np.argsort(distances)[:self.k]
            k_nearest = [self.y_train[y] for y in sorted_k]
            predictions.append(stats.mode(k_nearest)[0][0])

        return np.array(predictions)
    



    

