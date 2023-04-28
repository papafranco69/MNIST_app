from sklearn.neighbors import KNeighborsClassifier
from sklearn.base import BaseEstimator
from annoy import AnnoyIndex
import numpy as np



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

"""
Sample code for sklearn's KNN classifier
"""

# knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
# knn_clf.fit(X_train, y_train)
# # Return the mean accuracy on the given test data and labels.
# print("Test Score: ", knn_clf.score(X_test, y_test))



class KNN_scratch(BaseEstimator):
    """
    Original implementation of K-Nearest-Neighbor algorithm.
    Inheriting from sklearn.base.BaseEstimator,
    which allow users to get and set the parameters
    of the estimator as a dictionary. 

    Example:
    ----------
    from models import KNN_scratch

    n_neighbor = 3
    knn_scratch = KNN_scratch(k=n_neighbors)
    knn_scratch.fit(X_train, y_train)
    y_pred = knn_scratch.predict(X_test)
    
    """
    def __init__(self, k=5, n_trees=10, n_jobs=-1):
        self.k = k
        self.n_trees = n_trees
        self.n_jobs = n_jobs
        self.X_train = None
        self.y_train = None
        self.index = None

    def fit(self, X_train, y_train):
        """
        Using AnnoyIndex data structure to build an index of the training data;
        so we can retrieve the nearest neighbors of a query point more efficiently.
        Each data point in X_train is added to the AnnoyIndex object using its index
        as an unique indentifier. 

        Parameters:
        -----------
        X_train: pd.Dataframe features
        y_train: pd.Serie Target
        """
        self.X_train = X_train
        self.y_train = y_train
        self.index = AnnoyIndex(self.X_train.shape[1], metric='euclidean')
        for i, x in enumerate(self.X_train):
            self.index.add_item(i, x)
        self.index.build(self.n_trees)

    def predict(self, X_test):
        """
        Loops over each data point in the X_test and use the `get_nns_by_vector`
        to retrieve the K nearest neighbors of the query point in the training dataset.
        Using max() and key() functions to find the most common label among the k nearest neighbors. 

        Parameters:
        -----------
        X_test: pd.Dataframe features
        Return: numpy array predicted labels for each test sample in X_test
        """
        y_pred = []
        for x in X_test:
            idx = self.index.get_nns_by_vector(x, self.k)
            k_nearest_labels = [self.y_train[i] for i in idx]
            most_common_label = max(set(k_nearest_labels), key=k_nearest_labels.count)
            y_pred.append(most_common_label)
        return y_pred
    
    def get_params(self, deep=True):
        """
        Return the parameters of the model as a dictionary.
        """
        return {"k": self.k, "n_trees": self.n_trees, "n_jobs": self.n_jobs}
    
    def predict_proba(self, X_test):
        """
        Loops over each data point in the X_test and use the `get_nns_by_vector`
        to retrieve the K nearest neighbors of the query point in the training dataset.
        Compute the probability of each class using the proportion of neighbors 
        with the same label over the total number of neighbors.

        Parameters:
        -----------
        X_test: pd.Dataframe features
        Return: numpy array of shape (n_samples, n_classes) with predicted probabilities
        """
        y_proba = []
        for x in X_test:
            idx = self.index.get_nns_by_vector(x, self.k)
            k_nearest_labels = [self.y_train[i] for i in idx]
            proba = np.zeros(len(np.unique(self.y_train)))
            for label in np.unique(self.y_train):
                proba[label] = k_nearest_labels.count(label) / self.k
            y_proba.append(proba)
        return np.array(y_proba)


