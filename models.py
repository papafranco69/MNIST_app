from sklearn.datasets import fetch_openml
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import time
import joblib
from annoy import AnnoyIndex


"""
Notes to put on docs:



"""
mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')

X, y = mnist["data"], mnist["target"]
# cast y to integers
y = y.astype(np.uint8)

# Standardize the data
scaler = StandardScaler()
X_std = scaler.fit_transform(X)

# Reduce the dimensionality using PCA
pca = PCA(n_components=50)
X_pca = pca.fit_transform(X_std)


# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)




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

# knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights)
# knn_clf.fit(X_train, y_train)
# # Return the mean accuracy on the given test data and labels.
# print("Test Score: ", knn_clf.score(X_test, y_test))



class KNN_scratch:
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

    
knn = KNN_scratch(k=3)

startTime = time.time()

# Fit the model to the training data
knn.fit(X_train, y_train)

# Make predictions on the test data
y_pred = knn.predict(X_test)

# Compute the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)

print('Accuracy:', accuracy)
print("--- %s seconds ---" % (time.time() - startTime)) 
