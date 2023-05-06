import numpy as np
from DecisionTree import DecisionTree
from sklearn.model_selection import train_test_split
from scipy import stats as st
import time
from collections import Counter

"""This program is a from-scratch random forest classifier.
   The base DecisionTree class was written by Misra Turp and
   modified/improved to handle larger datasets and be used
   in a random forest classifier"""

"""Contributors: (UMGC Group 1): Ben Y., Lena Y., Peter K., Anishka F."""


class RandomForestScratch:

    def __init__(self, n_trees, max_depth):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.X_train = None
        self.y_train = None
        #PK- Added an attribute to make it compatible with gui
        self.trees = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

        self.X_train_split_arr = np.array_split(self.X_train, self.n_trees)
        self.y_train_split_arr = np.array_split(self.y_train, self.n_trees)

        forest = []
        i = 0
        while i <= (self.n_trees - 1):
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(self.X_train_split_arr[i], self.y_train_split_arr[i])
            forest.append(tree)
            i += 1
        
        self.trees = forest #PK - added this for GUI compatibility
        return forest

    def predict(self, X_test, forest = None): #PK - added default arg for GUI compatibility
        #PK - added this for GUI compatibility
        if forest == None:
            forest = self.trees
        
        forest_predictions = []
        i = 0

        while i <= (self.n_trees - 1):
            pred = forest[i].prediction(X_test)
            forest_predictions.append(pred)
            i += 1

        # Tally votes of each tree predictions
        forest_results, _ = st.mode(forest_predictions, keepdims=False)

        return forest_results
    
    #Needs this for GUI Testing
    def predict_proba(self, X_test, trees = None):
        '''
        Returns an array of probabilities for an element belonging to any
        of the classes, as an average of its probabilities among all trees.
        
        mean: NumPy array of shape (elements, classes)
        '''
        #For GUI compatibilty
        if trees == None:
            trees = self.trees
        
        probs = []
        
        for tree in trees:
            probs.append(tree.predict_proba(X_test))
        
        mean = np.mean(probs, axis = 0)
        
        return mean
        
