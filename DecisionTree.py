"""The Node class serves to provide the framework for each leaf, or node"""
import numpy as np
from collections import Counter


class Node:

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None, value_proba = None, all_proba = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
        self.value_proba = value_proba
        self.all_proba = all_proba

    def is_leaf_node(self):
        return self.value is not None


"""The DecisionTree class includes all associated functions for
   information gain, splitting, creating trees, and predicting"""


class DecisionTree:
    def __init__(self, min_samples_split=2, max_depth=30, n_features=None):
        self.min_samples_split = min_samples_split
        self.n_features = n_features
        self.root = None
        
        if max_depth > 100:
            raise ValueError( "Maximum Decision Depth must not be greater than 100")
        else:
            self.max_depth = max_depth
        

    def fit(self, X, y):
        """This function creates the decision tree"""
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)
        self.root = self._grow_tree(X, y)

    def _grow_tree(self, X, y, depth=0):
        """This function assists the fit() function in creating the tree"""
        n_samples, n_feats = X.shape
        n_labels = len(np.unique(y))

        # check the stopping criteria
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value, val_proba, all_proba = self._most_common_label(y)
            return Node(value=leaf_value, value_proba = val_proba, all_proba = all_proba)

        feat_idxs = np.random.choice(n_feats, self.n_features, replace=False)

        # find the best split
        best_feature, best_thresh = self._best_split(X, y, feat_idxs)

        # create child nodes
        left_idxs, right_idxs = self._split(X[:, best_feature], best_thresh)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        return Node(best_feature, best_thresh, left, right)

    def _best_split(self, X, y, feat_idxs):
        """This function determines which split has the best entropy"""
        best_gain = -1
        split_idx, split_threshold = None, None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]
            thresholds = np.unique(X_column)

            for thr in thresholds:
                # calculate the information gain
                gain = self._information_gain(y, X_column, thr)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        """This function calculates the entropy of each split"""
        # parent entropy
        parent_entropy = self._entropy(y)

        # create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs) == 0 or len(right_idxs) == 0:
            return 0

        # calculate the weighted avg. entropy of children
        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])
        child_entropy = (n_l / n) * e_l + (n_r / n) * e_r

        # calculate the IG
        information_gain = parent_entropy - child_entropy
        return information_gain

    def _split(self, X_column, split_thresh):
        """This function splits the data based on the best split"""
        left_idxs = np.argwhere(X_column <= split_thresh).flatten()
        right_idxs = np.argwhere(X_column > split_thresh).flatten()
        return left_idxs, right_idxs

    def _entropy(self, y):
        """This function calculates the entropy"""
        hist = np.bincount(y)
        ps = hist / len(y)
        return -np.sum([p * np.log(p) for p in ps if p > 0])

    def _most_common_label(self, y):
        """
        This function assists in classifying an item
        It also calculates the probability of each class for an item
        
        returns:
        int, float, array of floats.
        """
        counter = Counter(y)
        value = counter.most_common(1)[0][0]
        
        totalVals = counter.most_common(10) #10 is the number of classes in MNIST
        
        #Calculate the otal number of elements
        runningTotal = 0.0
        for val in totalVals:
            runningTotal += val[1]
        
        #Calculate the proportion of elements of a certain class out of all elements
        local_proba = []
        for i in range(10):
            local_proba.append(counter[i] / runningTotal)
        
        return value, totalVals[0][1]/runningTotal, local_proba

    def prediction(self, X):
        """This function predicts based on the fitted tree"""
        return np.array([self._traverse_tree(x, self.root) for x in X])

    def _traverse_tree(self, x, node):
        """This function assists the predict function in navigating
           the fitted tree"""

        if node.is_leaf_node():
            return node.value

        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)
    
    def _traverse_tree_proba(self, x, node):
        '''
        Traverses the Tree but returns a leaf node's probabilities
        rather than the leaf node's value.
        
        Parameters:
        x: pandas Dataframe or numpy array
        
        Returns:
        ap: matrix of shape (elements, classes)
        '''

        if node.is_leaf_node():
            ap = node.all_proba
            return ap
            #vp = node.value_proba
            #return vp

        if x[node.feature] <= node.threshold:
            return self._traverse_tree_proba(x, node.left)
        return self._traverse_tree_proba(x, node.right)
    
    
    #Needs this for GUI Testing
    def predict_proba(self, X):
        '''
        Returns the probability of each class from the leaf nodes in the decision
        tree based on what leaf node the value would be sorted into
        
        Parameters:
        X: pd.Dataframe Features
        
        Retunrs:
        value: Return: numpy array of shape (n_samples, n_classes) with probabilities
        '''
        results = []
        for x in X:
            local_proba = self._traverse_tree_proba(x, self.root)
            results.append(local_proba)
        value =  np.array(results)
        
        return value
        
