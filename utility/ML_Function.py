'''
Created on Apr 15, 2023

@author: Peter Koropey
'''
from models import KNN_scratch
from gui_package.ParameterReader import ParameterReader
from gui_package.DigitCanvas import DigitCanvas
from sklearn.datasets import  fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import  StandardScaler
import matplotlib.pyplot as plt
from scipy import  stats
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier


class ML_Function(object):
    '''
    Class performs all the functions "behind the scenes" for the 
    GUI class of the ML Application.
    '''


    def __init__(self, pr = None, dc = None, defaultModel = "knn"):
        '''
        Paramters:
        pr: ParameterReader class
        dc: DigitCanvas class
        '''
        
        self.pr = pr
        self.dc = dc
        self.isTrained = False
        self.mlModelName = defaultModel
        self.mlModel = None
        self.mlParamVals = self.getMLparameters()
    
    def loadDataSet(self, mlParamVals):
        '''
        Loads the MNIST dataset.
        
        Parameters:
        mlParamVals: a list of int/String derived from the ParamterReader class.
        '''
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
        self.X, self.y = mnist["data"], mnist["target"]
        self.y = self.y.astype(np.uint8)
        testPercent = 1.0 - mlParamVals[2]*0.01
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size =testPercent, random_state = 1)
    
    
    def printRowMajor(self, array):
        '''
        Prints an array as a matrix in row-major order.
        Used during testing.
        
        Paramters:
        array: numpy 1D integer array.
        '''
        string = ""
        x = 0
        for i in range(array.shape[0]):
            string = string + str(array[i]) + " "
            x+=1
            if x == 28:
                print(string)
                string = ""
                x = 0
    
    def setMLmodel(self, mlModelName):
        self.isTrained = False
        self.mlModelName = mlModelName
    
    def testMLmodel(self, mlModel, testingPartition):
        '''
        Tests the selected machine learning model after training.
        
        Parameters:
        
        Returns:
        '''
        if self.isTrained:
            return mlModel.predict(testingPartition)
        
    
    def trainMLmodel(self, mlModelName, mlParamVals):
        '''
        Instantiates and Trains the user-selected Machine Learning Model.
        
        Parameters:
        mlModelName: String
        mlParamVals: list of integers/strings
        
        Returns:
        mlModel: the machine learning model
        '''
        if mlModelName == "knn":
            #mlModel = KNN_scratch(mlParamVals[0], mlParamVals[1])
            mlModel = KNN_scratch(int(mlParamVals[0]))
            
            
        elif mlModelName == "randomForest":
            #uses sklearn implementation for now
            mlModel = RandomForestClassifier(max_depth = int(mlParamVals[0]), n_estimators = int(mlParamVals[1]))

        
        elif mlModelName == "knn_sk":
            mlModel = KNeighborsClassifier(n_neighbors = int(mlParamVals[0]))
        elif mlModelName == "randomForest_sk":
            mlModel = RandomForestClassifier(max_depth = int(mlParamVals[0]), n_estimators = int(mlParamVals[1]))
        
        mlModel.fit(self.X_train, self.y_train)
        self.isTrained = True
        
        self.mlModel = mlModel
        
        return mlModel
        
        
    
    def getMLparameters(self):
        '''
        Gets the parameters from the loaded ParameterReader class associated
        with the user-selected machine learning model.
        
        Returns:
        numpy 1D array of integers/lists
        '''
        return self.pr.getParamVals()
    
    def getDrawnDigitArray(self):
        '''
        Gets the 784 element integer array based on the user's drawing
        in the DigitCanvas class.
        
        Returns:
        numpy 1D integer array.
        '''
        return self.dc.deriveArray()
        
    def classifyDigit(self):
        '''
        Classifies the User Drawn digit by returning the expected class value
        from the selected ML model for the drawn digit.
        
        Returns:
        Integer
        '''
        if self.isTrained:
            return self.testMLmodel(self.mlModel, self.getDrawnDigitArray().reshape(1, -1))[0]
        else:
            raise ValueError("You must first Train a Machine Learning Model!")
    
    def checkParameterChange(self, mlParamVals):
        '''
        Checks if any user input parameters (or ML model type) has changed,
        so that model can be retrained if changes have occurred.
        Returns true if change is detected.
        
        Parameters:
        mlParamVals: list of integers/String
        
        Returns:
        Boolean (true if change detected)
        '''
        currentVals = self.getMLparameters()
        for i in range(len(mlParamVals)):
            if mlParamVals[i] != currentVals[i]:
                return True
        return False
        
        return self.isTrained
    
    def runMLtesting(self):
        try:
            if self.isTrained:
                self.testMLmodel(self.mlModel, self.X_test)
            else:
                raise ValueError("Machine Learning model is not yet trained!")
        except ValueError as e:
            raise ValueError(e)
    
    def runMLtraining(self):
        try:
            paramsChanged = self.checkParameterChange(self.mlParamVals)
            
            if not self.isTrained or paramsChanged:
                self.mlParamVals = self.getMLparameters()
                self.loadDataSet(self.mlParamVals)
                self.trainMLmodel(self.mlModelName, self.mlParamVals)
                #print(self.mlParamVals, self.pr.getParamLabels())
                return self.mlParamVals, self.pr.getParamLabels()
        except ValueError as e:
            self.isTrained = False
            raise ValueError(e)
        
    
    
    def main(self):
        paramsChanged = self.checkParameterChange(self.mlParamVals)
        
        if not self.isTrained or paramsChanged:
            self.mlParamVals = self.getMLparameters()
            self.loadDataSet(self.mlParamVals)
            mlModel = self.trainMLmodel(self.mlModelName, self.mlParamVals)
            print("Trained!")
        
        return self.classifyDigit()
    
    
    def setParamReader(self, pr):
        self.pr = pr
        
        
        