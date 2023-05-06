'''
Created on Apr 15, 2023

@author: Peter Koropey
'''
from models import KNN_scratch
from gui_package.ParameterReader import ParameterReader
from gui_package.DigitCanvas import DigitCanvas
from sklearn.datasets import  fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from RandomForestScratch import RandomForestScratch
from utility.ML_Model_Params import mlParams as mlParams
from utility.ML_Model_Params import preProAllowed as preProAllowed
from utils import plot_ROC
from utils import Metrics
from utils import Preprocessing



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
        defaultModel: the starting ML model (string)
        '''
        
        self.pr = pr
        self.dc = dc
        self.isTrained = False
        self.mlModelType = defaultModel
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
        #The partition value is always that last parameter from ParameterReader class.
        testPercent = 1.0 - mlParamVals[-1]*0.01
        randomSeed = mlParamVals[-2]
        
        #The preprocessing boolean value is the first parameter, if it has been made available
        self.preprocess = 0
        if preProAllowed:
            self.preprocess = mlParamVals[0]
        
        if self.preprocess:
            localX = Preprocessing(self.X).fit_transform()
        else:
            localX = self.X
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(localX, self.y, test_size = testPercent, random_state = randomSeed)

    
    def setMLmodel(self, mlModelType):
        '''
        Sets the "mlModelType" attribute to parameter.
        
        Paramters:
        mlModelType: String. Must be compatible with mlModels
        '''
        self.isTrained = False
        self.mlModelType = mlModelType
    
    def testMLmodel(self, mlModel, testingPartition):
        '''
        Tests the selected machine learning model after training.
        
        Parameters:
        mlModel: machine learning model class
        testingPartition: float
        
        Returns:
        numpy array (predicted value)
        '''
        if self.isTrained:
            return  mlModel.predict(testingPartition)
            
        
    
    def trainMLmodel(self, mlModelType, mlParamVals):
        '''
        Instantiates and Trains the user-selected Machine Learning Model.
        
        Parameters:
        mlModelType: String
        mlParamVals: list of integers/strings
        
        Returns:
        mlModel: the machine learning model
        '''
        idx = 0
        if preProAllowed:
            idx = 1
        
        
        if mlModelType == "knn":
            mlModel = KNN_scratch(int(mlParamVals[idx]))
            
            
        elif mlModelType == "randomForest":
            #mlModel = RandomForestClassifier(max_depth = int(mlParamVals[idx]), n_estimators = int(mlParamVals[idx+1]))
            mlModel = RandomForestScratch(max_depth = int(mlParamVals[idx]), n_trees = int(mlParamVals[idx+1]))
        
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
    
    def getMLparamLabels(self):
        '''
        Getter method for ML parameter labels.
        Returns:
        List of Strings
        '''
        return self.pr.getParamLabels()
    
    
    def getMLmodelName(self):
        '''
        Getter method for selected ML model name.
        Returns:
        String
        '''
        return mlParams[self.mlModelType][3]
    
    def getDrawnDigitArray(self):
        '''
        Gets the 784 element integer array based on the user's drawing
        in the DigitCanvas class.
        
        Returns:
        numpy 1D integer array.
        '''
        return self.dc.deriveArray()
            
    def getIsTrained(self):
        '''
        Checks whether the selected ML model is trained.
        Returns: Boolean.
        '''
        return self.isTrained
    
    def classifyDigit(self):
        '''
        Classifies the User Drawn digit by returning the expected class value
        from the selected ML model for the drawn digit.
        
        Paramters:
        preprocess: Boolean. Indicates preprocessing
        
        Returns:
        Integer
        '''
        if self.isTrained:
            
            if self.preprocess:
                temp = self.getDrawnDigitArray().reshape(1, -1)
                value = Preprocessing(temp).fit_transform()
            else:
                value = self.getDrawnDigitArray().reshape(1, -1) 
            return self.testMLmodel(self.mlModel, value)[0]
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
        '''
        Method for testing the machine learning model.
        This is what external classes should call.
        '''
        try:
            if self.isTrained:
                y_pred = self.testMLmodel(self.mlModel, self.X_test)
                eval = Metrics(self.y_test, y_pred)
                precision = eval.get_precision()
                recall = eval.get_recall()
                f1 = eval.get_f1_score()
                fig = plot_ROC(self.mlModel, self.X_test, self.y_test).plot(self.getMLmodelName())
                return precision, recall, f1, fig
            else:
                raise ValueError("Machine Learning model is not yet trained!")
        except ValueError as e:
            raise ValueError(e)
    
    def runMLtraining(self):
        '''
        Method for running all the machine learning training.
        This is what external classes should call.
        '''
        temp = self.mlParamVals
        try:
            self.mlParamVals = self.getMLparameters()
            self.loadDataSet(self.mlParamVals)
            self.trainMLmodel(self.mlModelType, self.mlParamVals)
            return self.mlParamVals, self.pr.getParamLabels()
        except ValueError as e:
            self.mlParamVals = temp
            raise ValueError(e)
        
    
    def setParamReader(self, pr):
        '''
        Setter method for the self.pr attribute (ParameterReader instance
        associated with instance of ML_Function)
        Parameters:
        pr: ParameterReader class.
        '''
        self.pr = pr
        
        
        