'''
Group 1 Project
CMSC 495-7381
CTPnCS-7381-2232 1

Created on Apr 6, 2023

@author: Peter Koropey
'''
from tkinter import HORIZONTAL
from tkinter import Scale
from tkinter import ttk
from tkinter import BooleanVar
from utility.ML_Model_Params import mlParams as mlParams


class ParameterReader(ttk.Frame):
    '''
    This class uses the Tkinter library to create a frame with widgets
    for inputting the various parameters which the user can define for the 
    different machine learning models in this application.
    This is essentially a self-contained unit acting as a frame.
    '''


    def __init__(self, container, mlModelName, **kw):
        '''
        This class creates a tkinter Frame with widgets for inputting
        parameters for the machine learning models. As a user selects a model,
        the parameters available for input will change as this class adjusts.
        
        Paramters:
        container: tkinter parent container (likely a frame)
        mlModelName: string. Denotes the selected machine learning model.
        '''
        super().__init__(master=container)
        self.root = self
        
        self.enablePrePro = BooleanVar(value = False)
        
        self.mlModelName = mlModelName
        self.paramValsChecked = []
        self.createPanel()
        self.pack


    def createPanel(self):
        '''
        Creates the ParameterReader panel.
        Called by the Constructor.
        '''

        self.paramList = self.setParameters(self.mlModelName)
        self.paramInputs = self.constructParameterInput(self.paramList)
        
        
        self.root.columnconfigure(0, weight = 3)
        self.root.columnconfigure(1, weight = 1)
        
        for i in range (len(self.paramList) + 3):
            self.root.rowconfigure(i, weight = 1)
        
    
    def setParameters(self, mlModelName):
        '''
        Sets the parameters available for editing by the user
        based on the selected machine learning model. This is hardcoded,
        into the utility.ML_Model_Params dictionary.
        
        Parameters:
        mlModelName: string. The selected machine learning model.
        
        Returns:
        A List of Lists with the following indices:
        [0]: the Parameter input labels (string)
        [1]: the Parameter input widget types (string)
        [2]: The parameter input default values (int/string)
        [3]: the Name of the selected machine learning model.
        These are based on the utility.ML_Model_Params dictionary.
        '''
        return mlParams[mlModelName]
    
    def constructParameterInput(self, paramList):
        '''
        Constructs the widgets for inputting parameters for the
        selected machine learning model.
        
        Paramters:
        paramList: A 2D array of the type returned from the
            "setParamters()" method. See above.
        
        Returns:
        A List of with the following indices:
        [0]: String. The parameter labels (derived in part from paramList argument)
        [1]: Int/String. The user-input values.
        '''
        titleLabel = ttk.Label(self.root, text = paramList[3]);
        titleLabel.grid(column = 0, row = 0, columnspan = 2)
        
        paramLabels = list()
        paramInputs= list()
        thisLabel = ttk.Label()
        i = 0
        
        for i in range(len(paramList[0])-1):
            thisLabel = ttk.Label(self.root, text = paramList[0][i])
            thisLabel.grid(column = 0, row = i+1, sticky = 'w')
            paramLabels.append(thisLabel)
            
            if paramList[1][i] == "textbox":
                thisInput = ttk.Entry(self.root)
                thisInput.insert(0, paramList[2][i])
                
            if paramList[1][i] == "combobox":
                thisInput = ttk.Combobox(self.root, state = "readonly")
                thisInput['values'] = paramList[2][i]
                thisInput.current(0)
                
            if paramList[1][i] == "checkbutton":
                self.enablePrePro.set(value = bool( paramList[2][i] ))
                thisInput = ttk.Checkbutton(self.root, variable = self.enablePrePro, onvalue = True, offvalue = False)
              
            thisInput.grid(column = 1, row = i+1, sticky = 'w')
            paramInputs.append(thisInput)

        #Add the Partition scale and its label separate, so they can be vertically arranged.
        sliderLabel = ttk.Label(self.root, text = paramList[0][-1])
        sliderLabel.grid(column = 0, row = i+3, columnspan = 2, sticky = 's')
        slider = Scale(self.root, from_=1.0, to=99.0, orient = HORIZONTAL, length = 200, variable = paramList[2][-1])
        slider.set(paramList[2][-1])
        slider.grid(column = 0, row = i+4, columnspan = 2, sticky = 'n')
        
        paramLabels.append(sliderLabel)
        paramInputs.append(slider)
        
        return [paramLabels, paramInputs]

    def getParameterInputValues(self, mlParamVals, mlParamTypes, mlParamNames):
        '''
        Error checking for the user-inputs for machine learning parameters.
        Will throw exceptions if the user selects invalid values.
        
        Parameters:
        mlParamVals: List. user-input values for machine learning model parameters
        mlParamTypes: List of strings. Corresponds to widget types for inputting values.
        mlParamNames: List of strings. Corresponds to parameter names/labels.
        
        Returns:
        values: List. Int/strings. The validated user-input values.
        '''
        
        values = []
        
        for i in range(len(mlParamVals[1])):
            if mlParamTypes[i] == "textbox" or mlParamTypes[i] == "scale":
                try:
                    val = int(mlParamVals[1][i].get())  
                    if val <= 0:
                        raise ValueError("Input for " + mlParamNames[i]+" must be greater than 0")
                    else:
                        values.append(val)
                except ValueError:
                    raise ValueError("Input for " + mlParamNames[i]+" must be a positive integer")
            elif mlParamTypes[i] == "combobox":
                try: 
                    values.append(str(mlParamVals[1][i].get()))
                except:
                    raise ValueError("Combobox Error")
            elif mlParamTypes[i] == "checkbutton":

                try:
                    #print(self.enablePrePro.get())
                    values.append(self.enablePrePro.get())
                except:
                    raise ValueError("Checkbutton Error")
            else:
                raise ValueError("Invalid widget type.")
            
        return values
    
    def validateInputs(self, mlParamVals, mlParamTypes, mlParamNames):
        '''
        Checks to ensure the user has input correct/valid values for each ML
        learning model parameter input.
        
        Parameters:
        mlParamVals: List of int/strings. User input values.
        mlParamTypes: List of Strings. The widgets for inputting values.
        mlParamNames: List of STrings. Name of the ML learning parameters.
        
        '''
        try:
            values = self.getParameterInputValues(mlParamVals, mlParamTypes, mlParamNames)
            self.paramValsChecked = values
            
            return values
            
        except ValueError as e:
            raise ValueError(e)
            
    def getParamVals(self):
        '''
        Getter method for retrieving the values in the parameter input boxes.
        Returns: list of int/string
        '''
        self.validateInputs(self.paramInputs, self.paramList[1], self.paramList[0])
        return self.paramValsChecked
    
    def getParamLabels(self):
        '''
        Getter method for retrieving the labels of the parameter input boxes.
        Returns: list of String.
        '''
        labels = []
        for label in self.paramInputs[0]:
            labels.append(label.cget("text"))
        
        return labels

