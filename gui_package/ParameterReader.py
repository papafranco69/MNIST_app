'''
Group 1 Project
CMSC 495-7381
CTPnCS-7381-2232 1

Created on Apr 6, 2023

@author: Peter Koropey
'''

from tkinter import StringVar
from tkinter import HORIZONTAL
from tkinter import Scale
from tkinter import messagebox
from tkinter import ttk

class ParameterReader(ttk.Frame):
    '''
    This class uses the Tkinter library to create a frame with widgets
    for inputting the various parameters which the user can define for the 
    different machine learning models in this application.
    '''


    def __init__(self, container, mlModelName):
        '''
        This class creates a tkinter Frame with widgets for inputting
        parameters for the machine learning models. As a user selects a model,
        the parameters available for input will change as this class adjusts.
        
        Paramters:
        container: tkinter parent container (likely a frame)
        mlModelName: string. Denotes the selected machine learning model.
        '''
        super().__init__(container)
        self.root = container
        self.mlModelName = mlModelName
        self.paramValsChecked = []
        self.createPanel()
    
    
    def createPanel(self):
        '''
        Creates the ParameterReader panel.
        Called by the Constructor.
        '''
        
        #Clears all widgets from this frame. Prevents graphical errors.
        for widget in self.root.winfo_children():
            widget.destroy()

        self.paramList = self.setParameters(self.mlModelName)
        self.paramInputs = self.constructParameterInput(self.paramList)
        
        button = ttk.Button(self.root, text = "Validate", command = lambda: self.validateInputs(self.paramInputs, self.paramList[1], self.paramList[0]))
        button.grid(column = 0, row = len(self.paramList) + 2, columnspan = 2)
        
        self.root.columnconfigure(0, weight = 3)
        self.root.columnconfigure(1, weight = 1)
        
        for i in range (len(self.paramList) + 3):
            self.root.rowconfigure(i, weight = 1)
        
    
    
    def setParameters(self, mlModelName):
        '''
        Sets the parameters available for editing by the user
        based on the selected machine learning model. This is hardcoded,
        and this method will need adjusting as more machine learning models are added.
        
        Parameters:
        mlModelName: string. The selected machine learning model.
        
        Returns:
        A List of Lists with the following indices:
        [0]: the Parameter input labels (string)
        [1]: the Parameter input widget types (string)
        [2]: The parameter input default values (int/string)
        [3]: the Name of the selected machine learning model.
        '''
        
        mlParamLabels = []
        mlParamTypes = []
        mlParamValues = []
        mlTitle = StringVar
        if mlModelName == "knn":
            mlParamLabels = ["Number of Neighbors", "Weight Function"]
            mlParamTypes = ["textbox", "combobox"]
            mlParamValues = [5, ('uniform', 'distance')]
            mlTitle = "K-Nearest Neighbors"
        
        if mlModelName == "randomForest":
            mlParamLabels = ["Decision Depth", "Number of Estimators"]
            mlParamTypes = ["textbox", "textbox"]
            mlParamValues = [10, 10]
            mlTitle = "Random Forest"
        
        return [mlParamLabels, mlParamTypes, mlParamValues, mlTitle]
    
    
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
        
        for i in range(len(paramList[0])):
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

              
            thisInput.grid(column = 1, row = i+1, sticky = 'w')
            paramInputs.append(thisInput)
        
        sliderLabel = ttk.Label(self.root, text="Percentage of Dataset\nUsed for Training:")
        sliderLabel.grid(column = 0, row = i+2, columnspan = 2, sticky = 's')
        sliderInput = 75.0
        slider = Scale(self.root, from_=1.0, to=99.0, orient = HORIZONTAL, length = 200, variable = sliderInput)
        slider.set(sliderInput)
        slider.grid(column = 0, row = i+3, columnspan = 2, sticky = 'n')
        
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
        mlParamTypes.append("scale")
        values = []
        for i in range(len(mlParamVals[1])):
            if mlParamTypes[i] == "textbox" or mlParamTypes[i] == "scale":
                try:
                    val = int(mlParamVals[1][i].get())
                    
                    if val < 0:
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
            self.paramValsChecked = []
            
            for value in mlParamVals[1]:
                self.paramValsChecked.append(value.get())
            
        except ValueError as e:
            messagebox.showerror(message=str(e), title = "Error")
            
    def getParamVals(self):
        self.validateInputs(self.paramInputs, self.paramList[1], self.paramList[0])
        return self.paramValsChecked

