'''
Anishka Forbes and Peter Koropey
'''

from tkinter import ttk
from gui_package.DigitCanvas import DigitCanvas
from gui_package.ParameterReader import ParameterReader
from gui_package.GraphFrame import GraphFrame
from gui_package.ML_Console import ML_Console
from utility.ML_Function import ML_Function
from tkinter import messagebox
from tkinter import Tk
from tkinter import Menu
from tkinter import BooleanVar


class ML_GUI(object):
    '''
    Creates the GUI for the Machine Learning Application
    '''

    def __init__(self):
        '''
        Constructor method.
        '''
        self.root = Tk()
        self.root.title('Machine Learning Graphical Tool')
        
        # create containers
        datasetFrame = ttk.Frame(self.root)
        datasetLabel = ttk.Label(datasetFrame, text = "MNIST Dataset")
        datasetLabel.grid(row = 0, column = 0, sticky = "n")

        self.trainingLabelFrame = ttk.Frame(datasetFrame)
        self.trainingLabelFrame.grid(row = 2, column = 0)
        
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight = 1)
        self.root.columnconfigure(2, weight = 2)
        
        self.root.rowconfigure(0, weight = 1)
        self.root.rowconfigure(1, weight = 1)
        self.root.rowconfigure(2, weight = 1)
        self.root.rowconfigure(3, weight = 1)

        parameterFrame = ttk.Frame(self.root)
        trainingFrame = ttk.Frame(self.root )
        metricFrame = ttk.Frame(self.root)

        parameterFrame.grid(row=0, column=0, sticky="n" + "w", rowspan = 2)
        datasetFrame.grid(row=0, column=1, sticky = "n")
        trainingFrame.grid(row=0, column=2, sticky="n" + "e", rowspan = 2)
        metricFrame.grid(row = 1, column = 1)
        

        #combobox for selecting ml learning model
        mlComboBox = ttk.Combobox(parameterFrame, state = "readonly")
        mlComboBox['values'] = ['knn', 'randomForest']
        mlComboBox.current(0)

        mlComboBox.grid(column = 0, row = 0)
        mlComboBox.bind('<<ComboboxSelected>>', lambda e: self.switchMLmodel(e, mlComboBox.get(), parameterFrame))
        
        
        self.dc = DigitCanvas(metricFrame)

        self.pr = ParameterReader(parameterFrame,'knn')
        self.pr.grid(row=1, column = 0)
        
        self.gf = GraphFrame(metricFrame)
        
        self.mlc = ML_Console(trainingFrame)
        self.mlc.grid(row = 2, column = 0)
        self.mlc.addText("Welcome to the Machine Learning App!")
        
        
        self.mlFunc = ML_Function(pr = self.pr, dc = self.dc)
        
        self.displayTrainingValues(self.trainingLabelFrame, self.mlFunc)
        
        trainBtn = ttk.Button(trainingFrame, text = "Train ML Model", command = lambda:self.train(self.mlFunc))
        testBtn = ttk.Button(trainingFrame, text = "Test ML Model", command = lambda:self.test(self.mlFunc))
        self.drawBtn = ttk.Button(trainingFrame, text = "Draw your own Digit!", command = lambda:self.drawDigit(self.drawBtn.cget('text')))
        
        trainBtn.grid(column = 0, row = 0)
        testBtn.grid(column = 0, row = 1)
        self.drawBtn.grid(column = 0, row = 3)
        
        self.generateMenus()
        
        self.root.mainloop()
        
    def generateMenus(self):
        '''
        Generates the Menus for this GUI
        '''
        #create menu bar
        mainMenuBar = Menu(self.root)
        fileMenu = Menu(mainMenuBar)
        graphMenu = Menu(mainMenuBar)
        mainMenuBar.add_cascade(menu = fileMenu, label = "File")
        mainMenuBar.add_cascade(menu = graphMenu, label = "Graph")
        self.root.config(menu = mainMenuBar)
        
        fileMenu.add_command(label = "Quit", command = lambda: self.exitGui())
        
        graphMenu.add_command(label = "ROC Curve", command = lambda: self.graphROC())        
        
    def exitGui(self):
        '''
        Exits the GUI
        '''
        self.root.quit()
        
    def train(self, mlFunc, preprocess = False):
        '''
        Trains the selected Machine Learning Model. Catches exceptions and displays error messages
        in the GUI console-box and as a pop-up message box
        Parameters:
        mlFunc: ML_Function class
        '''
        self.resetGraphFrame()
        try:
            self.mlc.addText("Status: Training " + mlFunc.getMLmodelName() + " model.")
            
            self.currentParamVals, self.currentParamLabels = mlFunc.runMLtraining()
            
            self.displayTrainingValues(self.trainingLabelFrame, mlFunc )
            self.mlc.addText("Update: " + mlFunc.getMLmodelName() + " model trained.")
            messagebox.showinfo(title = "Update", message = mlFunc.getMLmodelName() + " Successfully trained.")
            return self.currentParamVals, self.currentParamLabels
        except ValueError as e:
            self.mlc.addText("Update: Training failed.")
            self.mlc.addError(str(e))
            messagebox.showerror(message=str(e), title = "Error")
            
    def test(self, mlFunc):
        '''
        Tests the selected and trained Machine Learning Model. Catches exceptions and displays error messages
        in the GUI console-box and as a pop-up message box
        Parameters:
        mlFunc: ML_Function class
        '''
        self.mlc.addText("Status: Testing " + mlFunc.getMLmodelName() + " model.")
        try:
            precision, recall, f1, fig = mlFunc.runMLtesting()
            
            self.mlc.addText("Update: " + mlFunc.getMLmodelName() + " model testing completed.")
            self.mlc.addText("Precision: " + str(precision))
            self.mlc.addText("Recall: " + str(recall))
            self.mlc.addText("F1 Score: " + str(f1))
            self.gf.set_mplFig(fig)
            self.graphROC()
            messagebox.showinfo(title = "Update", message = "Testing Complete!")
        except ValueError as e:
            self.mlc.addText("Update: Testing Failed.")
            self.mlc.addError(str(e))
            messagebox.showerror(message=str(e), title = "Error")
    
    def drawDigit(self, currentText):
        '''
        Method for Bringing up the DigitCanvas for drawing and classifying a digit.
        Parameters:
        currentText: String. The text on the button
        '''
        if currentText == "Draw your own Digit!":
            #clear all elements from the frame
            self.resetParentFrame(self.dc)            
            self.dc.grid(row = 0, column = 0)
            self.drawBtn.config( text = "Submit your Digit!")
        else:
            self.classifyDigit(self.mlFunc)

    def classifyDigit(self, mlFunc):
        '''
        Classifies the user-drawn digit according to the selected and trained ML model.
        Parameters:
        mlFunc: ML_Function instance.
        '''
        try:
            value = mlFunc.classifyDigit()
            output = "Digit Class: " + str(value)
            
            ynMessage = "Your digit is a: " + str(value) + "\nIs this Right?"
            correctClass = messagebox.askyesno(title = "Digit Classified!", message = ynMessage)
            if correctClass:
                self.mlc.addText(output + " - Correct!")
            else:
                self.mlc.addText(output + " - Incorrect!")
        except ValueError as e:
            self.mlc.addError(str(e))
            messagebox.showerror(message=str(e), title = "Error")
    
    def resetParentFrame(self, unit):
        '''
        Removes all widgets from a tkinter frame.
        Parameters:
        unit: tkinter frame
        '''
        frame = unit.master
        for widget in frame.winfo_children():
            widget.grid_remove()
    
    def resetGraphFrame(self):
        '''
        Erases the GraphFrame containing the ROC Graph
        '''
        self.drawBtn.config(text = "Draw your own Digit!")
        self.resetParentFrame(self.gf)
        self.gf.set_mplFig(None)
    
    def graphROC(self):
        '''
        Displays the graph of the Receiver Operating Characteristic for a
        trained and tested Machine Learning model
        '''
        self.drawBtn.config(text = "Draw your own Digit!")
        self.resetParentFrame(self.gf)
        self.gf.drawFigure(self.gf.mplFig)
        self.gf.grid(row = 0, column = 0)
        
    
    def getSelectedModel(self, combobox):
        '''
        Returns the current ML MOdel combobox selection.
        Parameters:
        combobox: a TKinter combobox widget.
        '''
        return combobox.get()
    
    
    def switchMLmodel(self, event, mlModelName, readerContainer):
        '''
        Switches the selected ML model automatically based on the Combobox selection
        Parameters:
        event: TK inter event for combobox selection.
        mlModelName: the name of the selected ML model
        readerContainer: the tkinter widget (frame) containing the ParameterReader instance.
        '''
        self.resetGraphFrame()
        self.pr.grid_forget()
        self.pr = ParameterReader(readerContainer, mlModelName)
        self.mlFunc.setParamReader(self.pr)
        self.mlFunc.setMLmodel(mlModelName)
        self.displayTrainingValues(self.trainingLabelFrame, self.mlFunc)
        self.pr.grid(row = 1, column = 0)
        self.mlc.addText("Update: ML Model Reset to " + self.mlFunc.getMLmodelName() + ".")
    

    def displayTrainingValues(self, frame, mlFunc):
        '''
        Shows the values of the currently-trained ML model
        as text at the top center of the GUI. Useful for
        user reference.
        Paramters:
        frame - TKinter Frame widget.
        mlFunc - ML_Function instance.
        '''
        #clear all elements from the frame
        for widget in frame.winfo_children():
            widget.destroy()
       
        if mlFunc.getIsTrained():
            paramLabels = mlFunc.getMLparamLabels()
            paramVals = mlFunc.getMLparameters()
            
            ttk.Label(frame, text = "Status:").grid(column = 0, row = 0)
            ttk.Label(frame, text = "TRAINED").grid(column = 1, row = 0)
            ttk.Label(frame, text = "Trained ML Model:").grid(column = 0, row = 1)
            ttk.Label(frame, text = mlFunc.getMLmodelName()).grid(column = 1, row = 1)
               
            for i in range(len(paramLabels)):
                thisLabel = ttk.Label(frame, text = (paramLabels[i] + ":"))
                thisValue = ttk.Label(frame, text = paramVals[i])
                thisLabel.grid(row = i+3, column = 0)
                thisValue.grid(row = i+3, column = 1)
        
        else:
            ttk.Label(frame, text = "Status:").grid(column = 0, row = 0)
            ttk.Label(frame, text = "NOT TRAINED").grid(column = 1, row = 0)

