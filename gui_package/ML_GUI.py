'''
Anishka Forbes and Peter Koropey
'''

from tkinter import *
from tkinter import ttk
from gui_package.DigitCanvas import DigitCanvas
from gui_package.ParameterReader import ParameterReader
from gui_package.GraphFrame import GraphFrame
from utility.ML_Function import ML_Function
from tkinter import messagebox

class ML_GUI(object):

    def __init__(self):
        self.root = Tk()
        self.root.title('Machine Learning Graphical Tool')
        #self.root.geometry('{}x{}'.format(800, 800))
        
        # create containers
        datasetFrame = ttk.Frame(self.root)
        datasetLabel = ttk.Label(datasetFrame, text = "MNIST Dataset")
        datasetLabel.grid(row = 0, column = 0, sticky = "n")

        
        #self.root.rowconfigure(0, weight=1)
        self.root.columnconfigure(0, weight=1)
        self.root.columnconfigure(1, weight = 1)
        self.root.columnconfigure(2, weight = 2)



        # create widgets
        #datasetFrame.grid_rowconfigure(0, weight=1)
        #datasetFrame.grid_columnconfigure(1, weight=1)

        parameterFrame = ttk.Frame(self.root)
        trainingFrame = ttk.Frame(self.root )
        metricFrame = ttk.Frame(self.root)

        parameterFrame.grid(row=0, column=0, sticky="n" + "w")
        datasetFrame.grid(row=0, column=1, sticky = "n")
        trainingFrame.grid(row=0, column=2, sticky="n" + "e")
        metricFrame.grid(row = 1, column = 1)
        
        
        
        #combobox for selecting ml learning model
        mlComboBox = ttk.Combobox(parameterFrame, state = "readonly")
        mlComboBox['values'] = ['knn', 'randomForest']
        mlComboBox.current(0)
        mlComboBox.grid(column = 0, row = 0)
        mlComboBox.bind('<<ComboboxSelected>>', lambda e: self.switchMLmodel(e, mlComboBox.get(), parameterFrame))
        
        
        self.dc = DigitCanvas(metricFrame)
        #self.dc.grid(row = 0, column = 0)

        
        self.pr = ParameterReader(parameterFrame,'knn')
        self.pr.grid(row=1, column = 0)
        
        
        
        #parameterFrameInner = ttk.Frame(parameterFrameOuter)
        #self.pr = ParameterReader(parameterFrameInner, 'knn')
        #parameterFrameInner.grid(column = 0, row = 1)
        
        
        
        
        self.mlFunc = ML_Function(pr = self.pr, dc = self.dc)
        
        
        trainBtn = ttk.Button(trainingFrame, text = "Train ML Model", command = lambda:self.train(self.mlFunc))
        testBtn = ttk.Button(trainingFrame, text = "Test ML Model", command = lambda:self.test(self.mlFunc))
        self.drawBtn = ttk.Button(trainingFrame, text = "Draw your own Digit!", command = lambda:self.drawDigit(self.drawBtn.cget('text')))
        
        trainBtn.grid(column = 0, row = 0)
        testBtn.grid(column = 0, row = 1)
        self.drawBtn.grid(column = 0, row = 2)
        
        self.generateMenus()
        
        self.root.mainloop()
        
    def generateMenus(self):
        #create menu bar
        mainMenuBar = Menu(self.root)
        fileMenu = Menu(mainMenuBar)
        graphMenu = Menu(mainMenuBar)
        mainMenuBar.add_cascade(menu = fileMenu, label = "File")
        mainMenuBar.add_cascade(menu = graphMenu, label = "Graph")
        self.root.config(menu = mainMenuBar)
        
        fileMenu.add_command(label = "Quit", command = lambda: self.exitGui())
        
        graphMenu.add_command(label = "Accuracy", command = lambda: self.graphAccuracy())        
        
    def exitGui(self):
        self.root.quit()
        
    def train(self, mlFunc):
        try:
            self.currentParamVals, self.currentParamLabels = mlFunc.runMLtraining()
            messagebox.showinfo(title = "Update", message = "ML Model Successfully trained.")
            #print(self.currentParamVals, self.currentParamLabels)
            return self.currentParamVals, self.currentParamLabels
        except ValueError as e:
            messagebox.showerror(message=str(e), title = "Error")
            
    def test(self, mlFunc):
        try:
            mlFunc.runMLtesting()
            messagebox.showinfo(title = "Update", message = "Testing Complete!")
        except ValueError as e:
            messagebox.showerror(message=str(e), title = "Error")
    
    def drawDigit(self, currentText):
        if currentText == "Draw your own Digit!":
            self.dc.grid(row = 0, column = 0)
            self.drawBtn.config( text = "Submit your Digit!")
        else:
            self.classifyDigit(self.mlFunc)


    
    def classifyDigit(self, mlFunc):
        try:
            value = mlFunc.classifyDigit()
            ynMessage = "Your digit is a: " + str(value) + "\nIs this Right?"
            messagebox.askyesno(title = "Digit Classified!", message = ynMessage)
        except ValueError as e:
            messagebox.showerror(message=str(e), title = "Error")
    
    
    def graphAccuracy(self):
        self.drawBtn.config(text = "Draw your own Digit!")
        #stub function, doesn't do anything yet
        pass
    
    def getSelectedModel(self, combobox):
        #returns the value in a combobox
        return combobox.get()
    
    
    def switchMLmodel(self, event, mlModelName, readerContainer):
        self.pr.grid_forget()
        self.pr = ParameterReader(readerContainer, mlModelName)
        self.mlFunc.setParamReader(self.pr)
        self.mlFunc.setMLmodel(mlModelName)
        self.pr.grid(row = 1, column = 0)
    
mlgui = ML_GUI()
