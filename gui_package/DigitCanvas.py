'''
Group 1 Project
CMSC 495-7381
CTPnCS-7381-2232 1

Created on Apr 5, 2023

@author: Peter Koropey
'''
from tkinter import Canvas
from tkinter import ttk
import numpy as np
from utility.ImgMatConv import ImgMatConv
from tkinter import messagebox

class DigitCanvas(ttk.Frame):
    '''
    This class creates a frame with a canvas on it.
    A user can draw with an 11x11 pixel black box by clicking and dragging.
    The user is meant to draw a single numerical digit.
    As the user draws, a parallel matrix with the same dimensions as the canvas
    but filled with 0s has its corresponding elements change to 1s.
    This matrix is then centered (according to the center of mass), resized, padded
    and converted to an array using the ImgMatConv class. This makes the drawing
    compatible with the MNIST dataset and accordingly classifiable.
    '''


    def __init__(self, container, outputPath = r''):
        '''
        DigitCanvas. Creates a cnavas for drawing a digit,
        convertible to a matrix and then to an array for 
        classification by a machine learning model.
        
        Paramters:
        container: the parent container (a Tkinter frame)
        outputPath: String. the file path for outputting matrix conversions.
            Default is the directory the files are run from. 
        
        Attributes:
        root: the parent container.
        globalPixelMatrix: the matrix parallel to the canvas
        digitArray: an array, based on the resized pixel matrix.
        hasDrawn: a flag to prevent divide by zero errors. 
            Becomes True if the user has drawn something on the canvas.
            Becomes False when the canvas is cleared.
        canvas: the canvas parallel to the globalPixelMatrix, on which
            the user draws.
        
        '''
        super().__init__(master = container)
        self.root = self
        border = ttk.Labelframe(self.root)
        utilFrame = ttk.Frame(border)
        
        self.globalPixelMatrix = np.array(np.zeros((500, 500), dtype=int))

        self.outputPath = outputPath
        
        self.digitArray = np.zeros(784, dtype = int)
        
        self.hasDrawn = False
        
        self.canvas = Canvas(border, width = 500, height = 500, background = 'white')
        self.canvas.grid(column = 0, row = 0)
        
        self.canvas.bind("<Button-1>", lambda e :self.draw(event = e, localPixelMatrix=self.globalPixelMatrix))
        self.canvas.bind("<B1-Motion>", lambda e :self.draw(event = e, localPixelMatrix=self.globalPixelMatrix))
        
        border.pack()
        
        utilFrame.grid(column = 0, row = 1)
        
        
        clearBtn = ttk.Button(utilFrame, text = "Clear Canvas", command = lambda: self.clear_canvas(self.globalPixelMatrix))
        clearBtn.grid(column = 0, row = 0)
        


    def resetPixelMatrix(self, localPixelMatrix):
        '''
        Restores every element in the input matrix to a 0.
        
        Paramters: 
        localPixelMatrix: numpy 2D matrix.
        
        returns:
        localPixelMatrix: numpy 2D matrix.
        '''
        for i in range(localPixelMatrix.shape[0]):
            for j in range(localPixelMatrix.shape[1]):
                localPixelMatrix[i][j] = 0
        return localPixelMatrix
    

    
    def draw(self, event, localPixelMatrix):
        '''
        Lets the user draw on the canvas.
        Changes the values of the localPixelMatrix from 0s to 1s
        corresponding to where the user drew on the canvas.
        
        Parameters:
        event: a tkinter event (a mouse click or drag)
        localPixelMatrix: the input numpy matrix.
        '''
        
        #Draws an 11x11 rectangle around the pixel the user clicks/drags
        self.canvas.create_rectangle((event.x-5, event.y-5, event.x+5, event.y+5), width=10)
        
        iMax = localPixelMatrix.shape[0]
        jMax = localPixelMatrix.shape[1]       
        
        
        #Changes the values in localPixelMatirx from 0s to 1s around the element
        #corresponding to the canvas-pixel on which the user clicked.
        for i in range(int(event.y)-5, int(event.y)+6):
            for j in range(int(event.x)-5, int(event.x)+6):
                #added the extra condition so short-circuit logic prevents extra operations
                #It might not actually matter, however.
                if 0 <= i < iMax and 0 <= j < jMax and localPixelMatrix[i][j] != 1:
                    localPixelMatrix[i][j] = 1
        
        self.hasDrawn = True
    
    
    def clear_canvas(self, localPixelMatrix):
        '''
        Clears the canvas of drawings, and calls the function
        to reset the parallel matrix.
        
        Parameters:
        localPixelMatirx: numpy 2D matrix. 
        '''
        self.hasDrawn = False
        self.canvas.delete('all')
        self.globalPixelMatrix = self.resetPixelMatrix(localPixelMatrix)
                    
    
    def convertMatrixToTextFile(self, inputMatrix, hasDrawn):
        '''
        Converts the numpy matrix parallel/corresponding to the matrix
        into text files so the user can see what is going on. The text file
        conversion was mainly for testing. As this project approaches completion,
        this method will be rewritten to feed the converted matrix/array directly
        to the machine learning model for classification.
        
        
        Parameters:
        inputMatrix: a numpy 2D integer matrix.
        hasDraw: boolean
        '''
        if hasDrawn:
            try:
                imc = ImgMatConv(self.globalPixelMatrix, self.outputPath)
                self.digitArray = imc.main()
                self.classifyDigit(self.digitArray)
            except ValueError as e:
                messagebox.showerror(message=str(e), title = "Error")
        else:
            messagebox.showerror(message="You must first Draw a Digit!", title = "Error")

        
    def deriveArray(self):
        '''
        Generates the numpy 1D integer array corresponding to the centered. 28x28 pixel
        version of the user-drawn digit. 
        Returns:
        1D Numpy int array
        
        Throws exceptions if conditions are not met.
        '''
        if self.hasDrawn:
            try:
                imc = ImgMatConv(self.globalPixelMatrix, self.outputPath)
                self.digitArray = imc.imgToArray()
                
                return self.digitArray
            except ValueError as e:
                raise ValueError(e)

        else:
            raise ValueError("You must first Draw a Digit!")

    
    def getDigitArray(self):
        '''
        Getter method for the numpy 1D int array corresponding to centered,
        reduced version of drawing.
        Returns:
        1D Numpy int array (784 elements)
        '''
        return self.digitArray