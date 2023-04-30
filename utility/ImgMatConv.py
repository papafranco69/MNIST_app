'''
Group 1 Project
CMSC 495-7381
CTPnCS-7381-2232 1

Created on Apr 3, 2023

@author: Peter Koropey
'''

import PIL as pil
import numpy as np


class ImgMatConv:
    '''
    This class has methods for converting an image file to a matrix (using the Pillow library),
    for finding the center of mass of a matrix, for translating that matrix into a new,
    square matrix cetnered about its center of mass, reducing that matrix in resolution
    to 20x20 elements, padding that matrix, and converting a matrix into an array.
    
    Additionally, the matrices and arrays can be written to a textfile for visual verification
    by a user.
    
    This class is used as part of the Group 1 Project, for converting a user-drawn digit
    into an element of data based on the MNIST dataset.
    '''

    def __init__(self, inputMatrix, outputPath = r''):
        '''
        This class has methods for converting an image to an integer matrix,
        for centering an integer matrix about its center of mass, reducing
        the resolution of the matrix, padding matrices, and converting a matrix
        to a 1-D array.
        
        Parameters:
        inputMatrix: numpy 2D integer matrix.
        outputPath: string. Represents the file path textfiles are saved to.
        
        '''
        self.inputMatrix = inputMatrix
        self.outputPath = outputPath


    def main(self):
        '''
        Main function.
        Runs the other methods, and names the files for output.
        '''
        outputPath = self.outputPath
        
        pixelMatrix = self.inputMatrix
        self.writeMatrix(pixelMatrix, outputPath + 'output1.txt')
        
        com = self.findCenterOfMass(pixelMatrix)
        
        centeredMatrix = self.centerMatrix(pixelMatrix, com)
        self.writeMatrix(centeredMatrix, outputPath + 'centered1.txt')
        
        smallMatrix = self.shrinkMatrix(centeredMatrix, 20, 20)
        self.writeMatrix(smallMatrix, outputPath + 'small1.txt')
        
        paddedMatrix = self.padMatrix(smallMatrix, 4)
        self.writeMatrix(paddedMatrix, outputPath + 'padded1.txt')
        
        
        array = self.matrixToArray(paddedMatrix)
        self.writeMatrix(array, outputPath + 'array1.txt')
        
        
        array255 = self.mapTo255(array)
        self.writeMatrix(array255, outputPath + 'array255.txt')
        
        matrix255 = self.mapTo255(paddedMatrix)
        self.writeMatrix(matrix255, outputPath + 'matrix255.txt')
        
        return array

    def imgToArray(self):
        '''
        This runs the methods to center, resize, and pad a matrix
        which is then mapped to an array. The array is then returned.
        This method does not save the matrices/arrays to text files.
        
        This method will be used to export the data for a drawn digit
        to a form compatible with the MNIST dataset and the ML models
        the user can select from. Allows the user-drawn digit to be
        classified.
        
        Returns:
        array255: numpy 1D array
        '''
        pixelMatrix = self.inputMatrix
        com = self.findCenterOfMass(pixelMatrix)
        centeredMatrix = self.centerMatrix(pixelMatrix, com)
        smallMatrix = self.shrinkMatrix(centeredMatrix, 20, 20)
        paddedMatrix = self.padMatrix(smallMatrix, 4)
        array = self.matrixToArray(paddedMatrix)
        array255 = self.mapTo255(array)

        return array255

    def imgToMatrix(self, filePath):
        '''
        Uses the pillow library to convert a black-and-white image file
        into a 2D numpy matrix where white pixels correspond to elements witha  value of 0,
        and black pixels to elements with a value of 1.
        
        Parameters:
        filePath = the path of the image file being read.
        '''
        try:
            img = pil.Image.open(filePath)
            
            imgArray = np.array(img);

            pixelMatrix = np.array(np.zeros((imgArray.shape[0], imgArray.shape[1]), dtype=int))
            for i in range(imgArray.shape[0]):
                for j in range(imgArray.shape[1]):    
                    if imgArray[i][j][0] == 255:
                        pixelMatrix[i][j] = 0
                        
                    else:
                        pixelMatrix[i][j] = 1
            return pixelMatrix
            
            
        except FileNotFoundError:
            print('File not found.')
            return -1
    
    
    def writeMatrix(self, matrix, filePath = "matrix.txt"):
        '''
        Outputs the contents of an integer matrix to a text file, with
        a single whitespace between each element in a row. New rows are
        begun on a new line in the text file.
        
        Parameters:
        matrix: numpy integer array of 1D or 2D
        filePath: string. The output path for the textfile.
        '''
        try:
            with open(filePath, 'w+', encoding="utf-8") as output:
                if (len(matrix.shape) == 2):
                    for i in range(matrix.shape[0]):
                        line = ''
                        for j in range(matrix.shape[1]):
                            line = line + str(matrix[i][j]) + ' ';
                        print(line, file = output)
                elif (len(matrix.shape) == 1):
                    line = ''
                    for i in range(len(matrix)):
                        line = line + str(matrix[i]) + ' '
                    print(line, file = output)
                else:
                    raise ValueError("Invalid dimensions")
        except FileNotFoundError:
            print("Invalid File Path")

    
    def matrixToArray(self, matrix):
        '''
        Converts a matrix of "i" rows and "j" elements to an array
        of 1 row and i*j elements in row-major order.
        
        Paramters:
        matrix: numpy 2D integer matrix.
        
        Returns:
        array: a numpy 1D integer array.
        '''
        array = np.zeros(matrix.shape[0] * matrix.shape[1], dtype=int)
        x = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                array[x] = matrix[i][j]
                x +=1
        return array
    
    
    def mapTo255(self, matrix):
        '''
        Maps all values of "1" in the input array/matrix to the value
        of "255" to make the input array compatible with elements of the
        MNIST dataset (which have values of 0 for "no writing" and
        values between 1 and 255 for "writing". "255" in this case means
        "absolute writing" i.e., black ink/pixels.
        
        Parameters:
        matrix: numpy array (either 1D or 2D)
        
        Returns:
        outputMatrix: numpy array of same dimensions
        '''
        if (len(matrix.shape) == 2):
            outputMatrix = np.zeros((matrix.shape[0], matrix.shape[1]), dtype=int)
            for i in range(matrix.shape[0]):
                for j in range(matrix.shape[1]):
                    if matrix[i][j] > 0:
                        outputMatrix[i][j] = 255
            return outputMatrix
        
        elif(len(matrix.shape) == 1):
            outputMatrix = np.zeros((matrix.shape[0]), dtype=int)
            for i in range(matrix.shape[0]):
                if matrix[i] > 0:
                    outputMatrix[i] = 255
            return outputMatrix
        else: 
            raise ValueError("Invalid Dimensions")
                
    
    def findCenterOfMass(self, matrix):
        '''
        Calculates the center of mass of an integer matrix.
        Center of mass along an axis is found by multiplying the 
        position (along that axis) of an element by its value, and
        adding that factor to the corresponding factor for every other value
        along that axis.
        
        Parameters:
        matrix: a numpy 2D matrix
        
        Returns:
        com: Integer list, denoting the center of mass as an index [i][j].
        '''
        com = [0, 0]
        totalSum = 0
        iSum = 0
        jSum = 0
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                totalSum += matrix[i][j]
                iSum += matrix[i][j] * i
                jSum += matrix[i][j] * j
        
        try:
            com[0] = int(iSum / totalSum);
            com[1] = int(jSum / totalSum);
        except ValueError:
            raise ValueError("ImgMatConv.findCenterOfMass: Cannot divide by 0")
        
        return com
        
    def findIndexMinMax(self, matrix):
        '''
        Finds the minimum and maximum value along each axis
        in a matrix.
        
        Parameters:
        matrix: numpy 2D matrix.
        
        Returns:
        iMin: minimum along the "i" axis
        iMax: maximum along the "i" axis
        jMin: minimum along the "j" axis
        jMax: maximum along the "j" axis
        '''
        iMin = 1000000000
        iMax = -1
        jMin = 1000000000;
        jMax = -1;
        for i in range(matrix.shape[0]):
            for j in range (matrix.shape[1]):
                if matrix[i][j] > 0:
                    if iMin > i:
                        iMin = i
                    if jMin > j:
                        jMin = j
                    if iMax < i:
                        iMax = i
                    if jMax < j:
                        jMax = j
        
        return iMin, iMax, jMin, jMax
    
    
    def centerMatrix(self, matrix, com):
        '''
        Centers the non-zero elements of the argument matrix
        in a new, square matrix about the argument matrix's center of mass.
        
        Paramters:
        matrix: numpy 2D integer matrix.
        com: List of integers. Denotes the center-of-mass of "matrix" argument.
        
        Returns:
        outputMatrix: numpy 2D integer matrix.
        '''
        iCoord = com[0]
        jCoord = com[1]
        origDimI = matrix.shape[0]
        origDimJ = matrix.shape[1]
        
        
        iMin, iMax, jMin, jMax = self.findIndexMinMax(matrix)

        iLength = 0
        jLength = 0
        iRatio = 1
        jRatio = 1
        
        iDistComMin = iCoord - iMin
        jDistComMin = jCoord - jMin
        
        
        #offset is distance from CoM to longer side, minus distance from CoM to matrix end
        if (iMax - iCoord) > (iCoord - iMin):
            iLength = 2*(iMax - iCoord)
        else:
            iLength = 2 * (iCoord - iMin)
        
        if (jMax - jCoord) > (jCoord - jMin):
            jLength = 2*(jMax - jCoord)
        else:
            jLength = 2 * (jCoord - jMin)
            
        if (iLength > jLength):
            jRatio = (origDimJ * iLength) / (jLength * origDimI)
            jLength = int(jLength * jRatio)
        else:
            iRatio= (origDimI * jLength) / (origDimJ * iLength)
            iLength = int(iLength * iRatio)
        
        outputMatrix = np.array(np.zeros((iLength, jLength), dtype=int))

        iStart = int((iLength / 2 - iDistComMin))
        jStart = int((jLength / 2 - jDistComMin))

        jTemp = jStart
        for i in range (iMin, iMax):
            for j in range (jMin, jMax):
                outputMatrix[iStart][jStart] = matrix[i][j]
                jStart += 1
            iStart += 1
            jStart = jTemp
            
        return outputMatrix


    def shrinkMatrix(self, matrix, smallI, smallJ):
        '''
        Reduces the resolution of an integer matrix to
        new dimensions [smallI][smallJ]. Because the resolution
        is reduced, some information from the original matrix may be lost.
        
        Parameters:
        matrix: numpy 2D integer matrix.
        smallI: integer: magnitude along the "i" axis of the output matrix
        smallJ: integer: magntiude along the "j" axis of the output matrix
        
        Returns:
        output: numpy 2D integer matrix
        '''
        bigI = matrix.shape[0]
        bigJ = matrix.shape[1]
        
        iRatio = int(bigI/smallI)
        jRatio = int(bigJ/smallJ)
        
        output = np.array(np.zeros((smallI, smallJ), dtype=int))
        
        
        for iCounter in range(0, smallI):
            for jCounter in range(0, smallJ):
                
                sum = 0
                '''
                for i in range(iCounter * iRatio, iRatio * (iCounter+1)):
                    for j in range(jCounter * jRatio, jRatio * (jCounter+1)):
                        
                        sum += matrix[i][j]
                '''
                output[iCounter][jCounter] = self.shrinkMatrixInner(iCounter, jCounter, iRatio, jRatio, matrix)
                
                '''
                if (sum > 0):
                    output[iCounter][jCounter] = 1
                else:
                    output[iCounter][jCounter] = 0
                ''' 
        return output
    
    def shrinkMatrixInner(self, iCounter, jCounter, iRatio, jRatio, matrix):
        '''
        This function exists to prevent needless iterations of the matrix.
        As soon as a 1 is found, iteration of the sub-matrix immediately ends.
        
        Parameters:
        iCounter, jCounter: Int
        iRatio, jRatio: Int
        matrix: numpy 2D int matrix
        
        Returns: Int (1 or 0)
        '''
        for i in range(iCounter * iRatio, iRatio * (iCounter+1)):
            for j in range(jCounter * jRatio, jRatio * (jCounter+1)):
                if matrix[i][j] == 1:
                    return 1
        return 0
    
    def padMatrix(self, matrix, padSize=4):
        '''
        Pads a matrix's edges with rows and columns of 0s.
        
        Parameters:
        matrix: numpy 2D integer matrix
        padSize: the number of rows/columns of 0s added to each side.
        (Example: padsize=1 turns "[1,1],[1,1]" into 
        [0,0,0,0],[0,1,1,0],[0,1,1,0],[0,0,0,0]
        
        Returns:
        output: numpy 2D integer matrix
        '''
        iLength = matrix.shape[0]
        jLength = matrix.shape[1]
        
        output = np.array(np.zeros((iLength + 2*padSize, jLength + 2*padSize), dtype=int))
        
        
        for i in range(iLength):
            for j in range(jLength):
                output[i + padSize][j + padSize] = matrix[i][j]
        
        return output
        