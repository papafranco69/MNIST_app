'''
Created on May 4, 2023

@author: Peter
This script runs test cases 17 through 21 of the test plan.
'''

from utility.ImgMatConv import ImgMatConv
from PIL import Image
imc = ImgMatConv([0])
imc.inputMatrix = imc.imgToMatrix(r'digit7.bmp')
imc.main()
