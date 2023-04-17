'''
Created on Apr 14, 2023

@author: Peter Koropey
'''

from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)


class GraphFrame(ttk.Frame):
    '''
    classdocs
    '''


    def __init__(self, container, mplFig = None):
        '''
        Constructor
        '''
        super().__init__()
        self.root = container
        self.mplFig = mplFig
        self.drawFigure(self.mplFig)

        
        
    def drawFigure(self, mplFig):
        canvas = FigureCanvasTkAgg(mplFig, master = self.root)
        canvas.draw()
        canvas.get_tk_widget().pack( )