'''
Created on Apr 14, 2023

@author: Peter Koropey
'''

from tkinter import ttk
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg)


class GraphFrame(ttk.Frame):
    '''
    Class is used to display MatPlotLib charts directly onto a TKinter GUI.
    '''


    def __init__(self, container, mplFig = None):
        '''
        This class displays a MatPlotLib chart directly onto a TKinter GUI.
        
        Parameters:
        container: TKinter Parent Container (likely a Frame)
        mplFig: MatPlotLib.pyplot plot/figure
        '''
        super().__init__()
        self.root = container
        self.mplFig = mplFig
        self.drawFigure(self.mplFig)

        
        
    def drawFigure(self, mplFig):
        '''
        This method draws a matplotlib figure directly onto a TKinter canvas.
        
        Paramters:
        mplFig: MatPlotLib.pyplot plot/figure
        '''
        canvas = FigureCanvasTkAgg(mplFig, master = self.root)
        canvas.draw()
        canvas.get_tk_widget().pack( )