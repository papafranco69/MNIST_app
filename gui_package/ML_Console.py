'''
Created on Apr 21, 2023

@author: Peter
'''

import tkinter as tk
from tkinter import ttk

class ML_Console(tk.Text):
    '''
    This class simplifies creating a text console which informs the user what the 
    ML GUI application is doing. Class is a child of tk.Text (a text box).
    '''


    def __init__(self, container):
        '''
        Constructor for the ML_Console class.
        Parameters:
        container: a tkinter frame
        '''
        super().__init__(master = container, width = 35, height = 15, wrap = "word")
        self.root = self

        
        
        scrollBar = ttk.Scrollbar(self.root, orient = 'vertical', command = self.yview)
        self['yscrollcommand'] = scrollBar.set        
        self.config(state = 'disabled')
        
    def addText(self, text):
        '''
        Adds text the console.
        Paramters:
        text: String. The text added to the console.
        '''
        self.config(state = 'normal')
        self.insert('end', text+'\n')
        self.config(state = 'disabled')
    
    def addError(self, text):
        '''
        Adds an error message to the console. Prefaced by "Error" and 
        appears in Red color text.
        Paramters:
        text: String. The error message.
        '''
        self.config(state = 'normal')
        self.insert('end', "Error: "+text+'\n', ('error'))
        self.tag_configure('error', foreground = 'red')
        self.config(state = 'disabled')
