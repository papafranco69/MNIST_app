'''
Created on Apr 27, 2023

@author: Peter Koropey
'''

'''
This file exists to simplify the process of adding more ML models to this program. It is only a dictionary,
with the key referring to the ML model and the values referring to parameters for that model
It is structured as a list, with the following dimensions/values:
[0]: List - Names (Labels) of Parameters
[1]: List - Names of Parameter input types ("textbox" or "combobox")
[2]: List - Default Paramter Values (requires a list of values for a combobox)
[3]: String - Name of ML Model
Universal Values for Random State Seed and Partion scale are "tacked on" at the end
via the for loop, so programmers do not have to add them.
'''


mlParams = {
    'knn': [ ["Number of Neighbors", "Weight Function"], ["textbox", "combobox"], [5, ['uniform', 'distance']], "K-Nearest Neighbors" ], 
    'randomForest': [ ["Decision Depth", "Number of Estimators"], ["textbox", "textbox"], [10, 10], "Random Forest" ]
    }

#Add Necessary "Universal" Values: Enabling/Disabling of Preprocessing, Random Seed, and Partition Scale
for key in mlParams:
    mlParams[key][0].insert(0, "Enable Preprocessing")
    mlParams[key][0].append("Random State Seed")
    mlParams[key][0].append("Percentage of Dataset\nUsed for Training")
    mlParams[key][1].insert(0, "checkbutton")
    mlParams[key][1].append("textbox")
    mlParams[key][1].append("scale")
    mlParams[key][2].insert(0, 0)
    mlParams[key][2].append(1)
    mlParams[key][2].append(75.0)
