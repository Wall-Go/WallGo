import numpy as np
import math
import warnings
#import DRwizard
from scipy import integrate,interpolate,optimize,special,linalg,stats
#from cosmoTransitions.finiteT import Jb_spline as Jb
#from cosmoTransitions.finiteT import Jf_spline as Jf


print('kissa')


'''

# for Tuomas to mess up indents ;)

class Dank: 
    #pass
    def __init__(self,arg1,arg2):
        self.arg1 = arg1
        self.arg2 = arg2
    	#pass
    
    def func(self, x,y):
        
       return x+y+self.arg1+self.arg2
       
    	
       
var1 = 1
var2 = 0
dank1 = Dank(var1,var2)

kissa = dank1.func(3,4)

print(kissa)

'''

class Wallspeed: 
    #pass
    def __init__(self,model,grid,coll):
    	self.model = model
		self.grid = grid
        #self.coll = coll
    	pass


'''

# pseudocode example of what user could write, for testing communication between the different parts of the code. 

Import wallgo

# walgo needs to include classes:
# ModelBuild
# InitializeModel
# Collision
# Wallspeed

# this loads model specs from input model (collision related stuff) and DRalgo (veff related stuff)
modelbuild = wallgo.ModelBuild('inputModel.txt','DRalgoModelFile.m')

# InitializeModel builds all functions for a model in question
xSM = wallgo.InitializeModel(modelbuild)

# returns something related to what collision team fiddled with
coll = wallgo.Collision(xSM)

# Wall speed is computed to user specified model parameter space grid. User specifies this gruid here. 
# grid has to be compatible with the model, think about syntax!
# For now, scan only in lamhs, and only in one point lamhs = 1.2. Other points are now hardcoded in.

lamhs = 1.2 
 
grid = [lamhs] # this grid is just a single BM point.

# compute wall speed:
ws = walgo.Wallspeed(xSM, grid, coll)

	
'''





