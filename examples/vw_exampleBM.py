"""
Computing the wall velocity in the xSM
"""
import BMxSM
from time import time


#vwLTE = BMxSM.hydro.findvwLTE()

#print(f'Jouguet velocity: {BMxSM.hydro.vJ}')
#print("The wall velocity in local thermal equilibrium is")
#print(vwLTE)

"""
Compute the wall velocity with out-of-equilibrium effects
"""
#print(eom.findWallVelocityLoop())
t = time()
print(BMxSM.eom.findWallVelocityMinimizeAction())
print(BMxSM.eomOffEq.findWallVelocityMinimizeAction())
print(time()-t)
