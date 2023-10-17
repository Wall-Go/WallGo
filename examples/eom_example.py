"""
An attempt to run EOM.py 
"""
import BMxSMsimple
from WallSpeed.EOM import EOM
"""
Compute the wall velocity in local thermal equilibrium
"""
vwLTE = BMxSMsimple.hydro.findvwLTE()
print("The wall velocity in local thermal equilibrium is")
print(vwLTE)

"""
Compute the wall velocity from the loop without out-of-equilibrium effects
"""
print("The wall parameters without out-of-equilibrium effects, found by minimizing the action are")
print(BMxSMsimple.eom.findWallVelocityMinimizeAction())

