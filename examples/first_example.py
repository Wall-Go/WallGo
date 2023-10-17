"""
A first example.
"""
import BMxSMsimple

#Reset the nucleation temperature
BMxSMsimple.hydro.Tnucl=100.

# looking at thermodynamics
print("\nThermodynamics:", BMxSMsimple.thermo)
print(f"{BMxSMsimple.thermo.pHighT(100)=}")
print(f"{BMxSMsimple.thermo.pLowT(100)=}")
print(f"{BMxSMsimple.thermo.ddpLowT(100)=}")

# checking Tplus and Tminus
vJ = BMxSMsimple.hydro.vJ
c1, c2, Tplus, Tminus, velocityAtz0 = BMxSMsimple.hydro.findHydroBoundaries(0.59)

print("Jouguet velocity")
print(vJ)

print("c1,c2")
print(c1,c2)

print("\ntop quark:", BMxSMsimple.top)
