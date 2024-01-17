"""
A first example.
"""
import BMxSM

print("Testing against Benoit's earlier results")

# looking at thermodynamics
print("\nThermodynamics:", BMxSM.thermo)
print(f"{BMxSM.thermo.pHighT(100)=}")
print(f"{BMxSM.thermo.pLowT(100)=}")
print(f"{BMxSM.thermo.ddpLowT(100)=}")

# checking Tplus and Tminus
print(f"{BMxSM.fxSM.findPhases(100.1)=}")
print(f"{BMxSM.fxSM.findPhases(103.1)=}")
print(f"{BMxSM.fxSM.findTc()=}")

vJ = BMxSM.hydro.vJ
c1, c2, Tplus, Tminus, velocityAtz0 = BMxSM.hydro.findHydroBoundaries(0.5229)

print("Jouguet velocity")
print(vJ)
print(BMxSM.thermo.pLowT(100.1))
print(BMxSM.thermo.pHighT(103.1))

print("c1,c2")
print(c1,c2)

print("\ntop quark:", BMxSM.top)
