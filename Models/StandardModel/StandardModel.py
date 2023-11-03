import numpy as np

## WallGo imports
from WallSpeed import GenericModel
from WallSpeed import Particle
from WallSpeed import EffectivePotential
from WallSpeed import WallGoManager


class EffectivePotentialSM(EffectivePotential):

    def __init__(self, modelParameters: dict[str, float]):
        super().__init__(modelParameters)
        ## ... do SM specific initialization here. The super call already gave us the model params

    
    def evaluate(self, fields: np.ndarray[float], temperature: float) -> complex:
        v = fields[0] # phi ~ 1/sqrt(2) (0, v)
        T = temperature

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)
        
        msq = thermalParameters["msq"]
        lam = thermalParameters["lambda"]

        # tree level potential
        V0 = 0.5 * msq * v**2 + 0.25 * lam * v**4

        ## @todo should have something like a static class just for defining loop integrals. NB: m^2 can be negative for scalars
        J3 = lambda msq : -(msq + 0j)**(3/2) / (12.*np.pi) * T # keep 4D units

        ## Cheating a bit here and just hardcoding gauge/scalar masses in SM
        mWsq = thermalParameters["g2"]**2 * v**2 / 4.
        mZsq = (thermalParameters["g1"]**2 + thermalParameters["g2"]**2) * v**2 / 4.
        mHsq = msq + 3*lam*v**2
        mGsq = msq + lam*v**2
    
        # NLO 1-loop correction in Landau gauge, so g^3. Debyes are integrated out by getThermalParameters
        V1 = 2*(3-1) * J3(mWsq) + (3-1) * J3(mZsq) + J3(mHsq) + 3.*J3(mGsq)

        VTotal = V0 + V1
        return VTotal
    
    
    ## Calculates thermally corrected parameters to use in Veff. So basically 3D effective params but keeping 4D units
    def getThermalParameters(self, temperature: float) -> dict[str, float]:
        T = temperature
        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]
        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]
        ## LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["msq"] = msq + T**2 / 16. * (3. * g2**2 + g1**2 + 4.*yt**2 + 8.*lam)

        # how many Higgs doublets / fermion generations
        Nd = 1
        Nf = 3

        ## Debye masses squared (U1, SU2) 
        mDsq1 = g1**2 * T**2 * (Nd/6. + 5.*Nf/9.)
        mDsq2 = g2**2 * T**2 * ( (4. + Nd) / 6. + Nf/3.)
        mD1 = np.sqrt(mDsq1)
        mD2 = np.sqrt(mDsq2)

        ## Let's also integrate out A0/B0
        h3 = g2**2 / 4.
        h3p = g2**2 / 4.
        h3pp = g2*g1 / 2.

        thermalParameters["msq"] += -1/(4.*np.pi) * T * (3. * h3 * mD2 + h3p * mD1)
        thermalParameters["lambda"] += -1/(4.*np.pi) * T * (3.*h3**2 / mD2 + h3p**2 / mD1 + h3pp**2 / (mD1 + mD2))

        # skipping corrections to gauge couplings because those are not needed at O(g^3)

        return thermalParameters




## V = msq |phi|^2 + lambda (|phi|^2)^2
class StandardModel(GenericModel):

    particles = np.array([], dtype=Particle)
    outOfEquilibriumParticles = np.array([], dtype=Particle)


    def __init__(self):

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.Veff = EffectivePotentialSM(self.modelParameters)

        ## Define particles. this is a lot of clutter, especially if the mass expressions are long, 
        ## so @todo define these in a separate file? 
        
        # NB: particle multiplicity is pretty confusing because some internal DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately define up, down, charm, strange, bottom 
        
        ## === Top quark ===
        topMsqVacuum = lambda fields: 0.5 * self.modelParameters["yt"]**2 * fields[0]**2
        topMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

        topQuark = Particle("top", 
                            msqVacuum = topMsqVacuum,
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(topQuark)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

        lightQuark = Particle("lightQuark", 
                            msqVacuum = 0.0,
                            msqThermal = lightQuarkMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 5
        )
        self.addParticle(lightQuark)

        ## === SU(3) gluon ===
        gluonMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = 0.0,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            multiplicity = 1
        )
        self.addParticle(gluon)


    ## Define parameter dict here. @todo correct values, these are from top of my head.
    ## In practice the user would define a function that computes these from whatever their input is (mW, mt, mH etc).
    ## But this still needs to be initialized here as required by the interface. modelParameters = None is fine.
    modelParameters = {
        "RGScale" : 91, # MS-bar scale. Units: GeV
        "yt" : 1.0, # Top Yukawa
        "g1" : 0.3, # U1 coupling
        "g2" : 0.6,  # SU2 coupling
        "g3" : 1.4,  # SU3 coupling
        "lambda" : 0.13,
        "msq" : -7000 # Units: GeV^2
    }
    
    ## @todo kinda would want to have these as individual member variables for easier access. 
    ## But that alone is not good enough as we need to pass the params to other things like the collision module,
    ## and for that we want some common routine that does not involve hardcoding parameter names. So I anticipate that 
    ## some combination of these approaches would be good.

# end StandardModel



def main():

    model = StandardModel()

    # test / example
    mH = 60
    model.modelParameters["msq"] = -mH**2 / 2.
    model.modelParameters["lambda"] = 0.5 * mH**2 / 246.**2

    Tn = 200

    userInput = {
        "Tn" : Tn,
        "phaseLocation1" : [ 0.0 ],
        "phaseLocation2" : [ 246.0 ]
    }

    ## Create control class
    manager = WallGoManager(model, userInput)

    # At this point we should have all required input from the user
    # and the manager should have validated it, found phases etc.
    
    print(manager.phaseLocation1)
    print(manager.phaseLocation2)




## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()