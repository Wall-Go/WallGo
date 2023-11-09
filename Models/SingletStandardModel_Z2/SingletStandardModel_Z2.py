import numpy as np

## WallGo imports
from WallSpeed import GenericModel
from WallSpeed import Particle
from WallSpeed import EffectivePotential
from WallSpeed import WallGoManager


## Z2 symmetric SM + singlet model. V = msq |phi|^2 + lam (|phi|^2)^2 + 1/2 b2 S^2 + 1/4 b4 S^4 + 1/2 a2 |phi|^2 S^2
class SingletSM_Z2(GenericModel):

    particles = np.array([], dtype=Particle)
    outOfEquilibriumParticles = np.array([], dtype=Particle)
    modelParameters = {}


    def __init__(self, initialInputParameters: dict[str, float]):

        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialxSM_Z2(self.modelParameters)

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




    ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        mh1 = inputParameters["mh1"] # 125 GeV
        mh2 = inputParameters["mh2"]

        ## these are direct inputs:
        modelParameters["RGScale"] = inputParameters["RGScale"]
        modelParameters["a2"] = inputParameters["a2"]
        modelParameters["b4"] = inputParameters["b4"]
        

        modelParameters["lambda"] = 0.5 * mh1**2 / v0**2
        #modelParameters["msq"] = -mh1**2 / 2. # should be same as the following:
        modelParameters["msq"] = -modelParameters["lambda"] * v0**2
        modelParameters["b2"] = mh2**2 - 0.5 * v0**2 * inputParameters["a2"]

        ## Then the gauge/Yukawa sector
        
        Mt = inputParameters["Mt"] 
        MW = inputParameters["MW"]
        MZ = inputParameters["MZ"]

        # helper
        g0 = 2.*MW / v0
        modelParameters["g1"] = g0*np.sqrt((MZ/MW)**2 - 1)
        modelParameters["g2"] = g0
        # Just take QCD coupling as input
        modelParameters["g3"] = inputParameters["g3"]

        modelParameters["yt"] = np.sqrt(1./2.)*g0 * Mt/MW

        return modelParameters



# end model



class EffectivePotentialxSM_Z2(EffectivePotential):

    def __init__(self, modelParameters: dict[str, float]):
        super().__init__(modelParameters)
        ## ... do singlet+SM specific initialization here. The super call already gave us the model params

        self.num_boson_dof = 29 
        self.num_fermion_dof = 90 

    def evaluate(self, fields: np.ndarray[float], temperature: float) -> complex:
        #return evaluateHighT(fields, temperature)
        
        # for Benoit benchmark we don't use high-T approx and no resummation: just Coleman-Weinberg with numerically evaluated thermal 1-loop

        v = fields[0] # phi ~ 1/sqrt(2) (0, v)
        x = fields[1] # just S -> S + x 

        msq = self.modelParameters["msq"]
        b2 = self.modelParameters["b2"]
        lam = self.modelParameters["lambda"]
        b4 = self.modelParameters["b4"]
        a2 = self.modelParameters["a2"]

        """
        # Get thermal masses
        thermalParams = self.getThermalParameters(temperature)
        mh1_thermal = msq - thermalParams["msq"] # need to subtract since msq in thermalParams is msq(T=0) + T^2 (...)
        mh2_thermal = b2 - thermalParams["b2"]
        """

        ## These need to be arrays! Because other classes call this function with a nested list of field values.
        ## So @todo make all our funct arguments be numpy arrays?
        v = np.asanyarray(v)
        x = np.asanyarray(x)

        # tree level potential
        V0 = 0.5*msq*v**2 + 0.25*lam*v**4 + 0.5*b2*x**2 + 0.25*b4*x**4 + 0.25*a2*v**2 *x**2

        # From Philipp. @todo should probably use the list of defined particles here?
        bosonStuff = self.boson_massSq(fields, temperature)
        fermionStuff = self.fermion_massSq(fields, temperature)

        RGScale = self.modelParameters["RGScale"]
        VTotal = (
            + self.pressureLO(bosonStuff, fermionStuff, temperature)
            + V0
            + self.V1(bosonStuff, fermionStuff, RGScale) 
            + self.V1T(bosonStuff, fermionStuff, temperature))

        return VTotal

    
    ## Evaluate the potential in high-T approx (but keep 4D units)
    def evaluateHighT(self, fields: np.ndarray[float], temperature: float) -> complex:

        v = fields[0] # phi ~ 1/sqrt(2) (0, v)
        x = fields[1] # just S -> S + x 
        T = temperature

        # 4D units
        thermalParameters = self.getThermalParameters(temperature)
        
        msq = thermalParameters["msq"]
        lam = thermalParameters["lambda"]
        b2 = thermalParameters["b2"]
        b4 = thermalParameters["b4"]
        a2 = thermalParameters["a2"]
        

        # tree level potential
        V0 = 0.5 * msq * v**2 + 0.25 * lam * v**4 + 0.5*b2*x**2 + 0.25*b4*x**4 + 0.25*a2*v**2 * x**2

        ## @todo should have something like a static class just for defining loop integrals. NB: m^2 can be negative for scalars so make it complex
        J3 = lambda msq : -(msq + 0j)**(3/2) / (12.*np.pi) * T # keep 4D units

        ## Cheating a bit here and just hardcoding gauge/"goldstone" masses
        mWsq = thermalParameters["g2"]**2 * v**2 / 4.
        mZsq = (thermalParameters["g1"]**2 + thermalParameters["g2"]**2) * v**2 / 4.
        mGsq = msq + lam*v**2 + 0.5*a2*x**2


        ## Scalar mass matrix needs diagonalization, just doing it manually here
        # matrix ( a, b // b, c)

        A = msq + 0.5*a2*x**2 + 3.*v**2*lam
        B = b2 + 0.5*a2*v**2 + 3.*b4*x**2
        C = a2 *v*x 
        thingUnderSqrt = A**2 + B**2 - 2.*A*B + 4.*C**2

        msqEig1 = 0.5 * (A + B - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (A + B + np.sqrt(thingUnderSqrt))
        
    
        # NLO 1-loop correction in Landau gauge. So g^3, Debyes are integrated out by getThermalParameters
        V1 = 2*(3-1) * J3(mWsq) + (3-1) * J3(mZsq) + 3.*J3(mGsq) + J3(msqEig1) + J3(msqEig2)

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
        
        b2 = self.modelParameters["b2"]
        a2 = self.modelParameters["a2"]
        b4 = self.modelParameters["b4"]

        ## LO matching: only masses get corrected
        thermalParameters = self.modelParameters.copy()

        thermalParameters["msq"] = msq + T**2 / 16. * (3. * g2**2 + g1**2 + 4.*yt**2 + 8.*lam) + T**2 * a2 / 24.

        thermalParameters["b2"] = b2 + T**2 * (1./6. *a2 + 1./4. *b4)

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

        # But adding these as Benoit benchmark needs them explicitly...?
        thermalParameters["mDsq1"] = mDsq1
        thermalParameters["mDsq2"] = mDsq2

        return thermalParameters


    def boson_massSq(self, fields, temperature):

        # Is this necessary?
        fields = np.asanyarray(fields)
        v, x = fields[0,...], fields[1,...]

        # TODO: numerical determination of scalar masses from V0

        msq = self.modelParameters["msq"]
        lam = self.modelParameters["lambda"]
        yt = self.modelParameters["yt"]
        g1 = self.modelParameters["g1"]
        g2 = self.modelParameters["g2"]
        
        b2 = self.modelParameters["b2"]
        a2 = self.modelParameters["a2"]
        b4 = self.modelParameters["b4"]

        
        # Scalar masses, just diagonalizing manually. matrix (A C // C B)
        A = msq + 0.5*a2*x**2 + 3.*v**2*lam
        B = b2 + 0.5*a2*v**2 + 3.*b4*x**2
        C = a2 *v*x 
        thingUnderSqrt = A**2 + B**2 - 2.*A*B + 4.*C**2

        msqEig1 = 0.5 * (A + B - np.sqrt(thingUnderSqrt))
        msqEig2 = 0.5 * (A + B + np.sqrt(thingUnderSqrt))

        mWsq = self.modelParameters["g2"]**2 * v**2 / 4.
        mZsq = (self.modelParameters["g1"]**2 + self.modelParameters["g2"]**2) * v**2 / 4.
        # "Goldstones"
        mGsq = msq + lam*v**2 + 0.5*a2*x**2


        # this feels error prone:

        # h, s, chi, W, Z
        massSq = np.stack((msqEig1, msqEig2, mGsq, mWsq, mZsq), axis=-1)
        degreesOfFreedom = np.array([1,1,3,6,3]) 
        c = np.array([3/2,3/2,3/2,5/6,5/6])

        return massSq, degreesOfFreedom, c
    

    def fermion_massSq(self, fields, temperature):

        fields = np.asanyarray(fields)
        v, x = fields[0,...], fields[1,...]

        # Just top quark, others are taken massless
        yt = self.modelParameters["yt"]
        mtsq = yt**2 * v**2 / 2
    
        # @todo include spins for each particle

        massSq = np.stack((mtsq,), axis=-1)
        degreesOfFreedom = np.array([12])
        
        return massSq, degreesOfFreedom



def main():

    ## initial input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        #"RGScale" : 91.1876,
        "RGScale" : 125., # <- Benoit benchmark
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific, choose Benoit benchmark values
        "mh1" : 125.0,
        "mh2" : 120.0,
        "a2" : 0.9,
        "b4" : 1.0
    }

    model = SingletSM_Z2(inputParameters)

    Tn = 100

    userInput = {
        "Tn" : Tn,
        "phaseLocation1" : [ 0.0, 200.0 ],
        "phaseLocation2" : [ 246.0, 0.0 ]
    }

    ## Create control class
    manager = WallGoManager(model, userInput)

    # At this point we should have all required input from the user
    # and the manager should have validated it, found phases etc. So proceed to wall speed calculations

    M, N = 20, 20
    manager.initGrid(M, N)

    manager.solveWall()

    


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()