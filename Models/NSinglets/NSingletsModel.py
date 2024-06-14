import numpy as np
import numpy.typing as npt
import os
import pathlib

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
## For Benoit benchmarks we need the unresummed, non-high-T potential:
from WallGo import EffectivePotential
from WallGo import Fields, WallGoResults

## Generalization of the Z2 symmetric SM + singlet model now including N singlets.
class NSinglets(GenericModel):

    particles = []
    outOfEquilibriumParticles = []
    modelParameters = {}
    
    fieldCount = 3


    def __init__(self, initialInputParameters: dict[str, float], nbrSinglets: int):
        
        self.fieldCount = nbrSinglets + 1 # N singlets and 1 Higgs
        
        self.modelParameters = self.calculateModelParameters(initialInputParameters)

        # Initialize internal Veff with our params dict. @todo will it be annoying to keep these in sync if our params change?
        self.effectivePotential = EffectivePotentialNSinglets(self.modelParameters, self.fieldCount) 
        
        self.defineParticles()


    def defineParticles(self) -> None:
        # NB: particle multiplicity is pretty confusing because some internal DOF counting is handled internally already.
        # Eg. for SU3 gluons the multiplicity should be 1, NOT Nc^2 - 1.
        # But we nevertheless need something like this to avoid having to separately define up, down, charm, strange, bottom 
        
        self.clearParticles()

        ## === Top quark ===
        topMsqVacuum = lambda fields: 0.5 * self.modelParameters["yt"]**2 * fields.GetField(0)**2
        topMsqDerivative = lambda fields: self.modelParameters["yt"]**2 * np.transpose([(1 if i==0 else 0)*fields.GetField(i) for i in range(self.fieldCount)])
        topMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0

        topQuark = Particle("top", 
                            msqVacuum = topMsqVacuum,
                            msqDerivative = topMsqDerivative,
                            msqThermal = topMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = False,
                            ultrarelativistic = True,
                            totalDOFs = 12
        )
        self.addParticle(topQuark)


        ## === SU(3) gluon ===
        # The msqVacuum function must take a Fields object and return an array of length equal to the number of points in fields.
        gluonMsqVacuum = lambda fields: np.zeros_like(fields.GetField(0))
        # The msqDerivative function must take a Fields object and return an array with the same shape as fields.
        gluonMsqDerivative = lambda fields: np.zeros_like(fields)
        gluonMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 * 2.0

        gluon = Particle("gluon", 
                            msqVacuum = gluonMsqVacuum,
                            msqDerivative = gluonMsqDerivative,
                            msqThermal = gluonMsqThermal,
                            statistics = "Boson",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            totalDOFs = 16
        )
        self.addParticle(gluon)

        ## === Light quarks, 5 of them ===
        lightQuarkMsqThermal = lambda T: self.modelParameters["g3"]**2 * T**2 / 6.0
        lightQuark = Particle("lightQuark", 
                            msqVacuum = lambda fields: 0.0,
                            msqDerivative = 0.0,
                            msqThermal = lightQuarkMsqThermal,
                            statistics = "Fermion",
                            inEquilibrium = True,
                            ultrarelativistic = True,
                            totalDOFs = 60
        )
        self.addParticle(lightQuark)


    ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}

        v0 = inputParameters["v0"]
        # Scalar eigenvalues
        mh = inputParameters["mh"] # 125 GeV

        ## these are direct inputs:
        modelParameters["lHS"] = np.array(inputParameters["lHS"])
        modelParameters["lSS"] = np.array(inputParameters["lSS"])
        

        modelParameters["lHH"] = 0.5 * mh**2 / v0**2
        modelParameters["muHsq"] = -mh**2/2
        modelParameters["muSsq"] = np.array(inputParameters["muSsq"])

        ## Then the gauge/Yukawa sector
        
        Mt = inputParameters["Mt"] 
        MW = inputParameters["MW"]
        MZ = inputParameters["MZ"]

        # helper
        g0 = 2.*MW / v0
        modelParameters["g1"] = g0*np.sqrt((MZ/MW)**2 - 1)
        modelParameters["g2"] = g0
        modelParameters["g3"] = inputParameters["g3"]

        modelParameters["yt"] = np.sqrt(1./2.)*g0 * Mt/MW
        
        modelParameters["cH"] = (6*modelParameters["lHH"]+sum(modelParameters["lHS"])+6*modelParameters["yt"]**2+(9/4)*modelParameters["g2"]**2+(3/4)*modelParameters["g1"]**2)/12
        modelParameters["cS"] = (3*modelParameters["lSS"]+4*modelParameters["lHS"])/12
        print(modelParameters["cH"],modelParameters["cS"])

        return modelParameters

# end model


## For this benchmark model we use the UNRESUMMED 4D potential. Furthermore we use customized interpolation tables for Jb/Jf 
class EffectivePotentialNSinglets(EffectivePotential):

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do singlet+SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 27 + fieldCount
        self.num_fermion_dof = 90 
        
    

    ## ---------- EffectivePotential overrides. 
    # The user needs to define evaluate(), which has to return value of the effective potential when evaluated at a given field configuration, temperature pair. 
    # Remember to include full T-dependence, including eg. the free energy contribution from photons (which is field-independent!)

    def canTunnel(self, tunnelingTemperature=None):
        tunnel = True
        
        if tunnelingTemperature is None:
            # Higgs phase is the true vacuum at T=0
            if self.modelParameters["muHsq"]**2/self.modelParameters["lHH"] <= sum(self.modelParameters["muSsq"]**2/self.modelParameters["lSS"]):
                print("Higgs phase is not the true vacuum at T=0")
                print(f"""{self.modelParameters["muHsq"]**2/self.modelParameters["lHH"] - sum(self.modelParameters["muSsq"]**2/self.modelParameters["lSS"])=}""")
                tunnel = False
            
            # Higgs phase exists at T=0
            if self.modelParameters["muHsq"] >= 0 or self.modelParameters["lHH"] <= 0:
                print("Higgs phase doesn't exist at T=0")
                print(f"""{self.modelParameters["muHsq"]=} {self.modelParameters["lHH"]=}""")
                tunnel = False
            # Higgs phase is stable at T=0
            if np.any(self.modelParameters["muSsq"]-self.modelParameters["lHS"]*self.modelParameters["muHsq"]/self.modelParameters["lHH"] <= 0):
                print("Higgs phase is not stable at T=0")
                print(f"""{self.modelParameters["muSsq"]-self.modelParameters["lHS"]*self.modelParameters["muHsq"]/self.modelParameters["lHH"]=}""")
                tunnel = False
                
            T = self.findTc()
            print(f'Tc={T}')
        else: 
            T = tunnelingTemperature
        if T is None:
            tunnel = False
        else:
            muSsqT = self.modelParameters["muSsq"]+self.modelParameters["cS"]*T**2
            muHsqT = self.modelParameters["muHsq"]+self.modelParameters["cH"]*T**2
            
            # Higgs phase exists at T=Tc
            if muHsqT >= 0:
                print("Higgs phase doesn't exist at T=Tc")
                print(f"{muHsqT=}")
                tunnel = False
            # Higgs phase is stable at T=Tc
            if np.any(muSsqT-self.modelParameters["lHS"]*muHsqT/self.modelParameters["lHH"] <= 0):
                print("Higgs phase is not stable at T=Tc")
                print(f"""{muSsqT-self.modelParameters["lHS"]*muHsqT/self.modelParameters["lHH"]}""")
                tunnel = False
                
            # Singlets phase exists at T=Tc
            if np.any(muSsqT >= 0) or np.any(self.modelParameters["lSS"] <= 0):
                print("Singlets phase doesn't exist at T=Tc")
                print(f"{muSsqT=} {self.modelParameters['lSS']=}")
                tunnel = False
            # Singlets phase is stable at T=Tc
            if muHsqT - sum(self.modelParameters["lHS"]*muSsqT/self.modelParameters["lSS"]) <= 0:
                print("Singlets phase is not stable at T=Tc")
                print(f"""{muHsqT - sum(self.modelParameters["lHS"]*muSsqT/self.modelParameters["lSS"])=}""")
                tunnel = False
                
        return tunnel
            
    def findTc(self):
        A = self.modelParameters["cH"]**2/self.modelParameters["lHH"]-sum(self.modelParameters["cS"]**2/self.modelParameters["lSS"])    
        B = 2*(self.modelParameters["cH"]*self.modelParameters["muHsq"]/self.modelParameters["lHH"]-sum(self.modelParameters["cS"]*self.modelParameters["muSsq"]/self.modelParameters["lSS"])) 
        C = self.modelParameters["muHsq"]**2/self.modelParameters["lHH"]-sum(self.modelParameters["muSsq"]**2/self.modelParameters["lSS"])
        
        discr = B**2-4*A*C
        if discr < 0:
            print("No critical temperature : negative discriminant")
            return None
        
        Tc1 = (-B+np.sqrt(discr))/(2*A)
        Tc2 = (-B-np.sqrt(discr))/(2*A)
        
        if Tc1 <= 0 and Tc2 <= 0:
            print("Negative critical temperature squared")
            return None
        if Tc1 > 0 and Tc2 > 0:
            return min(np.sqrt(Tc1), np.sqrt(Tc2))
        if Tc1 > 0:
            return np.sqrt(Tc1)
        if Tc2 > 0:
            return np.sqrt(Tc2)
            
    def findPhases(self, temperature):
        muHsqT = self.modelParameters['muHsq']+self.modelParameters['cH']*temperature**2
        muSsqT = self.modelParameters['muSsq']+self.modelParameters['cS']*temperature**2
        
        phase1 = np.sqrt(np.append([0],-muSsqT/self.modelParameters['lSS']))
        phase2 = np.sqrt(np.append([-muHsqT/self.modelParameters['lHH']], (self.fieldCount-1)*[0]))
        
        return phase1, phase2
    
    def evaluate(self, fields: Fields, temperature: float, checkForImaginary: bool = False) -> complex:

        h,s = fields[...,0], fields[...,1:]
        temperature = np.array(temperature)

        muHsq = self.modelParameters['muHsq']
        muSsq = self.modelParameters['muSsq']
        lHH = self.modelParameters['lHH']
        lHS = self.modelParameters['lHS']
        lSS = self.modelParameters['lSS']
        cH = self.modelParameters['cH']
        cS = self.modelParameters['cS']

        muHsqT = muHsq+cH*temperature**2
        if len(temperature.shape) > 0:
            muSsqT = muSsq+cS*temperature[:,None]**2
        else:
            muSsqT = muSsq+cS*temperature**2

        # tree level potential
        V0 = 0.5*muHsqT*h**2 + 0.5*np.sum(muSsqT*s**2, axis=-1) + 0.25*lHH*h**4 + 0.25*np.sum(lSS*s**4, axis=-1) + 0.5*h**2*np.sum(lHS*s**2, axis=-1)

        VTotal = V0 + self.constantTerms(temperature)

        return VTotal
    

    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:
        """Need to explicitly compute field-independent but T-dependent parts
        that we don't already get from field-dependent loops. At leading order in high-T expansion these are just
        (minus) the ideal gas pressure of light particles that were not integrated over in the one-loop part.
        """

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(self.num_boson_dof + (7./8.) * self.num_fermion_dof) * np.pi**2 * temperature**4 / 90.



def main():

    WallGo.initialize()

    # loading in local config file
    WallGo.config.readINI(
        pathlib.Path(__file__).parent.resolve() / "WallGoSettings.ini"
    )

    ## Modify the config, we use N=11 for this example
    WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "11")
    WallGo.config.config.set("PolynomialGrid", "spatialGridSize", "51")

    # Print WallGo config. This was read by WallGo.initialize()
    print("=== WallGo configuration options ===")
    print(WallGo.config)

    ## Guess of the wall thickness
    wallThicknessIni = 0.05
    
    # Estimate of the mean free path of the particles in the plasma
    meanFreePath = 0

    ## Create WallGo control object
        # The following 2 parameters are used to estimate the optimal value of dT used 
    # for the finite difference derivatives of the potential.
    # Temperature scale over which the potential changes by O(1). A good value would be of order Tc-Tn.
    temperatureScale = 10.
    # Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the same scale, or an array.
    fieldScale = [10.,10.]
    manager = WallGoManager(wallThicknessIni, meanFreePath, temperatureScale, fieldScale)


    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input, but this is likely to change in the future
    """

    ## QFT model input. Some of these are probably not intended to change, like gauge masses. Could hardcode those directly in the class.
    inputParameters = {
        "RGScale" : 125., 
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific
        "mh" : 125.0,
        "muSsq" : [-16000],
        "lHS" : [3.2],
        "lSS" : [3]
    }

    model = NSinglets(inputParameters, 1)
    print(model.effectivePotential.canTunnel())
    Tc = model.effectivePotential.findTc()
    Tn = 0.9*Tc
    print(model.effectivePotential.canTunnel(Tn))

    """ Register the model with WallGo. This needs to be done only once. 
    If you need to use multiple models during a single run, we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)


    ## ---- Directory name for collisions integrals. Currently we just load these
    collisionDirectory = pathlib.Path(__file__).parent.resolve() / "CollisionOutput"
    collisionDirectory.mkdir(parents=True, exist_ok=True)


    manager.loadCollisionFiles(collisionDirectory)


    modelParameters = model.calculateModelParameters(inputParameters)
    
    phase1, phase2 = model.effectivePotential.findPhases(Tn)
    print(phase1, phase2,Tn)
    print(model.effectivePotential.evaluate(phase1[None,:], Tn), model.effectivePotential.evaluate(phase2[None,:], Tn))

    phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                    phaseLocation1 = WallGo.Fields(phase1[None,:]), 
                                    phaseLocation2 = WallGo.Fields(phase2[None,:]))
    

    """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
        1) WallGo needs the PhaseInfo 
        2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
    """ 
    manager.setParameters(phaseInfo)

    ## TODO initialize collisions. Either do it here or already in registerModel(). 
    ## But for now it's just hardcoded in Boltzmann.py and __init__.py

    """WallGo can now be used to compute wall stuff!"""

    ## ---- Solve wall speed in Local Thermal Equilibrium approximation

    vwLTE = manager.wallSpeedLTE()

    print(f"LTE wall speed: {vwLTE}")

    ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:

    ## This will contain wall widths and offsets for each classical field. Offsets are relative to the first field, so first offset is always 0
    wallParams = WallGo.WallParams(np.array(2*[5/Tn]), np.array(2*[0]))
    
    manager.eom.includeOffEq = False
    # print(manager.eom.wallPressure(manager.hydro.vJ-1e-4, wallParams, True, 0, 1e-3))
    print(manager.hydro.vJ,manager.hydro.template.vJ)
    # vs, ps, wallParamsList,_,_,hydroResultsList = manager.eom.gridPressure(0.01, manager.hydro.vJ-1e-4, 50)
    
    
    import matplotlib.pyplot as plt
    vs = np.linspace(0.01, manager.hydro.vJ-1e-4,50)
    plt.plot(vs,[manager.hydro.findMatching(v)[2] for v in vs])
    plt.plot(vs,[manager.hydro.template.findMatching(v)[2] for v in vs])
    plt.grid()
    plt.show()
    
    # plt.plot(vs,ps)
    # plt.grid()
    # plt.show()
    # for i in range(2):
    #     plt.plot(vs,[wallP.widths[i] for wallP in wallParamsList])
    # plt.grid()
    # plt.show()
    # for i in range(2):
    #     plt.plot(vs,[wallP.offsets[i] for wallP in wallParamsList])
    # plt.grid()
    # plt.show()
    # plt.plot(vs,[hydroR.temperaturePlus for hydroR in hydroResultsList],vs,[hydroR.temperatureMinus for hydroR in hydroResultsList])
    # plt.grid()
    # plt.show()
    

    # bIncludeOffEq = False
    # print(f"=== Begin EOM with {bIncludeOffEq=} ===")

    # results = manager.solveWall(bIncludeOffEq)
    # print(f"results=")
    # wallVelocity = results.wallVelocity
    # widths = results.wallWidths
    # offsets = results.wallOffsets

    # print(f"{wallVelocity=}")
    # print(f"{widths=}")
    # print(f"{offsets=}")

    # ## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
    # bIncludeOffEq = True
    # print(f"=== Begin EOM with {bIncludeOffEq=} ===")

    # results = manager.solveWall(bIncludeOffEq)
    # wallVelocity = results.wallVelocity
    # wallVelocityError = results.wallVelocityError
    # widths = results.wallWidths
    # offsets = results.wallOffsets

    # print(f"{wallVelocity=}")
    # print(f"{wallVelocityError=}")
    # print(f"{widths=}")
    # print(f"{offsets=}")
    


    # end parameter-space loop

# end main()


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
