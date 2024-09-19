import numpy as np
import numpy.typing as npt
import pathlib

## WallGo imports
import WallGo ## Whole package, in particular we get WallGo.initialize()
from WallGo import GenericModel
from WallGo import Particle
from WallGo import WallGoManager
from WallGo import EffectivePotential
from WallGo import Fields

## Generalization of the Z2 symmetric SM + singlet model now including N singlets.
class NSinglets(GenericModel):

    particles: list[Particle] = []
    outOfEquilibriumParticles: list[Particle] = []
    modelParameters: dict[str, float] = {}
    collisionParameters: dict[str, float] = {}
    
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

        # === Top quark ===
        # The msqVacuum function of an out-of-equilibrium particle must take
        # a Fields object and return an array of length equal to the number of
        # points in fields.
        def topMsqVacuum(fields: Fields) -> Fields:
            return 0.5 * self.modelParameters["yt"] ** 2 * fields.getField(0) ** 2

        # The msqDerivative function of an out-of-equilibrium particle must take
        # a Fields object and return an array with the same shape as fields.
        def topMsqDerivative(fields: Fields) -> Fields:
            return self.modelParameters["yt"]**2 * np.transpose([(1 if i==0 else 0)*fields.getField(i) for i in range(self.fieldCount)])

        def topMsqThermal(T: float) -> float:
            return self.modelParameters["g3"] ** 2 * T**2 / 6.0

        topQuark = Particle(
            "top",
            index=0,
            msqVacuum=topMsqVacuum,
            msqDerivative=topMsqDerivative,
            msqThermal=topMsqThermal,
            statistics="Fermion",
            totalDOFs=12,
        )
        self.addParticle(topQuark)


    ## Go from whatever input params --> action params
    def calculateModelParameters(self, inputParameters: dict[str, float]) -> dict[str, float]:
        super().calculateModelParameters(inputParameters)
    
        modelParameters = {}
        
        # Higgs VEV
        v0 = inputParameters["v0"]
        # Higgs mass
        mh = inputParameters["mh"] # 125 GeV

        # Couplings between Higgs and singlets (should be an array of length N)
        modelParameters["lHS"] = np.array(inputParameters["lHS"])
        # Singlets self-couplings (array of length N)
        modelParameters["lSS"] = np.array(inputParameters["lSS"])
        
        # Higgs self-coupling
        modelParameters["lHH"] = 0.5 * mh**2 / v0**2
        # mu^2 parameters
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
        
        # High-T expansion coefficients
        modelParameters["cH"] = (6*modelParameters["lHH"]+sum(modelParameters["lHS"])+6*modelParameters["yt"]**2+(9/4)*modelParameters["g2"]**2+(3/4)*modelParameters["g1"]**2)/12
        modelParameters["cS"] = (3*modelParameters["lSS"]+4*modelParameters["lHS"])/12

        return modelParameters

# end model


## For this benchmark model we use the UNRESUMMED 4D potential. Furthermore we use customized interpolation tables for Jb/Jf 
class EffectivePotentialNSinglets(EffectivePotential):
    r"""
    Implementation of the Z2-symmetric N-singlet scalars + SM model with the high-T
    1-loop thermal corrections. This model has the potential
    :math:`V = \frac{1}{2}\sum_{i=0}^N\mu_i^2(T)\phi_i^2 + \frac{1}{4}\sum_{i,j=0}^N\lambda_{ij}\phi_i^2\phi_j^2`
    where :math:`\phi_0` is assumed to be the Higgs and :math:`\phi_{i>0}` the 
    singlet scalars. 
    For simplicity, we only consider models with no couplings between the different 
    singlets; only couplings between the Higgs and the singlets are allowed.
    This means :math:`\lambda_{ij}=0` when :math:`i,j>0` and :math:`i\neq j`.
    """

    def __init__(self, modelParameters: dict[str, float], fieldCount: int):
        super().__init__(modelParameters, fieldCount)
        ## ... do singlet+SM specific initialization here. The super call already gave us the model params

        ## Count particle degrees-of-freedom to facilitate inclusion of light particle contributions to ideal gas pressure
        self.num_boson_dof = 27 + fieldCount
        self.num_fermion_dof = 90 

    def canTunnel(self, tunnelingTemperature: float=None) -> bool:
        """
        Function used to determine if tunneling can happen with this potential.
        Verifies that the Higgs phase exists at T=0 and that it is stable and the
        true vacuum. Also verifies that both phases exist and are stable at T=Tc
        (or T=tunnelingTemperature).

        Parameters
        ----------
        tunnelingTemperature : float, optional
            Temperature at which the tunneling takes place. If None, uses Tc. 
            The default is None.

        Returns
        -------
        tunnel : bool
            Returns True if all the conditions mentioned above are satisfied. 
            Returns False otherwise.

        """
        tunnel = True
        
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
            
        if tunnelingTemperature is None:    
            # If no temperature was provided, computes and uses Tc
            T = self.findTc()
            print(f'Tc={T}')
            if T is None:
                tunnel = False
        else: 
            T = tunnelingTemperature
            
        if T is not None:
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
            
    def findTc(self) -> float:
        """
        Computes the critical temperature

        Returns
        -------
        float
            Value of the critical temperature. If there is no solution, returns None.

        """
        A = self.modelParameters["cH"]**2/self.modelParameters["lHH"]-sum(self.modelParameters["cS"]**2/self.modelParameters["lSS"])    
        B = 2*(self.modelParameters["cH"]*self.modelParameters["muHsq"]/self.modelParameters["lHH"]-sum(self.modelParameters["cS"]*self.modelParameters["muSsq"]/self.modelParameters["lSS"])) 
        C = self.modelParameters["muHsq"]**2/self.modelParameters["lHH"]-sum(self.modelParameters["muSsq"]**2/self.modelParameters["lSS"])
        
        discr = B**2-4*A*C
        if discr < 0:
            # The discriminant is negative, which would lead to imaginary Tc^2.
            print("No critical temperature : negative discriminant")
            return None
        
        # Finds the two solutions for Tc^2, and keep the smallest positive one.
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
            
    def findPhases(self, temperature: float) -> tuple:
        """
        Computes the position of the two phases at T=temperature.

        Parameters
        ----------
        temperature : float
            Temperature at which to evaluate the position of the phases.

        Returns
        -------
        phase1 : array-like
            Array containing the position of the singlet phase.
        phase2 : array-like
            Array containing the position of the Higgs phase.

        """
        muHsqT = self.modelParameters['muHsq']+self.modelParameters['cH']*temperature**2
        muSsqT = self.modelParameters['muSsq']+self.modelParameters['cS']*temperature**2
        
        phase1 = np.sqrt(np.append([0],-muSsqT/self.modelParameters['lSS']))
        phase2 = np.sqrt(np.append([-muHsqT/self.modelParameters['lHH']], (self.fieldCount-1)*[0]))
        
        return phase1, phase2
    
    def evaluate(self, fields: Fields, temperature: float, checkForImaginary: bool=False) -> np.ndarray:
        """
        Evaluates the tree-level potential with the 1-loop high-T thermal corrections.

        Parameters
        ----------
        fields : Fields
            Fields object containing the VEVs of the fields.
        temperature : float or array-like
            Temperature at which the potential is evaluated.
        checkForImaginary: bool, optional
            Has no effect because the potential is always real with the 1-loop 
            high-T thermal corrections with no resummation. Default is False.

        Returns
        -------
        array-like
            Values of the potential.

        """

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
        if len(temperature.shape) > 0: # If temperature is an array
            muSsqT = muSsq+cS*temperature[:,None]**2
        else: # If temperature is a float
            muSsqT = muSsq+cS*temperature**2

        # Tree level potential with high-T 1-loop thermal corrections.
        V0 = 0.5*muHsqT*h**2 + 0.5*np.sum(muSsqT*s**2, axis=-1) + 0.25*lHH*h**4 + 0.25*np.sum(lSS*s**4, axis=-1) + 0.5*h**2*np.sum(lHS*s**2, axis=-1)

        # Adding the terms proportional to T^4
        VTotal = V0 + self.constantTerms(temperature)

        return VTotal
    

    def constantTerms(self, temperature: npt.ArrayLike) -> npt.ArrayLike:

        ## Fermions contribute with a magic 7/8 prefactor as usual. Overall minus sign since Veff(min) = -pressure
        return -(self.num_boson_dof + (7./8.) * self.num_fermion_dof) * np.pi**2 * temperature**4 / 90.



def main() -> None:
    #########################################
    ## Example with 1 Higgs and 2 singlets ##
    #########################################

    scriptLocation = pathlib.Path(__file__).parent.resolve()
    
    # Number of singlets
    nbrSinglets = 2

    WallGo.initialize()

    # loading in local config file
    WallGo.config.readINI(
        pathlib.Path(__file__).parent.resolve() / "WallGoSettings.ini"
    )

    ## Modify the config, we use N=11 and M=25 for this example
    WallGo.config.config.set("PolynomialGrid", "momentumGridSize", "11")
    WallGo.config.config.set("PolynomialGrid", "spatialGridSize", "25")

    # Print WallGo config. This was read by WallGo.initialize()
    print("=== WallGo configuration options ===")
    print(WallGo.config)

    ## Guess of the wall thickness
    wallThicknessIni = 0.05
    
    # Estimate of the mean free path of the particles in the plasma
    meanFreePath = 1.0

    ## Create WallGo control object
        # The following 2 parameters are used to estimate the optimal value of dT used 
    # for the finite difference derivatives of the potential.
    # Temperature scale over which the potential changes by O(1). A good value would be of order Tc-Tn.
    temperatureScale = 10.
    # Field scale over which the potential changes by O(1). A good value would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the same scale, or an array.
    fieldScale = [10.,10.,10.]
    manager = WallGoManager(wallThicknessIni, meanFreePath, temperatureScale, fieldScale)


    """Initialize your GenericModel instance. 
    The constructor currently requires an initial parameter input, but this is likely to change in the future
    """

    ## QFT model input. 
    ## The parameters related to the singlets (muSsq, lHS and lSS) must be arrays
    ## of length nbrSinglets.
    inputParameters = {
        "RGScale" : 125., 
        "v0" : 246.0,
        "MW" : 80.379,
        "MZ" : 91.1876,
        "Mt" : 173.0,
        "g3" : 1.2279920495357861,
        # scalar specific
        "mh" : 125.0,
        "muSsq" : [-8000,-10000],
        "lHS" : [0.75,0.9],
        "lSS" : [0.5,0.7]
    }

    model = NSinglets(inputParameters, nbrSinglets)
    
    Tc = model.effectivePotential.findTc()
    if Tc is None:
        return 0
    Tn = 0.8*Tc
    if model.effectivePotential.canTunnel(Tn) == False:
        print('Tunneling impossible. Try with different parameters.')
        return 0

    """
    Register the model with WallGo. This needs to be done only once.
    If you need to use multiple models during a single run,
    we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    collisionDirectory = scriptLocation / "CollisionOutput"

    try:
        # Load collision files and register them with the manager. They will be used by the internal Boltzmann solver
        manager.loadCollisionFiles(collisionDirectory)
    except Exception:
        print(
            """\nLoad of collision integrals failed! This example files comes with pre-generated collision files for N=11,
              so load failure here probably means you've either moved files around or changed the grid size.
              If you were trying to generate your own collision data, make sure you run this example script with the --recalculateCollisions command line flag.
              """
        )
        exit(2)
    
    phase1, phase2 = model.effectivePotential.findPhases(Tn)

    phaseInfo = WallGo.PhaseInfo(temperature = Tn, 
                                    phaseLocation1 = WallGo.Fields(phase1[None,:]), 
                                    phaseLocation2 = WallGo.Fields(phase2[None,:]))
    

    """Give the input to WallGo. It is NOT enough to change parameters directly in the GenericModel instance because
        1) WallGo needs the PhaseInfo 
        2) WallGoManager.setParameters() does parameter-specific initializations of internal classes
    """ 
    manager.setParameters(phaseInfo)

    """WallGo can now be used to compute wall stuff!"""

    ## ---- Solve wall speed in Local Thermal Equilibrium approximation

    vwLTE = manager.wallSpeedLTE()

    print(f"LTE wall speed: {vwLTE}")

    ## ---- Solve field EOM. For illustration, first solve it without any out-of-equilibrium contributions. The resulting wall speed should match the LTE result:
    
    manager.eom.includeOffEq = False
    

    bIncludeOffEq = False
    print(f"=== Begin EOM with {bIncludeOffEq=} ===")

    results = manager.solveWall(bIncludeOffEq)
    
    wallVelocity = results.wallVelocity
    widths = results.wallWidths
    offsets = results.wallOffsets

    print(f"{wallVelocity=}")
    print(f"{widths=}")
    print(f"{offsets=}")

    ## Repeat with out-of-equilibrium parts included. This requires solving Boltzmann equations, invoked automatically by solveWall()  
    bIncludeOffEq = True
    print(f"=== Begin EOM with {bIncludeOffEq=} ===")

    results = manager.solveWall(bIncludeOffEq)
    wallVelocity = results.wallVelocity
    wallVelocityError = results.wallVelocityError
    widths = results.wallWidths
    offsets = results.wallOffsets

    print(f"{wallVelocity=}")
    print(f"{wallVelocityError=}")
    print(f"{widths=}")
    print(f"{offsets=}")


## Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
