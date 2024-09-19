"""
A simple example model, of a real scalar field coupled to a Dirac fermion
c.f. 2310.02308
"""

import pathlib
import argparse
import numpy as np
import WallGo


class YukawaModel(WallGo.GenericModel):
    """
    The Yukawa model, inheriting from WallGo.GenericModel.
    """

    @property
    def fieldCount(self) -> int:
        """How many classical background fields"""
        return 1

    def __init__(self, inputParameters: dict[str, float]):
        """
        Initialisation of the Yukawa model:
            - stores modelParameters
            - initialises effectivePotential
            - constructs list of Particles
        """

        self.clearParticles()

        # must initialise and store the model parameters with this variable name
        self.modelParameters = inputParameters

        # must do the same for the effective potential
        self.effectivePotential = EffectivePotentialYukawa(
            self.modelParameters, self.fieldCount
        )

        # constructing the list of particles, starting with psi
        # taking its fluctuations out of equilibrium
        y = self.modelParameters["y"]
        psiL = WallGo.Particle(
            "psiL",
            index=1,  # old collision data has top at index 0
            msqVacuum=lambda fields: (
                self.modelParameters["mf"] + y * fields.getField(0)
            ),
            msqDerivative=lambda fields: y,
            msqThermal=lambda T: 1 / 16 * y**2 * T**2,
            statistics="Fermion",
            totalDOFs=4,
        )
        psiR = WallGo.Particle(
            "psiR",
            index=2,  # old collision data has top at index 0
            msqVacuum=lambda fields: (
                self.modelParameters["mf"] + y * fields.getField(0)
            ),
            msqDerivative=lambda fields: y,
            msqThermal=lambda T: 1 / 16 * y**2 * T**2,
            statistics="Fermion",
            totalDOFs=4,
        )
        self.addParticle(psiL)
        self.addParticle(psiR)

        # Parameters for "phi" field
        msq = self.modelParameters["msq"]
        gamma = self.modelParameters["gamma"]
        lam = self.modelParameters["lam"]


class EffectivePotentialYukawa(WallGo.EffectivePotential):
    """
    The effective potential for a specific model inherits from the
    WallGo class EffectivePotential.
    """

    # HACK! not a fan of requiring checkForImaginary when manifestly real
    def evaluate(
        self, fields: WallGo.Fields, temperature: float, checkForImaginary: bool = False
    ) -> float:
        """
        It is necessary to define a member function called 'evaluate'
        which returns the value of the potential.
        """
        # getting the field from the list of fields (here just of length 1)
        fields = WallGo.Fields(fields)
        phi = fields.getField(0)

        # the constant term
        f_0 = -np.pi**2 / 90 * (1 + 4 * 7 / 8) * temperature**4

        # coefficients of the temperature and field dependent terms
        y = self.modelParameters["y"]
        mf = self.modelParameters["mf"]
        sigma_eff = (
            self.modelParameters["sigma"]
            + 1 / 24 * (self.modelParameters["gamma"] + 4 * y * mf) * temperature**2
        )
        msq_eff = (
            self.modelParameters["msq"]
            + 1 / 24 * (self.modelParameters["lam"] + 4 * y**2) * temperature**2
        )

        # the combined result
        return (
            f_0
            + sigma_eff * phi
            + 1 / 2 * msq_eff * phi**2
            + 1 / 6 * self.modelParameters["gamma"] * phi**3
            + 1 / 24 * self.modelParameters["lam"] * phi**4
        )


def main() -> int:
    """The main function, run with `python3 Yukawa.py`"""

    scriptLocation = pathlib.Path(__file__).parent.resolve()

    argParser = argparse.ArgumentParser()
    argParser.add_argument(
        "--recalculateCollisions",
        help="""Forces full recalculation of relevant collision integrals instead of loading the provided data files for this example.
                This is very slow and disabled by default.
                The resulting collision data will be written to a directory labeled _UserGenerated; the default provided data will not be overwritten.
                """,
        action="store_true",
    )

    args = argParser.parse_args()

    # Initialise WallGo, including loading the default config.
    WallGo.initialize()

    ## Modify the config, we use N=11 for this example
    momentumBasisSize = 3
    WallGo.config.set("PolynomialGrid", "momentumGridSize", str(momentumBasisSize))

    # Print WallGo config. This was read by WallGo.initialize()
    print("\n=== WallGo configuration options ===")
    print(WallGo.config)

    # Scales used for determining suitable values of dxi, dT, dphi etc in derivatives.

    # Guess of the wall thickness: 5/Tn
    wallThicknessIni = 0.05

    # Estimate of the mean free path of the particles in the plasma: 100/Tn
    meanFreePath = 1.0

    # Create WallGo control object
    # The following 2 parameters are used to estimate the optimal value of dT used
    # for the finite difference derivatives of the potential.
    # Temperature scale (in GeV) over which the potential changes by O(1).
    # A good value would be of order Tc-Tn.
    temperatureScaleInput = 1.0
    # Field scale (in GeV) over which the potential changes by O(1). A good value
    # would be similar to the field VEV.
    # Can either be a single float, in which case all the fields have the
    # same scale, or an array.
    fieldScaleInput = (100.0,)

    # Create WallGo control object
    manager = WallGo.WallGoManager(
        wallThicknessIni, meanFreePath, temperatureScaleInput, fieldScaleInput
    )

    # Lagrangian parameters.
    inputParameters = {
        "sigma": 0,
        "msq": 1,
        "gamma": -1.28565864794053,
        "lam": 0.01320208496444000,
        "y": -0.177421729274665,
        "mf": 2.0280748307391000,
    }

    # Initialise the YukawaModel instance.
    model = YukawaModel(inputParameters)

    """
    Register the model with WallGo. This needs to be done only once.
    If you need to use multiple models during a single run,
    we recommend creating a separate WallGoManager instance for each model. 
    """
    manager.registerModel(model)

    # Specify if new collision integrals should be generated by this example. Currently this option is read from WallGo.config
    shouldGenerateCollisions = WallGo.config.getboolean(
        "Collisions", "generateCollisionIntegrals"
    )

    if args.recalculateCollisions:

        # Failsafe, in general you should not worry about the collision module being unavailable as long as it has been properly installed (eg. with pip)
        assert WallGo.isCollisionModuleAvailable(), """WallGoCollision module could not be loaded, cannot proceed with collision integration.
        Please verify you have successfully installed the module ('pip install WallGoCollision')"""

        """Load the collision module and generate a stub collision model from the WallGo model.
        """
        import WallGoCollision

        collisionModelDef = WallGo.collisionHelpers.generateCollisionModelDefinition(
            model,
            # Do not define any model parameters yet./
            includeAllModelParameters=False,
            parameterSymbolsToInclude=[],
        )

        # Add in-equilibrium particles that appear in collision processes
        phiParticle = WallGoCollision.ParticleDescription()
        phiParticle.name = "phi"
        phiParticle.index = 0
        phiParticle.bInEquilibrium = True
        phiParticle.bUltrarelativistic = True
        phiParticle.type = WallGoCollision.EParticleType.eBoson
        # mass-sq function not required or used for UR particles, and it cannot be field-dependent for collisions.
        # Backup of what the vacuum mass was intended to be:
        """
        msqVacuum=lambda fields: (
                msq + g * fields.getField(0) + lam / 2 * fields.getField(0) ** 2
            ),
        """
        collisionModelDef.defineParticleSpecies(phiParticle)

        # The parameter container used by WallGo collision routines is of WallGoCollision.ModelParameters type which behaves somewhat like a Python dict.
        # Here we write our parameter definitions to a local ModelParameters variable and pass it to modelDefinitions later.
        parameters = WallGoCollision.ModelParameters()

        parameters.addOrModifyParameter("y", -0.177421729274665)
        parameters.addOrModifyParameter("gamma", -1.28565864794053)
        parameters.addOrModifyParameter("lam", 0.01320208496444000)
        parameters.addOrModifyParameter("v", 0.0)

        # For particles here we include the thermal mass only
        def psiThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
            return 1 / 16 * p["y"] ** 2

        def phiThermalMassSquared(p: WallGoCollision.ModelParameters) -> float:
            return p["lam"] / 24.0 + p["y"] ** 2.0 / 6.0

        parameters.addOrModifyParameter("msq[0]", phiThermalMassSquared(parameters))
        parameters.addOrModifyParameter("msq[1]", psiThermalMassSquared(parameters))

        # Copy the parameters to our ModelDefinition helper. This finishes the parameter part of model definition.
        collisionModelDef.defineParameters(parameters)

        collisionModel = WallGoCollision.PhysicsModel(collisionModelDef)

        matrixElementFile = (
            scriptLocation / "MatrixElements/MatrixElements_Yukawa_massive.txt"
        )
        collisionModel.readMatrixElements(
            str(matrixElementFile), bPrintMatrixElements=True
        )

        # Create a CollisionTensor object and initialize to the same grid size used elsewhere in WallGo
        collisionTensor: WallGoCollision.CollisionTensor = (
            collisionModel.createCollisionTensor(momentumBasisSize)
        )

        print("Entering collision integral computation, this may take long", flush=True)
        collisionTensorResult: WallGoCollision.CollisionTensorResult = (
            collisionTensor.computeIntegralsAll()
        )

        # Write output to a different directory than where the default data is
        collisionDirectory = (
            scriptLocation / f"CollisionOutput_N{momentumBasisSize}_UserGenerated"
        )
        collisionTensorResult.writeToIndividualHDF5(str(collisionDirectory))

    else:
        # Load pre-generated collision files
        collisionDirectory = scriptLocation / f"CollisionOutput_N{momentumBasisSize}"

    try:
        # Load collision files and register them with the manager. They will be used by the internal Boltzmann solver
        manager.loadCollisionFiles(collisionDirectory)
    except Exception:
        print(
            """\nLoad of collision integrals failed!
              If you were trying to generate your own collision data,
              make sure you run this example script with the --recalculateCollisions command line flag.
              """
        )
        exit(2)

    # Phase information.
    Tn = 89.0
    phasePrevious = WallGo.Fields([30.79])
    phaseNext = WallGo.Fields([192.35])

    phaseInfo = WallGo.PhaseInfo(
        temperature=Tn, phaseLocation1=phasePrevious, phaseLocation2=phaseNext
    )

    # Pass the input to WallGo.
    manager.setParameters(phaseInfo)

    # Solve for the wall velocity
    results = manager.solveWall(bIncludeOffEq=True)
    print(f"Wall speed: {results.wallVelocity}")

    return 0  # return value: success


# Don't run the main function if imported to another file
if __name__ == "__main__":
    main()
