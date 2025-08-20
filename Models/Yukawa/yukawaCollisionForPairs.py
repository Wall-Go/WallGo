"""
Demonstration of the computeIntegralsForPair methods.

This script shows how to compute collision integrals for specific particle pairs
rather than the full set of out-of-equilibrium partice pairs.
"""

import sys
import pathlib
import os

# WallGo imports
import WallGo
import WallGoCollision

# Add the Yukawa folder to the path to import YukawaModel
sys.path.append(str(pathlib.Path(__file__).resolve().parent))
from yukawa import YukawaModel


def initCollisionModel(wallGoModel: "YukawaModel") -> "WallGoCollision.PhysicsModel":
    """Initialize the Collision model and set the seed."""

    import WallGoCollision  # pylint: disable = C0415

    # Collision integrations utilize Monte Carlo methods, so RNG is involved.
    # We can set the global seed for collision integrals as follows.
    # This is optional; by default the seed is 0.
    WallGoCollision.setSeed(0)

    collisionModelDefinition = (
        WallGo.collisionHelpers.generateCollisionModelDefinition(wallGoModel)
    )

    # Add in-equilibrium particles that appear in collision processes
    # The out-of-equilibrium particles are taken from the definition in the model file
    phiParticle = WallGoCollision.ParticleDescription()
    phiParticle.name = "phi"
    phiParticle.index = 0
    phiParticle.bInEquilibrium = True
    phiParticle.bUltrarelativistic = True
    phiParticle.type = WallGoCollision.EParticleType.eBoson
    # mass-sq function not required or used for UR particles,
    # and it cannot be field-dependent for collisions.
    # Backup of what the vacuum mass was intended to be:
    
    parameters = WallGoCollision.ModelParameters()

    parameters.add("y", wallGoModel.modelParameters["y"])
    parameters.add("gamma", wallGoModel.modelParameters["gamma"])
    parameters.add("lam", wallGoModel.modelParameters["lam"])
    parameters.add("v", 0.0)

    # fermion asymptotic thermal mass^2 (twice the static thermal mass)
    # in units of T
    parameters.add(
        "mf2", 1 / 8 * wallGoModel.modelParameters["y"] ** 2
    )
    # scalar thermal mass^2 in units of T
    parameters.add(
        "ms2",
        +wallGoModel.modelParameters["lam"] / 24.0
        + wallGoModel.modelParameters["y"] ** 2.0 / 6.0,
    )

    collisionModelDefinition.defineParticleSpecies(phiParticle)
    collisionModelDefinition.defineParameters(parameters)

    collisionModel = WallGoCollision.PhysicsModel(collisionModelDefinition)

    return collisionModel

def demonstrateComputeIntegralsForPair(collision_model: "WallGoCollision.PhysicsModel", save_dir: str | None = None) -> None:
    """Demonstrate computing collision integrals for specific particle pairs using computeIntegralsForPair()."""
    
    gridSize = 3  # Small grid for faster demonstration
    
    # Set default save directory if not provided
    if save_dir is None:
        save_dir = f"CollisionOutput_N{gridSize}_Pairs"
    
    # Create output directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Load matrix elements FIRST (before creating collision tensor)
    print("Loading matrix elements...")
    matrix_element_file = pathlib.Path(__file__).resolve().parent / "MatrixElements" / "matrixElements.yukawa.json"
    bShouldPrintMatrixElements = True
    if not collision_model.loadMatrixElements(str(matrix_element_file), bShouldPrintMatrixElements):
        print(f"FATAL: Failed to load matrix elements from {matrix_element_file}")
        return {}
    
    # Create collision tensor AFTER loading matrix elements
    print("Creating collision tensor...")
    collisionTensor = collision_model.createCollisionTensor(gridSize)
    
    # Configure integration options
    options = WallGoCollision.IntegrationOptions()
    options.maxIntegrationMomentum = 10  # Reduced for speed
    options.absoluteErrorGoal = 1e-6
    options.relativeErrorGoal = 1e-1
    options.maxTries = 20
    options.calls = 10000  # Reduced for speed
    options.bIncludeStatisticalErrors = True
    
    collisionTensor.setIntegrationOptions(options)
    
    # Set verbosity for demonstration
    verbosity = WallGoCollision.CollisionTensorVerbosity()
    verbosity.bPrintElapsedTime = True
    verbosity.progressReportPercentage = 0.5
    verbosity.bPrintEveryElement = False
    collisionTensor.setIntegrationVerbosity(verbosity)
    
    # Demonstrate computing collision integrals for multiple particle pairs
    print("\n5. Computing collision integrals for particle pairs")
    
    # Define the particle pairs to compute
    particle_pairs = [("psiL", "psiL"), ("psiR", "psiR")]
    
    results = {}
    
    for particle1, particle2 in particle_pairs:
        print(f"\nComputing collision integrals for pair: {particle1} - {particle2}")
        
        # Use computeIntegralsForPair to compute the collision integrals for this specific pair
        pair_result = collisionTensor.computeIntegralsForPair(particle1, particle2)
            
        if pair_result is not None:
            print(f"Successfully computed integrals for {particle1}-{particle2}")
            
            # Save directly to the main directory
            filename = os.path.join(save_dir, f"{particle1}_{particle2}.h5")
            print(f"Saving to: {filename}")
            
            pair_result.writeToHDF5(filename)
            print(f"Successfully saved {particle1}-{particle2}")
            
            results[f"{particle1}_{particle2}"] = pair_result
        else:
            print(f"Failed to compute integrals for {particle1}-{particle2} (returned None)")
    
    return


if __name__ == "__main__":

    # Initialize the Yukawa model
    wallGoModel = YukawaModel()
    
    # Set example parameters
    inputParameters = {
        "sigma": 0.0,
        "msq": 1.0,
        "gamma": -1.2,
        "lam": 0.10,
        "y": 0.55,
        "mf": 0.30,
    }
    wallGoModel.modelParameters.update(inputParameters)

    collisionModel = initCollisionModel(wallGoModel)

    results = demonstrateComputeIntegralsForPair(collisionModel)
