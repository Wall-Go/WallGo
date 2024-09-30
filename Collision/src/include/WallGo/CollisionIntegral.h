#pragma once

#include <cmath>
#include <vector>
#include <cstdint>

#include "Common.h"
#include "FourVector.h"
#include "ThreeVector.h"
#include "CollisionElement.h"
#include "ResultContainers.h"
#include "ModelParameters.h"
#include "IntegrationOptions.h"

namespace wallgo
{

struct ModelChangeContext;

/**** Comment about particle masses. Masses in the integrals appear in two places: 
 *  1) Inside dispersion relations, E^2 = p^2 + m^2
 *  2) Inside propagators in the matrix elements
 * Handling of these two kinds of masses is different in the code and usually also in physical applications:
 * 
 * For 1) we use mass-squares defined by ParticleSpecies and get them through our stored CollisionElements.
 * Hence the kinematic factors need to be computed separately for each CollisionElement.
 * For ultrarelativistic (UR) particles this mass is 0.
 * 
 * For 2) we treat these masses similarly to any other parameter in matrix elements, ie. they are variables contained in MatrixElement objects.
 * Note that in leading-log approximation the common approach is to only use thermal masses in propagators. This is NOT built in to our
 * integration logic in any way, but can be achieved by passing the wanted (symbol, value) pair to matrix elements.  
*/

/**** Ultrarelativistic approximations. We separate CollisionElement objects to ultrarelativistic (UR) and non-UR elements.
 * A CollisionElement is UR if all its external particles have the UR flag enabled. For particles, the UR flag means that the
 * their mass is neglected in dispersion relations, ie. E(p) = |p| always. For UR CollisionElements, the kinematic factors
 * can be calculated in a more optimized way. Whether this optimization is used or not is controlled by our bOptimizeUltrarelativistic flag,
 * which can be changed by passing a IntegrationOptions struct to CollisionIntegral4::integrate(). 
 * NOTE: If a particle is UR, its mass is ALWAYS neglected in energy expressions,
 * irrespectively of whether bOptimizeUltrarelativistic is enabled or not. As described above, masses in propagators are treated differently.  
*/

/* Kinematics struct - holds data for computing the "kinematic" factor in a collision integral. The kinematic factor is: 
    p2^2/E2 * p3^2/E3 * theta(E4) * delta(g(p3))
where the delta function enforces momentum conservation. Standard delta-trick expresses it as sum_i |1/g'(p3)| where we sum over roots of g(p3) = 0.
This struct describes one such root, and we only allow cases with p3 > 0, E4 >= 0.
*/ 
struct Kinematics 
{
    FourVector FV1, FV2, FV3, FV4;
    double prefactor; // This is p2^2/E2 * p3^2/E3 * |1 / g'(p3)|
};

/* Helper struct for computing unknown kinematic factors. 
There is some redundancy but we're trying to avoid having to compute them many times. */ 
struct InputsForKinematics
{
    double p1;
    double p2;
    ThreeVector p1Vec;
    ThreeVector p2Vec;
    ThreeVector p3VecHat;
    double p1p2Dot;
    double p1p3HatDot;
    double p2p3HatDot;
};

struct IntegrationResult
{
    double result;
    double error;
    // TODO add some error flags etc here
    //bool bConverged; // Did the integration converge to goal accuracy
};

/*
Describes a Boltzmann collision integral for 2 -> 2 scattering process. This is always defined per-particle pair,
so that only processes relevant for Boltzmann mixing of those particles are included in a given CollisionIntegral4 object.
"Particle1" refers to the "incoming" particle whose momentum is NOT integrated over; this has 3-momentum p1 and index 'a'.
This is always assumed to be the first particle in each stored CollisionElement.
"Particle2" refers to the other particle that is kept fixed but has its momentum integrated over.
One particle is fixed as the "incoming" particle whose momentum is NOT integrated over.

Our collision integrals are:
    C_a[\delta f] = 1/4 \sum{bcd} \int d^3p_2 d^3p_3 d^3p_4 / ((2pi)^5 (2E_2 2E_3 2E_4)) * \delta^4(P1 + P2 - P3 - P4) * |M_{ab->cd}|^2 P_{ab->cd}[\delta f]
where we only include processes that include \delta f of "Particle2". In practice the logic for choosing the right processes is handled by the PhysicsModel
at CollisionIntegral4 creation time.

The actual integration is done by analytically reducing the 9D integral to a 5D integral of form:
    1/4 \sum_{bcd} int_{0}^{infty} (dp2 p2^2/E2) (dp3 p3^2/E3) int_{0}^{2pi} dphi2 dphi3 int_{-1}^{1} dcosTheta2 dcosTheta3 Theta(E4) delta(P4^2 - m4^2) |M|^2 P[ij -> mn]
which we then integrate with importance-sampled Monte Carlo.

Notable conventions:
1) Commonly a combinatorical factor 1/N_a (degrees of freedom of 'a' particle) is included in front of the collision integral.
Here we assume it to be contained in the matrix element |M|^2.
2) Similarly, our collision operator comes with factor 1/4 in front to avoid overcounting of some ab->cd processes in the sum over external particles.
The other common convention is 1/2. This difference must also be taken into account if using matrix elements not obtained from the WallGoMatrix companion package.
*/
class CollisionIntegral4 {

public:

    /* Struct to carry info about parameters other than the 5 integration variables. 
    * Good for optimization (precalculate p1 etc) + thread safety (since each thread can have its own copy of this struct) */
    struct WALLGO_API IntegrandParameters {
        // Basis polynomial indices (Tbar_m, Ttilde_n)
        int m, n;
        double rhoZ1, rhoPar1;
        double pZ1, pPar1;
        // Magnitude of p1 3-vector
        double p1;
        // Tm(rhoZ1)*Tn(rhoPar1)
        double TmTn_p1; 
    };

    CollisionIntegral4() : mBasisSize(1) {}
    CollisionIntegral4(size_t polynomialBasisSize, const ParticleNamePair& particlePair);
    CollisionIntegral4(const CollisionIntegral4&) = default;

    // Must be copy-assignable for OpenMP
    CollisionIntegral4& operator=(const CollisionIntegral4&) = default;

    void changePolynomialBasis(size_t newBasisSize);

    /* Calculates the whole 5D collision integrand as defined in the "Collision terms" section of WallGo paper [TODO equation number]
    and in the description of this class. */
    double calculateIntegrand(
        double p2,
        double phi2,
        double phi3,
        double cosTheta2,
        double cosTheta3, 
        const IntegrandParameters &integrandParameters);

    /* Calculates the integrand of the reduced 5D integral as described in the description of this class on a given GridPoint. */
    double calculateIntegrand(
        double p2,
        double phi2,
        double phi3,
        double cosTheta2,
        double cosTheta3,
        const GridPoint& gridPoint);

    // Calculate the integral C[m,n; j,k] with Monte Carlo vegas. As always, mn = polynomial indices, jk = grid momentum indices
    IntegrationResult integrate(const GridPoint& gridPoint, const IntegrationOptions& options);

    /* Evaluates the integral everywhere on the (m,n,j,k) grid. */
    CollisionResultsGrid evaluateOnGrid(const IntegrationOptions& options, const CollisionTensorVerbosity& verbosity);

    inline size_t getPolynomialBasisSize() const { return mBasisSize; }

    void addCollisionElement(const CollisionElement<4>& elem);

    /* Call to propagate changes in a PhysicsModel to our CollisionElements and MatrixElements inside them. */
    void handleModelChange(const ModelChangeContext& changeContext);

    // How many integrals need to be computed with the current grid size
    size_t countIndependentIntegrals() const;

    // True if our collision elements lists are empty
    bool isEmpty() const;

    bool isValidGridPoint(const GridPoint& gridPoint) const;

private:

    // For avoiding 1/0
    static constexpr double SMALL_NUMBER = 1e-50;

    size_t mBasisSize;
    ParticleNamePair mParticlePair;

    // mn = polynomial indices, jk = momentum indices
    IntegrandParameters initializeIntegrandParameters(const GridPoint& gridPoint) const;

    /* Kinematic factor depends on masses in the collision element so in principle each element has its own kinematics. 
    We also use a delta-function trick to do delta(g(p3)) as a sum over roots of g(p3) = 0 so this is returns a vector. */
    std::vector<Kinematics> calculateKinematics(const CollisionElement<4> &CollisionElement, const InputsForKinematics& kinematicInput) const;

    /* Optimized computation of the kinematic factor for ultrarelativistic CollElems. This only depends on input momenta and not the CollisionElement itself. 
    In UR limit the momentum-conserving delta function gives only one solution for p3, so this one does not return an array. */
    Kinematics calculateKinematics_ultrarelativistic(const InputsForKinematics& kinematicInput) const;

    /* Evaluate |M^2|/N * P[TmTn] * (kinematics.prefactor). TmTn are the polynomial factors evaluated at each momenta.
    NB: No way of making this const as long as the matrix element evaluation is not const */
    double evaluateCollisionElement(CollisionElement<4> &CollisionElement, const Kinematics& kinematics, const std::array<double, 4>& TmTn);

    /* We separate the collision elements into two subsets: ones with only ultrarelativistic (UR) external particles, and all others. 
    This allows optimizations related to the kinematic factor in UR terms */
    std::vector<CollisionElement<4>> collisionElements_ultrarelativistic;
    std::vector<CollisionElement<4>> collisionElements_nonUltrarelativistic;

    // This is set again whenever integrate() is called, based on the options inputted there
    bool bOptimizeUltrarelativistic = true;

    /* Copy of the model's ModelParameters, this is not strictly needed for evaluation because matrix elements manage their own
    * parameters internally. But storing these here as well allows us to easily write them to HDF5 metadata.
    */
    ModelParameters mModelParameters;
    // Hacky friend declaration used to write the initial model params from PhysicsModel
    friend class PhysicsModel;
};


} // namespace
