/*******/

#include <iostream>
#include <cmath>
#include <chrono>

#include "CollElem.h"
#include "CollisionIntegral.h"
#include "hdf5Interface.h"


// TEMP
void calculateAllCollisions() {}



// Temporary routine for illustrating how we can generate all collision terms + write them to hdf5 file
void calculateAllCollisions(CollisionIntegral4 &collisionIntegral) {

     std::array<double, 4> massSquared({0.0, 0.0, 0.0, 0.0});

     int gridSizeN = collisionIntegral.getPolynomialBasisSize();

     Array4D collGrid(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);
     Array4D collGridErrors(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);

     std::cout << "Now evaluating all collision integrals\n" << std::endl;

     // m,n = Polynomial indices
     for (int m = 2; m < gridSizeN; ++m) for (int n = 1; n < gridSizeN; ++n) {
          // j,k = grid momentum indices 
          for (int j = 1; j < gridSizeN; ++j) for (int k = 1; k < gridSizeN; ++k) {

               // Note symmetries: C[Tm(-rho_z), Tn(rho_par)] = (-1)^m C[Tm(rho_z), Tn(rho_par)]
               // TODO

               // Monte Carlo result for the integral + its error
               std::array<double, 2> resultMC = collisionIntegral.evaluate(m, n, j, k, massSquared);
               
               collGrid[m-2][n-1][j-1][k-1] = resultMC[0];
               collGridErrors[m-2][n-1][j-1][k-1] = resultMC[1];

               printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);

          } // end j, k
     } // end m,n

     // Write these to file
     std::string filename = "collisions_Chebyshev_" + std::to_string(gridSizeN) + ".hdf5";
     WriteToHDF5(collGrid, filename, "top");
     filename = "errors_Chebyshev_" + std::to_string(gridSizeN) + ".hdf5";
     WriteToHDF5(collGridErrors, filename, "top");

}


//***************


//int main(int argc, char const *argv[]) {
int main() {

     // 2->2 scatterings so 4 external particles
     using CollisionElement = CollElem<4>;
     
     // define particles that we include in matrix elements
     ParticleSpecies topQuark("top", EParticleType::FERMION);
     ParticleSpecies lightQuark("quark", EParticleType::FERMION);
     ParticleSpecies gluon("gluon", EParticleType::BOSON);

     // Set masses. These need to be in units of temperature, ie. (m/T)^2
     double mq2 = 0.251327; // quark
     double mg2 = 3.01593; // SU(3) gluon

     topQuark.thermalMassSquared = mq2;
     lightQuark.thermalMassSquared = mq2;
     gluon.thermalMassSquared = mg2;

     topQuark.vacuumMassSquared = 0.0;
     lightQuark.vacuumMassSquared = 0.0;
     gluon.vacuumMassSquared = 0.0;

     // Which particles are treated as always being in equilibrium?
     topQuark.bInEquilibrium = false;
     lightQuark.bInEquilibrium = true;
     gluon.bInEquilibrium = true;

     // TODO should prob have a constructor that takes all these in as eg. a struct

     // Then create collision elements for 2->2 processes involving these. 
     // By 'collision element' I mean |M|^2 * P[ij -> nm], where P is the population factor involving distribution functions.
     // Right now the correct matrix elements are hardcoded in for the top quark. 
     // In a real model-independent calculation this needs to either calculate the matrix elements itself (hard, probably needs its own class)
     // or read them in from somewhere.
     CollisionElement tt_gg({ topQuark, topQuark, gluon, gluon });
     CollisionElement tg_tg({ topQuark, gluon, topQuark, gluon });
     CollisionElement tq_tq({ topQuark, lightQuark, topQuark, lightQuark });
     
     const int basisSizeN = 20;

     CollisionIntegral4 collInt(basisSizeN);
     collInt.addCollisionElement(tt_gg);
     collInt.addCollisionElement(tg_tg);
     collInt.addCollisionElement(tq_tq);

     // How many collision terms do we need in total
     int nCollisionTerms = std::pow(basisSizeN-1, 4);

     //-------------------- Measure wall clock time

     std::cout << "Running speed test: integral C[2,1,1,1]\n";
     auto startTime = std::chrono::steady_clock::now();

     collInt.evaluate(2, 1, 1, 1, {0.0, 0.0, 0.0, 0.0});

     auto endTime = std::chrono::steady_clock::now();

     auto elapsedTime = endTime - startTime;
     // Convert the elapsed time to milliseconds
     auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count();

     // How long for all collision integrals
     auto totalTime = elapsedTime * nCollisionTerms;

     auto hours = std::chrono::duration_cast<std::chrono::hours>(totalTime).count();
     auto minutes = std::chrono::duration_cast<std::chrono::minutes>(totalTime).count() % 60;

     std::cout << "Test done, took " << elapsedTimeMs << "ms\n";
     std::cout << "Estimated time for all " << nCollisionTerms << " collision integrals: " 
               << hours << " hours " << minutes << " minutes\n";

     //--------------------

     // This would calculate all required collision terms but is currently slow:
     //calculateAllCollisions(collInt);

     // FOR PROFILING: just calculate a few terms and exit

     std::array<double, 4> massSquared({0.0, 0.0, 0.0, 0.0});
     std::array<double, 2> resultMC;

     int m, n, j, k;

     
     m = 2; n = 1; j = 1; k = 1;
     resultMC = collInt.evaluate(m, n, j, k, massSquared);
     printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);

          
     m = 6; n = 4; j = 11; k = 9;
     resultMC = collInt.evaluate(m, n, j, k, massSquared);
     printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);


     return 0;
}
