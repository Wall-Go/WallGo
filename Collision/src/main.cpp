/*******/

#include <iostream>
#include <chrono>
#include <cmath>
#include <sys/time.h>
#include <time.h>

#include "CollElem.h"
#include "CollisionIntegral.h"
#include "hdf5Interface.h"


void printTime(long duration){
     //3600000000 microseconds in an hour
     long hr = duration / 3600000000;
     duration = duration - 3600000000 * hr;
     //60000000 microseconds in a minute
     long min = duration / 60000000;
     duration = duration - 60000000 * min;

     //1000000 microseconds in a second
     long sec = duration / 1000000;

     std::cout << hr << " hours and " << min << " minutes and " << sec << " seconds" << std::endl;

}


// TEMP
void calculateAllCollisions() {}



// Temporary routine for illustrating how we can generate all collision terms + write them to hdf5 file
void calculateAllCollisions(CollisionIntegral4 &collisionIntegral) {

     std::array<double, 4> massSquared({0.0, 0.0, 0.0, 0.0});

     int gridSizeN = collisionIntegral.getPolynomialBasisSize();

     Array4D collGrid(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);
     Array4D collGridErrors(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);

     // m,n = Polynomial indices
     for (int m = 2; m < gridSizeN; ++m) for (int n = 1; n < gridSizeN; ++n) {
          // j,k = grid momentum indices 
          for (int j = 1; j < gridSizeN; ++j) for (int k = 1; k < gridSizeN; ++k) {

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

     calculateAllCollisions(collInt);

/*
     std::array<double, 4> massSquared({0.0, 0.0, 0.0, 0.0});

     std::array<double, 2> integral;

     int m, n, j, k;

     m = 2; n = 1; j = 1; k = 1;
     double p2 = 5.4;
     double phi2 = 4.1;
     double phi3 = 1.2;
     double cosTheta2 = std::cos(1.7);
     double cosTheta3 = std::cos(0.3);

     printf("integrand only: %g\n", collInt.calculateIntegrand(m, n, j, k, p2, phi2, phi3, cosTheta2, cosTheta3, massSquared));

     std::cin.get();

     m = 2; n = 1; j = 1; k = 1;
     integral = collInt.evaluate(m, n, j, k, massSquared);
     printf("%d %d %d %d: %g +/- %g\n", m, n, j, k, integral[0], integral[1]);

     k = 2;
     integral = collInt.evaluate(m, n, j, k, massSquared);
     printf("%d %d %d %d: %g +/- %g\n", m, n, j, k, integral[0], integral[1]);
     
     k = 3;
     integral = collInt.evaluate(m, n, j, k, massSquared);
     printf("%d %d %d %d: %g +/- %g\n", m, n, j, k, integral[0], integral[1]);

*/

     //calculateAllCollisions();

     return 0;
}
