/*******/

#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;
#include <math.h>
#include <sys/time.h>
#include <time.h>

#include "kinematics.h"
#include "operators.h"
#include "CollElem.h"
#include "CollisionIntegral.h"


#define PI 3.14159265358979323846
#define GS 1.2279920495357861




void printTime(long duration){
     //3600000000 microseconds in an hour
     long hr = duration / 3600000000;
     duration = duration - 3600000000 * hr;
     //60000000 microseconds in a minute
     long min = duration / 60000000;
     duration = duration - 60000000 * min;

     //1000000 microseconds in a second
     long sec = duration / 1000000;

     cout << hr << " hours and " << min << " minutes and " << sec << " seconds" << endl;

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
     
     CollisionIntegral4 collInt;
     collInt.addCollisionElement(tt_gg);
     collInt.addCollisionElement(tg_tg);
     collInt.addCollisionElement(tq_tq);

     double test = collInt.calculateIntegrand(std::array<double, 3>( {0.0, 1.42, 3.1} ), 5, 4.2, 1.3, 0.4, -0.13, std::array<double, 4>( {0.0, 0.0, 0.0, 0.0} ));

     printf("test %g\n", test);
     
     //calculateAllCollisions();

     return 0;
}
