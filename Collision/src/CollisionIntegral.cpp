
#include "CollisionIntegral.h"
#include "CollElem.h"
#include <iostream>

void calculateAllCollisions() {}



double CollisionIntegral4::kinematicPrefactor(const FourVector &p1, const FourVector &p2, const FourVector &p3, const FourVector &p4) {

     // FOR NOW: assume that everything is ultrarelativistic

     

}

// p1Vec is fixed 3-vector
double CollisionIntegral4::calculateIntegrand(const std::array<double, 3> &p1Vec, double p2, double phi2, double phi3, double cosTheta2, double cosTheta3, 
          const std::array<double, 4> &massSquared) 
     {
     
     // Sines
     double sinTheta2 = std::sin(std::acos(cosTheta2));
     double sinTheta3 = std::sin(std::acos(cosTheta3));
     double sinPhi2 = std::sin(phi2);
     double sinPhi3 = std::sin(phi3);
     
     // Cosines
     double cosPhi2 = std::cos(phi2);
     double cosPhi3 = std::cos(phi3);
     
     
     // Magnitude of the fixed 3-vector 
     double p1 = 0.0;
     for ( double comp : p1Vec) p1 += comp*comp;
     p1 = std::sqrt(p1);
     
     // SLOPPY: Create 3-vectors, but I just use FourVectors with vanishing 0-component
     FourVector FV1dummy(0.0, p1Vec[0], p1Vec[1], p1Vec[2]);
     FourVector FV2dummy(0.0, p2*sinTheta2*cosPhi2, p2*sinTheta2*sinPhi2, p2*cosTheta2);
     // 'p3Hat': like p3, but normalized to 1. We will fix its magnitude using a delta(FV4^2 - msq4)
     FourVector FV3Hat(0.0, sinTheta3*cosPhi2, sinTheta3*sinPhi3, cosTheta3);

     // dot products of 3-vectors. Need minus sign here since I'm hacking this with 4-vectors with (+1 -1 -1 -1) metric
     double p1p2Dot = -1.0 * FV1dummy*FV2dummy;
     double p1p3HatDot = -1.0 * FV1dummy*FV3Hat;
     double p2p3HatDot = -1.0 * FV2dummy*FV3Hat;

     // Energies: Since p3 is not fixed yet we only know E1, E2 
     double E1 = std::sqrt(p1*p1 + massSquared[0]);
     double E2 = std::sqrt(p2*p2 + massSquared[1]);

     //------------------------------- TODO move this bit elsewhere
     
     // Now def. function g(p3) = FV4(p3) - msq4 and express the delta function in terms of roots of g(p3) and delta(p3 - p3root); p3 integral becomes trivial.
     // Will need to solve a quadratic equation; some helper variables for it:
     double Q = massSquared[0] + massSquared[1] + massSquared[2] - massSquared[3];
     double kappa = Q + 2.0 * (E1*E2 - p1p2Dot);
     double eps = 2.0 * (E1 + E2);
     double delta = 2.0 * (p1p3HatDot + p2p3HatDot);
     
     auto funcG = [&](double p3) {
          double m3sq = massSquared[2];
          return kappa + delta*p3 - eps * sqrt(p3*p3 + m3sq);
     };

     // Quadratic eq. A p3^2 + B p3 + C = 0, take positive root(s)
     double A = delta*delta - eps*eps;
     double B = 2.0 * kappa * delta;
     double C = kappa*kappa;
     // Roots of g(p3):
     double discriminant = B*B - 4.0*A*C;
     double root1 = 0.5 * (-B - sqrt(discriminant)) / A;
     double root2 = 0.5 * (-B + sqrt(discriminant)) / A;

     // TODO:
     // 1) Check for possible singularities
     // 2) Check that the root(s) satisfy the original eq. with square root
     // 3) Is it possible to have 2 positive roots??

     //-------------------------------
     std::vector<double> rootp3;
     if (root1 >= 0.0) 
          rootp3.push_back(root1);
     if (root2 >= 0.0) 
          rootp3.push_back(root2);

     // TODO need way of calculating and assigning deltaFs, or Chebyshevs. (maybe even give this a pointer to funct that calculates deltaF for given FourVector?)
     double fullIntegrand = 0.0;

     // Now proceed to fix remaining 4-momenta
     for (double p3 : rootp3) {
          double E3 = std::sqrt(p3*p3 + massSquared[2]);

          // Fix 4-momenta for real this time
          FourVector FV1 = FV1dummy;
          FV1[0] = E1;
          FourVector FV2 = FV2dummy;
          FV2[0] = E2;
          FourVector FV3 = p3*FV3Hat;
          FV3[0] = E3;

          // momentum conservation fixed P4 
          FourVector FV4 = FV1 + FV2 - FV3;

          if (FV4.energy() < 0.0) {
               std::cerr << "! Negative energy E4: " << FV4.energy() << "\n";
               continue;
          }
          // Check that P4 is on-shell // TODO

          // Now add all collision elements at these momenta
          double integrand = 0.0;
          for (CollElem<4> collElem : collisionElements) {

               // TODO fix deltaF before this
               integrand += collElem.evaluate( std::array<FourVector, 4>({FV1, FV2, FV3, FV4}) );
          } 

          // Kinematic prefactor
          double kinPrefac = 1.0;
          // Avoid spurious singularity at zero momenta, zero mass
          if (std::abs(massSquared[1]) < massSquaredLowerBound) kinPrefac *= p2;
          else kinPrefac *= p2*p2 / E2;

          if (std::abs(massSquared[2]) < massSquaredLowerBound) kinPrefac *= p3;
          else kinPrefac *= p3*p3 / E3;

          // additional factor from delta(g(p3))
          double gDer = 0.0;
          if (std::abs(massSquared[2]) < massSquaredLowerBound) gDer = delta - eps;
          else gDer = delta - eps * p3 / E3;

          integrand /= std::abs(gDer);


          fullIntegrand += integrand;
     } // end p3 : rootp3
     

     double pi2Pow5 = (2.0*PI) * (2.0*PI) * (2.0*PI) * (2.0*PI) * (2.0*PI);
     return fullIntegrand / pi2Pow5 / 4.0;
}
