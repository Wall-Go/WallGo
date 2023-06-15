#ifndef POLYNOMIAL_H
#define POLYNOMIAL_H

#include <cmath>
#include "constants.h"
#include "FourVector.h"

/* Note to self: Probably the 'best' implementation in terms of clarity would be 
* an interface 'PolynomialBasis' that declares functions for computing basis polynomials and 
* grid momenta for a given fixed grid size N (template parameter?).
* Then we could implement the interface in different bases like Chebyshev, Cardinal etc.
* However people have warned about the performance cost of virtual functions so let's not do this
* at least for now. Instead, just def. class for Chebyshev polynomials only 
*/

class Chebyshev {

private:
    size_t N = 1;

public:
    Chebyshev(size_t basisSize) {
        N = basisSize;
    }

    // Chebyshev polynomials of 1st kind (see eg. Wikipedia for the definition) 
    inline double T(int n, double x) const { return std::cos(n* std::acos(x)); }

    // "Reduced" Chebyshev polynomials Tbar and Ttilde. Eq. (28) in 2204.13120 
    inline double Tbar(int m, double x) const { return (m % 2 == 0 ? T(m, x) - 1.0 : T(m, x) - x); }
    inline double Ttilde(int n, double x) const { return T(n, x) - 1.0; }
    
    // Construct "rho" momenta on the grid
    inline double rhoZGrid(int j) const { return std::cos(j * constants::pi / N); }
    inline double rhoParGrid(int k) const { return std::cos(k * constants::pi / (N-1)); }

    // Convert p_z and p_par to rho_z, rho_par
    inline double pZ_to_rhoZ(double pZ) const { return tanh(pZ / 2.0); }
    inline double pPar_to_rhoPar(double pPar) const { return 1.0 - 2.0 * exp(-pPar); }

    // Calculate "physical" momentum components p_z and p_par, in units of T, by inverting definitions of rho_z and rho_par
    inline double rhoZ_to_pZ(double rho_z) const { return 2.0 * atanh(rho_z); }
    inline double rhoPar_to_pPar(double rho_par) const { return -log(0.5 * (1 - rho_par)); } 

    // Calculate Tm(rhoZ) Tn(rhoPar) for a given input momenta
    inline double TmTn(int m, int n, const FourVector &FV) {
        double pZ = FV.zComp();
        double pPar = FV.parComp();
        double rhoZ = pZ_to_rhoZ(pZ);
        double rhoPar = pPar_to_rhoPar(pPar);
        return Tbar(m, rhoZ) * Ttilde(n, rhoPar);
    }

    inline size_t getBasisSize() const { return N; }

};

#endif // header guard