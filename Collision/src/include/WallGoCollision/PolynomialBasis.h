#pragma once

#include <cmath>

#include "Utils.h"
#include "FourVector.h"
#include "EnvironmentMacros.h"

/* Note to self: GSL library has functions for computing Chebyshev series expansions so could investigate 
* if their implementation of Chebyshev polynomials are faster than what we have here. However I didn't find 
* any documentation about computing individual polynomials in GSL, just series expansions. */

/* TODO would be good to investigate if there's some other basis where the collision integrals are faster to evaluate.
The Chebyshev class should be straightforward to generalize as a virtual interface for example.
*/

namespace wallgo 
{

class Chebyshev
{
private:
    size_t N = 1;

public:

    Chebyshev() : N(1) {}
    Chebyshev(size_t basisSize) : N(basisSize) {}

    // Chebyshev polynomials of 1st kind (see eg. Wikipedia for the definition) 
    inline WG_CONSTEXPR20 double T(int n, double x) const { return std::cos(n* std::acos(x)); }

    // "Reduced" Chebyshev polynomials Tbar and Ttilde. Eq. (28) in 2204.13120 
    inline WG_CONSTEXPR20 double Tbar(int m, double x) const { return (m % 2 == 0 ? T(m, x) - 1.0 : T(m, x) - x); }
    inline WG_CONSTEXPR20 double Ttilde(int n, double x) const { return T(n, x) - 1.0; }
    
    // Construct "rho" momenta on the grid
    inline WG_CONSTEXPR20 double rhoZGrid(int j) const { return std::cos(j * constants::pi / static_cast<double>(N)); }
    inline WG_CONSTEXPR20 double rhoParGrid(int k) const { return std::cos(k * constants::pi / static_cast<double>(N-1)); }

    // Convert p_z and p_par to rho_z, rho_par
    inline WG_CONSTEXPR20 double pZ_to_rhoZ(double pZ) const { return std::tanh(pZ / 2.0); }
    inline WG_CONSTEXPR20 double pPar_to_rhoPar(double pPar) const { return 1.0 - 2.0 * std::exp(-pPar); }

    // Calculate "physical" momentum components p_z and p_par, in units of T, by inverting definitions of rho_z and rho_par
    inline WG_CONSTEXPR20 double rhoZ_to_pZ(double rho_z) const { return 2.0 * std::atanh(rho_z); }
    inline WG_CONSTEXPR20 double rhoPar_to_pPar(double rho_par) const { return -std::log(0.5 * (1 - rho_par)); } 

    // Calculate Tm(rhoZ) Tn(rhoPar) for a given input momenta
    inline WG_CONSTEXPR20 double TmTn(int m, int n, double rhoZ, double rhoPar) const
    {
        return Tbar(m, rhoZ) * Ttilde(n, rhoPar);
    }
    
    // Calculate Tm(rhoZ) Tn(rhoPar) for a given input momenta
    inline WG_CONSTEXPR20 double TmTn(int m, int n, const FourVector &FV) const
    {
        double pZ = FV.zComp();
        double pPar = FV.parComp();
        double rhoZ = pZ_to_rhoZ(pZ);
        double rhoPar = pPar_to_rhoPar(pPar);
        return Tbar(m, rhoZ) * Ttilde(n, rhoPar);
    }

    inline constexpr size_t getBasisSize() const { return N; }
};

} // namespace
