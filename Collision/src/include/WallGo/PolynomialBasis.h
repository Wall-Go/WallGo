#pragma once

#include <cmath>
#include <cstdint>

#include "Common.h"
#include "FourVector.h"

/* Note to self: GSL library has functions for computing Chebyshev series expansions so could investigate 
* if their implementation of Chebyshev polynomials are faster than what we have here. However I didn't find 
* any documentation about computing individual polynomials in GSL, just series expansions. */

/* TODO would be good to investigate if there's some other basis where the collision integrals are faster to evaluate.
* The repeated evaluation of sines/cosines for Chebyshev polynomials is a real bottleneck right now.
*/

namespace wallgo 
{

namespace chebyshev
{

// Chebyshev polynomials of 1st kind (see eg. Wikipedia for the definition)
static inline WG_CONSTEXPR20 double T(uint32_t n, double x) { return std::cos(n* std::acos(x)); }

// "Reduced" Chebyshev polynomials Tbar and Ttilde. Eq. (28) in 2204.13120 
static inline WG_CONSTEXPR20 double Tbar(uint32_t m, double x) { return (m % 2 == 0 ? T(m, x) - 1.0 : T(m, x) - x); }
static inline WG_CONSTEXPR20 double Ttilde(uint32_t n, double x) { return T(n, x) - 1.0; }
    
// Construct "rho" momenta on the grid. N is the basis size
static inline WG_CONSTEXPR20 double rhoZGrid(uint32_t j, size_t N) { return std::cos(j * constants::pi / static_cast<double>(N)); }
static inline WG_CONSTEXPR20 double rhoParGrid(uint32_t k, size_t N) { return std::cos(k * constants::pi / static_cast<double>(N-1)); }

// Convert p_z and p_par to rho_z, rho_par
static inline WG_CONSTEXPR20 double pZ_to_rhoZ(double pZ) { return std::tanh(pZ / 2.0); }
static inline WG_CONSTEXPR20 double pPar_to_rhoPar(double pPar) { return 1.0 - 2.0 * std::exp(-pPar); }

// Calculate "physical" momentum components p_z and p_par, in units of T, by inverting definitions of rho_z and rho_par
static inline WG_CONSTEXPR20 double rhoZ_to_pZ(double rho_z) { return 2.0 * std::atanh(rho_z); }
static inline WG_CONSTEXPR20 double rhoPar_to_pPar(double rho_par) { return -std::log(0.5 * (1 - rho_par)); }

// Calculate Tm(rhoZ) Tn(rhoPar) for a given input momenta
static inline WG_CONSTEXPR20 double TmTn(uint32_t m, uint32_t n, double rhoZ, double rhoPar)
{
    return Tbar(m, rhoZ) * Ttilde(n, rhoPar);
}
    
// Calculate Tm(rhoZ) Tn(rhoPar) for a given input momenta
static inline double TmTn(uint32_t m, uint32_t n, const FourVector &FV)
{
    double pZ = FV.zComp();
    double pPar = FV.parComp();
    double rhoZ = pZ_to_rhoZ(pZ);
    double rhoPar = pPar_to_rhoPar(pPar);
    return Tbar(m, rhoZ) * Ttilde(n, rhoPar);
}

} // namespace chebyshevgrid
} // namespace wallgo
