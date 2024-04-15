#ifndef GSLWRAPPER_H
#define GSLWRAPPER_H

// Monte Carlo integration
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte_vegas.h>

#include "CollisionIntegral.h"

namespace wallgo
{

// Helpers for GSL integration routines.
namespace gslWrapper {
     
    /* RNG: This needs to be alloc'd with gsl_rng_alloc. However the RNG is not thread safe, so all threads need their own RNG.
    This will point to a different RNG for each thread (using threadprivate) */
    extern gsl_rng* rng;
    #pragma omp threadprivate(rng)
    
    /* Note that we cannot pass a member function by reference,
    so we dodge this by passing a reference to the object whose function we want to integrate */
    struct gslFunctionParams {
        // Pointer to the object whose member function we are integrating
        CollisionIntegral4* pointerToObject;
        // Additional (non-integration variable) parameters needed to evaluate the integrand
        CollisionIntegral4::IntegrandParameters integrandParameters;
    };

    /* Initializes RNG used by GSL routines. Needs to be called before eg. integrating anything. seed = 0 means we use the default gsl seed (which is 0). */
    void initializeRNG(int seed = 0);
    
    /* Set RNG seed used by GSL routines. By default we use 0. This can safely be called at any time after initializeRNG(). 
    NOTE: if using OpenMP, all threads will get their own RNG, but with the same seed. This is OK since our calculations are trivially parallel, (independent of each other).*/ 
    void setSeed(int seed);

    void clearRNG();

    // pp should be of gslFunctionParams type
    double integrandWrapper(double* intVars, size_t dim, void* pp); 

    extern bool bInitialized;
}

} // namespace

#endif