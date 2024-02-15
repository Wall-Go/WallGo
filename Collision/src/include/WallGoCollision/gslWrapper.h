#ifndef GSLWRAPPER_H
#define GSLWRAPPER_H

// Monte Carlo integration
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte_vegas.h>

#include "CollisionIntegral.h"

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

    void initializeRNG();
    void clearRNG();

    // pp should be of gslFunctionParams type
    double integrandWrapper(double* intVars, size_t dim, void* pp); 
}

#endif