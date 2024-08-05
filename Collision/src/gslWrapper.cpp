#include "gslWrapper.h"

namespace wallgo
{

namespace gslWrapper
{
    gsl_rng* rng = nullptr;
    bool bInitialized = false;
    #pragma omp threadprivate(rng)
}    

void gslWrapper::initializeRNG(int seed)
{
    if (!bInitialized)
    {
        gsl_rng_env_setup();
        #pragma omp parallel 
        {
            gslWrapper::rng = gsl_rng_alloc(gsl_rng_default);
        }

        bInitialized = true;
        setSeed(seed);
    }
}

void gslWrapper::setSeed(int seed)
{
    if (!bInitialized) return;

    // All threads get same seed (OK for now)
    #pragma omp parallel
    {
        gsl_rng_set(gslWrapper::rng, static_cast<unsigned long>(seed));
    }
}

void gslWrapper::clearRNG()
{
    if (bInitialized)
    {
        #pragma omp parallel 
        {
            gsl_rng_free(gslWrapper::rng);
        }

        bInitialized = false;
    }
}

double gslWrapper::integrandWrapper(double *intVars, size_t dim, void *pp)
{
    // GSL requires size_t as input: the logic is that dim should be used to check that intVars is correct size.
    // But I'm a badass and just do this:
    (void)dim;

    gslWrapper::gslFunctionParams* params = static_cast<gslWrapper::gslFunctionParams*>(pp);

    double p2 = intVars[0];
    double phi2 = intVars[1];
    double phi3 = intVars[2];
    double cosTheta2 = intVars[3];
    double cosTheta3 = intVars[4];
    return params->pointerToObject->calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, params->integrandParameters);
}

} // namespace