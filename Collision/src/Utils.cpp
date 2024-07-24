#include "Utils.h"
#include "gslWrapper.h"

namespace wallgo
{

namespace utils
{
} // namespace utils


void initializeRNG(int seed)
{
    gslWrapper::initializeRNG(seed);
}

void setSeed(int seed)
{
    gslWrapper::setSeed(seed);
}

void cleanup()
{
    clearRNG();
}

void clearRNG()
{
    gslWrapper::clearRNG();
}

} // namespace wallgo
