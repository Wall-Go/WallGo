#include "Utils.h"
#include "gslWrapper.h"

namespace wallgo
{

void initializeRNG(int seed)
{
    gslWrapper::initializeRNG(seed);
}

void setSeed(int seed)
{
    gslWrapper::setSeed(seed);
}

void clearRNG()
{
    gslWrapper::clearRNG();
}

} // namespace