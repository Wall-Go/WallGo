#include "Utils.h"
#include "gslWrapper.h"

namespace wallgo
{

namespace utils
{

std::function<bool()> gExitSignalChecker = nullptr;


bool receivedExitSignal()
{
    return (gExitSignalChecker && gExitSignalChecker());
}

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
