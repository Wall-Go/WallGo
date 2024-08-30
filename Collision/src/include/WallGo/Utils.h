#pragma once

#include <vector>
#include <string>
#include <utility>
#include <filesystem>
#include <unordered_map>

#include "Common.h"

namespace wallgo
{

// Current seed for GSL RNG
inline uint64_t gSeedGSL = 0;

// ---- Global functions etc

namespace utils
{

// Clamp number between [min, max]
template <typename T>
inline T clamp(T value, T lower, T upper)
{
	return std::max(lower, std::min(value, upper));
}

/* Used to manually check for kill signals during long-running functions.
Required for the Python module because CTRL-C does not automatically propagate to C++ from Python */
extern std::function<bool()> gExitSignalChecker;

bool receivedExitSignal();

} // namespace utils

//---- Generic wrappers for RNG routines (so that the user doesn't have to access the gslWrapper namespace)

/* Initializes RNG used by Monte Carlo integrators. Needs to be called before eg. integrating anything. */
void WALLGO_API initializeRNG(uint64_t seed = 0);

/* Set seed used by Monte Carlo integrators. By default we use 0. This can safely be called at any time after initializeRNG().
NOTE: if using OpenMP, all threads will get their own RNG, but with the same seed.
This is OK since our calculations are trivially parallel, (independent of each other).*/
void WALLGO_API setSeed(uint64_t seed);

// Cleanup of global state (in practice, just the RNG)
void WALLGO_API cleanup();
void WALLGO_API clearRNG();

} // namespace wallgo
