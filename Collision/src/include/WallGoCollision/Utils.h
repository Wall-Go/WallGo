#pragma once

#include <vector>
#include <string>
#include <utility>
#include <filesystem>
#include <unordered_map>

#include "Common.h"

namespace wallgo
{

// ---- Global functions etc

namespace utils
{

// Clamp number between [min, max]
template <typename T>
inline T clamp(T value, T lower, T upper)
{
	return std::max(lower, std::min(value, upper));
}

//---- Generic wrappers for RNG routines (so that the user doesn't have to access the gslWrapper namespace)

/* Initializes RNG used by Monte Carlo integrators. Needs to be called before eg. integrating anything. */
void WALLGO_API initializeRNG(int seed = 0);

/* Set seed used by Monte Carlo integrators. By default we use 0. This can safely be called at any time after initializeRNG().
NOTE: if using OpenMP, all threads will get their own RNG, but with the same seed.
This is OK since our calculations are trivially parallel, (independent of each other).*/
void WALLGO_API setSeed(int seed);

// Cleanup of global state (in practice, just the RNG)
void WALLGO_API cleanup();
void WALLGO_API clearRNG();


} // namespace utils
} // namespace wallgo
