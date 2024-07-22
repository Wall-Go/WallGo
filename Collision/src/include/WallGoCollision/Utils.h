#pragma once

#include <vector>
#include <string>


#include "EnvironmentMacros.h"
#include "ModelDefinitions.h"

namespace wallgo
{

// Namespace for numerical constants 
namespace constants
{
    constexpr double pi = 3.141592653589793;
}

// ---- Global functions etc

extern bool gExitSignaled;

inline std::string particlePairToString(const ParticleNamePair& pair)
{
	return "(" + pair.first + ", " + pair.second + ")";
}

// Clamp number between [min, max]
template <typename T>
inline T clamp(T value, T lower, T upper) 
{
    return std::max(lower, std::min(value, upper));
}

/* Recursive boilerplate for nested D-dimensional std::vectors. 
* Note that this has a large memory overhead - ideally we would use genuine D-dimensional arrays. */
template<size_t D, typename T>
struct Vec : public std::vector<Vec<D - 1, T>>
{
	static_assert(D >= 1, "Vector dimension needs to be > 0");
	template<typename... Args>
	Vec(size_t n = 0, Args... args) : std::vector<Vec<D - 1, T>>(n, Vec<D - 1, T>(args...)) {}
};

template<typename T>
struct Vec<1, T> : public std::vector<T>
{
	Vec(size_t n = 0, const T& val = T()) : std::vector<T>(n, val) {}
};

using Array4D = Vec<4, double>;


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

} // namespace
