#pragma once

#include <string>
#include <utility>
#include <cstdint>
#include <map>
#include <unordered_map>
#include <vector>
#include <functional>

#include "EnvironmentMacros.h"

namespace wallgo
{

// Namespace for numerical constants 
namespace constants
{
	constexpr double pi = 3.141592653589793;
}


template<typename Name_t, typename Index_t>
using TNameMap = std::unordered_map<Name_t, Index_t>;

// Map particle name -> particle index
using ParticleNameMap = TNameMap<std::string, uint32_t>;
using ParticleNamePair = std::pair<ParticleNameMap::key_type, ParticleNameMap::key_type>;

// NB: cannot be used in std::unorderd_map due to lack of a hash function. std::map works
using IndexPair = std::pair<uint32_t, uint32_t>;

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

// Point on (polynomial, momentum) grid
struct GridPoint
{
	GridPoint() {}
	GridPoint(uint32_t polynomialIndex1, uint32_t polynomialIndex2, uint32_t momentumIndex1, uint32_t momentumIndex2)
	{
		m = polynomialIndex1;
		n = polynomialIndex2;
		j = momentumIndex1;
		k = momentumIndex2;
	}
	// Polynomial indices
	uint32_t m, n;
	// Momentum indices
	uint32_t j, k;
};

} //namespace wallgo
