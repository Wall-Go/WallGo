#ifndef COMMON_H
#define COMMON_H

#include <vector>

// Generic helpers for shared library support
#if defined _WIN32 || defined __CYGWIN__
    #define COLLISION_API __declspec(dllexport)
    
#elif __GNUC__ >= 4 // GCC, clang
    #define COLLISION_API __attribute__ ((visibility ("default")))
    
#else
    #define COLLISION_API
#endif

namespace wallgo
{

// Namespace for numerical constants 
namespace constants {
    constexpr double pi = 3.141592653589793;

}

// ---- Global functions etc

// Clamp number between [min, max]
template <typename T>
inline T clamp(T value, T lower, T upper) 
{
    return std::max(lower, std::min(value, upper));
}



/* Recursive boilerplate for nested D-dimensional std::vectors. 
* Note that this has a large memory overhead - ideally we would use genuine D-dimensional arrays. */
template<int D, typename T>
struct Vec : public std::vector<Vec<D - 1, T>> {
	static_assert(D >= 1, "Vector dimension needs to be > 0");
	template<typename... Args>
	Vec(int n = 0, Args... args) : std::vector<Vec<D - 1, T>>(n, Vec<D - 1, T>(args...)) {
	}
};


template<typename T>
struct Vec<1, T> : public std::vector<T> {
	Vec(int n = 0, const T& val = T()) : std::vector<T>(n, val) {
	}
};


using Array4D = Vec<4, double>;

} // namespace



#endif // Header guard