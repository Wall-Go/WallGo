#ifndef ENVIRONMENTMACROS_H_
#define ENVIRONMENTMACROS_H_

// Generic helpers for shared library support
#if defined _WIN32 || defined __CYGWIN__
    #define WALLGO_API __declspec(dllexport)
    
#elif __GNUC__ >= 4 // GCC, clang
    #define WALLGO_API __attribute__ ((visibility ("default")))
    
#else
    #define WALLGO_API
#endif

// Stuff that only works for C++20 and newer. For example: std::sin, std::cos were not constexpr until C++20 
#if __cplusplus >= 202002L
	#define WG_CONSTEXPR20 constexpr
#else 
	#define WG_CONSTEXPR20
#endif

#endif // header guard