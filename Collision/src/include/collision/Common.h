#ifndef COMMON_H
#define COMMON_H


// Def 'uint' in case the compiler doesn't do it for us 
#if !defined(uint)
    typedef unsigned int uint;
#endif


// Generic helpers for shared library support
#if defined _WIN32 || defined __CYGWIN__ // include Windows for completeness, but note that Windows builds of WallGo are untested
    #define COLLISION_API __declspec(dllexport)
    
#elif __GNUC__ >= 4 // GCC, clang
    #define COLLISION_API __attribute__ ((visibility ("default")))
    
#else
    #define COLLISION_API
#endif


// Namespace for numerical constants 
namespace constants {
    constexpr double pi = 3.141592653589793;

}


#endif // Header guard