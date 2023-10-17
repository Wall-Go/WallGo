#ifndef COMMON_H
#define COMMON_H


// Def 'uint' in case the compiler doesn't do it for us 
#if !defined(uint)
    typedef unsigned int uint;
#endif


// Namespace for numerical constants 
namespace constants {
    constexpr double pi = 3.141592653589793;

};


#endif // Header guard