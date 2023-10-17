#ifndef COMMON_H
#define COMMON_H


#if WITH_PYTHON
    // Used here to check user interruptions (CTRL-C) during long functions
    #include <pybind11/pybind11.h>
#endif


// Def 'uint' in case the compiler doesn't do it for us 
#if !defined(uint)
    typedef unsigned int uint;
#endif


// Namespace for numerical constants 
namespace constants {
    constexpr double pi = 3.141592653589793;

};


#endif // Header guard