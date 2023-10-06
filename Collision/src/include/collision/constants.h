#ifndef CONSTANTS_H
#define CONSTANTS_H

// Defines namespace for numerical constants 

namespace constants {
    constexpr double pi = 3.141592653589793;

};

// Def 'uint' in case the compiler doesn't do it for us 
#if !defined(uint)
    typedef unsigned int uint;
#endif

#endif // header guard