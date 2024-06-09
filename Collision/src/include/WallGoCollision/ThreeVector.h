#ifndef THREEVECTOR_H
#define THREEVECTOR_H

#include <cassert>
#include <cmath>

#include "EnvironmentMacros.h"

namespace wallgo
{

// Class for describing 3-vectors. Consider using an established library for maximum performance
class ThreeVector
{
private:
    double x, y, z;

public:

    ThreeVector() : x(0), y(0), z(0) {}
    
    ThreeVector(double x_, double y_, double z_) : x(x_), y(y_), z(z_) {}

    inline double getX() const { return x; }
    inline double getY() const { return y; }
    inline double getZ() const { return z; }


    inline double squared() const 
    {
        return x*x + y*y + z*z;
    }

    inline double norm() const
    {
        return std::sqrt( x*x + y*y + z*z );
    }


    ThreeVector operator+(const ThreeVector& other) const 
    {
        return ThreeVector(x + other.x, y + other.y, z + other.z);
    }

    ThreeVector operator-(const ThreeVector& other) const 
    {
        return ThreeVector(x - other.x, y - other.y, z - other.z);
    }

    ThreeVector operator*(double scalar) const 
    {
        return ThreeVector(x * scalar, y * scalar, z * scalar);
    }


    friend ThreeVector operator*(double scalar, const ThreeVector& vector) 
    {
        return ThreeVector(scalar * vector.x, scalar * vector.y, scalar * vector.z);
    }

    // Dot product
    double operator*(const ThreeVector& other) const 
    {
        return x*other.x + y*other.y + z*other.z;
    }

};

} // namespace

#endif