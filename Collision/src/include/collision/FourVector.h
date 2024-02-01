#ifndef FOURVECTOR_H_
#define FOURVECTOR_H_

#include <cmath>
#include <array>
#include <assert.h>

// The four-vector class. Contains (p0,p1,p2,p3). The first element is the energy. Metric is diag(1, -1, -1, -1).
class FourVector {
	
private:
	// Don't change this...
	const int NCOMPONENTS = 4;
	// Cartesian components of the four-vector
	std::array<double, 4> comp;

public:

	//------------- Constructors 

	// Initialize a null vector
	FourVector() {
		for (int i=0; i<NCOMPONENTS; ++i) comp[i] = 0.0;
	}

	// Copy-constructor
	FourVector(const FourVector& other) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] = other.comp[i];
        }
    }

	FourVector(double x0, double x1, double x2, double x3) {
		comp[0] = x0;
		comp[1] = x1;
		comp[2] = x2;
		comp[3] = x3;
	}


	// Constructor from length-4 std::array
	FourVector(const std::array<double, 4> arr) {
		comp = arr;
	}

	//------------- Operator overloads

    inline FourVector& operator=(const FourVector& other) {
        if (this != &other) {
            for (int i = 0; i < NCOMPONENTS; ++i) {
                comp[i] = other.comp[i];
            }
        }
        return *this;
    }

    inline double& operator[](int index) {
        assert(index < 4);
        return comp[index];
    }

    inline const double& operator[](int index) const {
        assert(index < 4);
        return comp[index];
    }

    /** profiler says that operator+ and operator- are eating surprisingly big chunk of computation time when doing Mandelstams, can we optimize? */

	// Component-wise addition of two FourVectors
    inline FourVector operator+(const FourVector& other) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] + other.comp[i];
        }
        return result;
    }

	// Component-wise subtraction of two FourVectors
    inline FourVector operator-(const FourVector& other) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] - other.comp[i];
        }
        return result;
    }

	// Minkowskian dot product of FourVectors
    inline double operator*(const FourVector& other) const {
		double result = comp[0]*other.comp[0];
		for (int i = 1; i < NCOMPONENTS; ++i) {
			// 'mostly minus'
			result -= comp[i]*other.comp[i];
		}
        return result;
    }

	// Multiply all components by a scalar
    inline FourVector operator*(double scalar) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] * scalar;
        }
        return result;
    }

	// Scalar multiplication (left operand)
    inline friend FourVector operator*(double scalar, const FourVector& vector) {
        return vector * scalar;
    }

	// Divide all components by a scalar
    inline FourVector operator/(double scalar) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] / scalar;
        }
        return result;
    }

	inline FourVector& operator+=(const FourVector& other) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] += other.comp[i];
        }
        return *this;
    }

    inline FourVector& operator-=(const FourVector& other) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] -= other.comp[i];
        }
        return *this;
    }
    
    inline FourVector& operator*=(double scalar) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] *= scalar;
        }
        return *this;
    }

    inline FourVector& operator/=(double scalar) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] /= scalar;
        }
        return *this;
    }


	//-------------------- TODO cleanup of these

	// Minkowskian square of the four-vector (can be negative!)
	inline double squared() const {
		double result = comp[0]*comp[0];
		for (int i=1; i<NCOMPONENTS; ++i) result -= comp[i]*comp[i];
		return result;
	}

    // Norm of the 3-vector part. Always non-negative
	inline double norm3() const {
		double result = 0.0;
		for (int i=1; i<NCOMPONENTS; ++i) result += comp[i]*comp[i];
		return std::sqrt(result);
	}

	// Returns the energy which resides at the first component
	inline double energy() const {
		return comp[0];
	}


	// Returns the z-component
	inline double zComp() const {
		return comp[3];
	}

	// Returns component perpendicular to z-axis, ie. component parallel to the wall
	inline double parComp() const {
		return std::sqrt(comp[1]*comp[1]+comp[2]*comp[2]);
	}

};

#endif // header guard

