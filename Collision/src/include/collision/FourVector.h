#ifndef FOURVECTOR_H_
#define FOURVECTOR_H_

#include <cmath>
#include <array>

// The four-vector class. Contains (p0,p1,p2,p3). The first element is the energy. Metric is diag(1, -1, -1, -1).
class FourVector {
	
private:
	//To perform scalar products it is fastest to use Cos(theta) and phi
	//But to perform vector addition we keep the cartesian coordinates

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

/*
	//Initialize the four-vector components in terms of angles // TODO 
	FourVector(double vectorSph[4]) {
		//vectorSph[0] contains the radius, vectorSph[1] is cos(polar angle),
		//and vectorSph[2] the azimuthal angle
		comp[0]=vectorSph[0];
		comp[1]=vectorSph[0]*std::cos(vectorSph[2])*std::sqrt(1-vectorSph[1]*vectorSph[1]);
		comp[2]=vectorSph[0]*std::sin(vectorSph[2])*std::sqrt(1-vectorSph[1]*vectorSph[1]);
		comp[3]=vectorSph[0]*vectorSph[1];
	}
*/


	//------------- Operator overloads

    FourVector& operator=(const FourVector& other) {
        if (this != &other) {
            for (int i = 0; i < NCOMPONENTS; ++i) {
                comp[i] = other.comp[i];
            }
        }
        return *this;
    }

	// TODO range check
    double& operator[](int index) {
        return comp[index];
    }

	// TODO range check
    const double& operator[](int index) const {
        return comp[index];
    }

	// Component-wise addition of two FourVectors
    FourVector operator+(const FourVector& other) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] + other.comp[i];
        }
        return result;
    }

	// Component-wise subtraction of two FourVectors
    FourVector operator-(const FourVector& other) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] - other.comp[i];
        }
        return result;
    }

	// Minkowskian dot product of FourVectors
    double operator*(const FourVector& other) const {
		double result = comp[0]*other.comp[0];
		for (int i = 1; i < NCOMPONENTS; ++i) {
			// 'mostly minus'
			result -= comp[i]*other.comp[i];
		}
        return result;
    }

	// Multiply all components by a scalar
    FourVector operator*(double scalar) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] * scalar;
        }
        return result;
    }

	// Scalar multiplication (left operand)
    friend FourVector operator*(double scalar, const FourVector& vector) {
        return vector * scalar;
    }

	// Divide all components by a scalar
    FourVector operator/(double scalar) const {
        FourVector result;
        for (int i = 0; i < NCOMPONENTS; ++i) {
            result.comp[i] = comp[i] / scalar;
        }
        return result;
    }

	FourVector& operator+=(const FourVector& other) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] += other.comp[i];
        }
        return *this;
    }

    FourVector& operator-=(const FourVector& other) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] -= other.comp[i];
        }
        return *this;
    }
    
    FourVector& operator*=(double scalar) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] *= scalar;
        }
        return *this;
    }

    FourVector& operator/=(double scalar) {
        for (int i = 0; i < NCOMPONENTS; ++i) {
            comp[i] /= scalar;
        }
        return *this;
    }


	//-------------------- TODO cleanup of these

	// Minkowskian norm of the four-vector (can be negative!)
	inline double norm() const {
		double result = comp[0]*comp[0];
		for (int i=1; i<NCOMPONENTS; ++i) result -= comp[i]*comp[i];
		return result;
	}


	// Returns the energy which resides at the first component
	inline double energy() const {
		return comp[0];
	}


	// Returns the z-component
	inline double zComp() const {
		return comp[3];
	}

	// Returns component perpendicular to z-axis
	inline double perpComp() const {
		return sqrt(comp[1]*comp[1]+comp[2]*comp[2]);
	}



};

#endif // header guard

