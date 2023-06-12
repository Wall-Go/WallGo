#ifndef FOURVECTOR_H_
#define FOURVECTOR_H_

#include <math.h>

// The four-vector class. Contains (p0,p1,p2,p3). The first element is the energy.
class FourVector {
	
	private:
		//To perform scalar products it is fastest to use Cos(theta) and phi
		//But to perform vector addition we keep the cartesian coordinates

		double comp[4]; //the cartesian components of the four-vector

	public:

		//By default all the elements of the four vector are zero
		FourVector() {
			comp[0]=0.0;
			comp[1]=0.0;
			comp[2]=0.0;
			comp[3]=0.0;
	    }

	    //Initialize the four-vector components in terms of angles
	    FourVector(double vectorSph[4]) {
	    	//vectorSph[0] contains the radius, vectorSph[1] is cos(polar angle),
	    	//and vectorSph[2] the azimuthal angle
			comp[0]=vectorSph[0];
			comp[1]=vectorSph[0]*cos(vectorSph[2])*sqrt(1-vectorSph[1]*vectorSph[1]);
			comp[2]=vectorSph[0]*sin(vectorSph[2])*sqrt(1-vectorSph[1]*vectorSph[1]);
			comp[3]=vectorSph[0]*vectorSph[1];
	    }


	   	//Returns all vector components
	    double* components() {
	        return comp;
	    }

	   //Sets the length of the spatial vector
	    void setLength(double Length) {
	    	for (int i = 0; i < 4; ++i)
	    	{
	    		comp[i]*=Length;
	    	}
	    }

	    //Returns the energy which recides at the first component
	    double energy() {
	        return *comp;
	    }


	    //Returns the z-component
	    double zComp() {
	        return comp[3];
	    }

	   	//Returns the perp-component
	    double perpComp() {
	        return sqrt(comp[1]*comp[1]+comp[2]*comp[2]);
	    }


};

#endif // header guard