#include <math.h>



// The four-vector class. Contains (p0,p1,p2,p3). The first element is the energy.
class FourVector {
	
	private:
		double comp[4]; //the components of the four-vector

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
	    	//vectorSph[0] contains the radius, vectorSph[1] the polar angle,
	    	//and vectorSph[2] the azimuthal angle
			comp[0]=vectorSph[0];
			comp[1]=vectorSph[0]*cos(vectorSph[2])*sin(vectorSph[1]);
			comp[2]=vectorSph[0]*sin(vectorSph[2])*sin(vectorSph[1]);
			comp[3]=vectorSph[0]*cos(vectorSph[1]);
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



	    //Sets the components of the vector by hand
	    void setComponents(double vector[4]) {
			comp[0]=vector[0];
			comp[1]=vector[1];
			comp[2]=vector[2];
			comp[3]=vector[3];
	    }


	    //Sets the components of the vector by hand using angles
	    void setComponentsSph(double vectorSph[4]) {
	    	//vectorSph[0] contains the radius, vectorSph[1] the polar angle,
	    	//and vectorSph[2] the azimuthal angle
			comp[0]=vectorSph[0];
			comp[1]=vectorSph[0]*cos(vectorSph[2])*sin(vectorSph[1]);
			comp[2]=vectorSph[0]*sin(vectorSph[2])*sin(vectorSph[1]);
			comp[3]=vectorSph[0]*cos(vectorSph[1]);
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


	    //Returns the four-vector norm, which should be 0 in our case
	   	double norm() {
	        return comp[0]*comp[0]-comp[1]*comp[1]-comp[2]*comp[2]-comp[3]*comp[3];
	    }

	   //Returns the norm of the spatial components
	   	double norm3D() {
	         return comp[1]*comp[1]+comp[2]*comp[2]+comp[3]*comp[3];
	    }

};

