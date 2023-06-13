#ifndef COLLELEM_H_
#define COLLELEM_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "FourVector.h"
#include <array>
#include <vector>

// definition in main.cpp
void calculateAllCollisions();


//For now I just hard code the thermal masses
#define  MQ2 0.251327 				//Top quark mass squared
#define  MG2 3.01593 				//Gluon mass squared
#define PI 3.14159265358979323846


// Boson or fermion?
enum class EParticleType {
	BOSON, FERMION
};

class Particle {

public:

	Particle(EParticleType particleType) : type(particleType), momentum(0.0, 0.0, 0.0, 0.0) {
		
		vacuumMassSquared = 0.0;
		thermalMassSquared = 0.0;
	}

	Particle(EParticleType particleType, double msqVacuum, double msqThermal, const FourVector& mom)
		: type(particleType) {

		momentum = mom;
		vacuumMassSquared = msqVacuum;
		thermalMassSquared = msqThermal;
	}

	inline void setMomentum(const FourVector& p) { momentum = p; }

	inline bool isUltrarelativistic() const { return bUltrarelativistic; }
	inline bool isInEquilibrium() const { return bInEquilibrium; }

	// Equilibrium distribution function for the particle
	double fEq() const {
		double res = 0.0;
		double energy = momentum[0];
		if (type == EParticleType::BOSON) {
			// TODO better cutoff
			res = 1.0 / (exp(energy) - 1.0 + 1e-6);
		} else {
			res = 1.0 / (exp(energy) + 1.0);
		}
		return res;
	}

private:
	FourVector momentum;
	double vacuumMassSquared;
	double thermalMassSquared;
	// Set this in constructor using initialization list
	const EParticleType type;
	// Neglect mass in dispersion relations or not?
	bool bUltrarelativistic = true;
	// Is the particle assumed to be in thermal equilibrium?
	bool bInEquilibrium = false;
};



/* This describes the full collision integrand for i,j -> n, m scattering process. 
* In general there are many matrix elements contributing, and for each matrix element there is 
* a statistical "population" factor P(i,j -> n,m)
*/
template <std::size_t NPARTICLES>
class CollElem {


public:
	CollElem() : particles( Particle(), Particle(), Particle(), Particle() ) {}

	CollElem(const std::array<Particle, NPARTICLES> &inputParticles) {
		particles = inputParticles;
	}

	// TODO properly. Right now I've just hand-coded matrix elements for the top
	void makeMatrixElements() {
		
	}

private:

	// Particle 0 is the 'incoming' one whose momentum is kept fixed to p1

	std::array<Particle, NPARTICLES> particles;

	// Matrix elements squared: |M(i,j -> n,m)|^2 
	std::vector<double> matrixElements;
	// Statistical population factors P(i,j -> n,m)
	std::vector<double> populationFactor;
};


//Note to self: Move these matrix elements to their own file

//Q+Q-> V+ V matrix element
static inline double matrixElementQQVVX(double s,double t, double u) {

	return s*t/(t-MQ2+1e-6)/(t-MQ2+1e-6);

}

//Q+V-> Q+ V matrix element
static inline double matrixElementQVQVX(double s,double t, double u) {
//I here use equation (A4) and pull out a factor -64/9 from both terms
	return s*u/(u-MQ2+1e-6)/(u-MQ2+1e-6)-9.0/4.0*(s*s+u*u)/(t-MG2+1e-6)/(t-MG2+1e-6);

}


//T+Q-> T+ Q matrix element
static inline double matrixElementTQTQX(double s,double t, double u) {
//I here use equation (A4) and pull out a factor -64/9 from the function
	return -5.0/4.0*(s*s+u*u)/(t-MG2+1e-6)/(t-MG2+1e-6);
}


//Q+Q-> Q+ Q matrix element. So all fermions are equal
static inline double matrixElementQQQQX(double s,double t, double u) {
//Table II in https://arxiv.org/pdf/hep-ph/0209353.pdf. Pulled out a factor of -64/9
	return -((s*s+u*u)/(t-MG2+1e-6)/(t-MG2+1e-6)+(s*s+t*t)/(u-MG2+1e-6)/(u-MG2+1e-6));
}


//V+V-> V+ V matrix element. So all fermions are equal
static inline double matrixElementVVVVX(double s,double t, double u){
//Table II in https://arxiv.org/pdf/hep-ph/0209353.pdf. Pulled out a factor of -64/9
	return 81.0/16.0*(s*u/(t-MG2+1e-6)/(t-MG2+1e-6)+s*t/(u-MG2+1e-6)/(u-MG2+1e-6));
}








// ???
typedef double (*funcptr)(double);

// The collElem class specifies which matrix element, the spin of the particles, and what
//delta_f terms the user wants to calculate
//The idea is that the user can create #collElem classes at the start which are all included in a given
// integration call

class CollElemOld {
	
// why protected?
protected:
	double (*matrixFunc)(double, double, double); //Returns the value of the matrix element for given s,t,u

	funcptr distributions[4];//Returns the value of all distribution function for p

	int matrixElem; //specifies the matrix element that should be used
	int spin[4]; //specifies the spin of the particles in the matrix element. Even boson, odd fermion

	// p:deltaF=0, k:deltaF=1, p2:deltaF=2, k2=deltaF=3
	int deltaF[4]; //specifies which deltaF terms should be calculated

public:
	CollElemOld() {
		//By default all particles are bosons
		spin[0]=2;
		spin[1]=2;
		spin[2]=2;
		spin[3]=2;

		//By default the matrix element is 0: QQ ->VV
		matrixElem=0;

		//By default only the delta_p term is calculated
		deltaF[0]=1;
		deltaF[1]=0; //Negative values specifies that the term is not included
		deltaF[2]=0;
		deltaF[3]=0;

	}


	double calculateMatrixElement(double s, double t, double u){
		//Returns the matrix element as a function of the mandelstam variables
		return matrixFunc(s,t,u);
	}


	/*For the definition of these terms "see https://journals.aps.org/prd/abstract/10.1103/PhysRevD.106.023501"
		equation A.3 */

	//We multiply all terms with deltaF to ensure that only the requested terms are evaluated

	//The product of distribution functions that multiply delta f_p
	double calculateDeltaP(double p,double k, double p2, double k2){
		return *(deltaF)*exp(k)*distributions[1](k)*distributions[2](p2)*distributions[3](k2)/distributions[0](p);
	}


	//The product of distribution functions that multiply delta f_k
	double calculateDeltaK(double p,double k, double p2, double k2){
		return *(deltaF+1)*exp(p)*distributions[2](p2)*distributions[3](k2)*distributions[0](p)/distributions[1](k);
	}

	//The product of distribution functions that multiply delta f_p2
	double calculateDeltaP2(double p,double k, double p2, double k2){
		return -*(deltaF+2)*exp(k2)*distributions[3](k2)*distributions[0](p)*distributions[1](k)/distributions[2](p2);
	}

	//The product of distribution functions that multiply delta f_k2
	double calculateDeltaK2(double p,double k, double p2, double k2){
		return -*(deltaF+3)*exp(p2)*distributions[0](p)*distributions[1](k)*distributions[2](p2)/distributions[3](k2);
	}


	void specifyMatrixElement(){
		//Function to choose which matrix element should be used
		// QQVV:0, QVQV:1, TQTQ: 2
			switch(matrixElem) {
				case 0:
				matrixFunc=matrixElementQQVVX;
				break;
				case 1:
				matrixFunc=matrixElementTQTQX;
				break;
				case 2:
				matrixFunc=matrixElementTQTQX;
				break;
				
			}
		}

	//Specifies the matrix element that should be used
	void setMatrixElem(int matrixElemInput) {
		matrixElem=matrixElemInput;
	}

	//Prints the Matrix element
	int printMatrixElem() {
		return matrixElem;
	}

	//Specifies the distribution functions for all particles
	void setDistributions() {
		for (int i = 0; i < 4; ++i)
		{
			if (*(spin+i)%2==0)
			{
					*(distributions+i)=nB;
			}else{
					*(distributions+i)=nF;
			}
		}
	}

	//Specifies which delta_F terms should be used
	void setDeltaF(int *deltaFInput) {
		deltaF[0]=*(deltaFInput);
		deltaF[1]=*(deltaFInput+1); 
		deltaF[2]=*(deltaFInput+2);
		deltaF[3]=*(deltaFInput+3);
	}

	//Prints the delta_F terms
	int* printDeltaF() {
		return deltaF;
	}

	//Specifies the spin of the particles
	void setSpin(int *spinInput) {
		spin[0]=*(spinInput);
		spin[1]=*(spinInput+1);
		spin[2]=*(spinInput+2);
		spin[3]=*(spinInput+3);
	}

	//Prints the spin of the particle
	int* printSpin() {
		return spin;
	}

};

#endif // header guard