#ifndef COLLELEM_H_
#define COLLELEM_H_

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "FourVector.h"
#include <array>
#include <vector>
#include <string>

// definition in main.cpp
void calculateAllCollisions();


//For now I just hard code the thermal masses
#define PI 3.14159265358979323846


// Boson or fermion?
enum class EParticleType {
	BOSON, FERMION
};

struct Mandelstam {
	double s, t, u;
};

class ParticleSpecies {

public: 

	ParticleSpecies(std::string speciesName, EParticleType particleType) : type(particleType) {
		name = speciesName;
		vacuumMassSquared = 0.0;
		thermalMassSquared = 0.0;
	}

	ParticleSpecies(std::string speciesName, EParticleType particleType, double msqVacuum, double msqThermal) 
		: type(particleType)  
	{
		name = speciesName;
		vacuumMassSquared = msqVacuum;
		thermalMassSquared = msqThermal;
	}

	inline bool isUltrarelativistic() const { return bUltrarelativistic; }
	inline bool isInEquilibrium() const { return bInEquilibrium; }

	inline std::string getName() const { return name; }


	// Equilibrium distribution function for the particle species
	double fEq(double energy) const {
		double res = 0.0;
		if (type == EParticleType::BOSON) {
			// TODO better cutoff
			res = 1.0 / (exp(energy) - 1.0 + 1e-6);
		} else {
			res = 1.0 / (exp(energy) + 1.0);
		}
		return res;
	}

	inline double getDeltaF() const { return deltaF; }

public:
	// TODO setters for these 
	// Neglect mass in dispersion relations or not?
	bool bUltrarelativistic = true;
	// Is the particle assumed to be in thermal equilibrium?
	bool bInEquilibrium = false;

	double vacuumMassSquared;
	double thermalMassSquared;

private:
	std::string name;

	// Set this in constructor using initialization list
	const EParticleType type;
	// Current deviation from equilibrium for the particle // TODO does this make sense here? It's a feature of whole particle species
	double deltaF = 0.0;
	
};

/*
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

	inline double getDeltaF() const { return deltaF; }

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
	// Current deviation from equilibrium for the particle // TODO does this make sense here? It's a feature of whole particle species
	double deltaF = 0.0;
};

*/



/* CollElem class: takes in 4 particle species i,j,m,n and constructs matrix element 
* and population factor ij -> mn scattering process. The full collision integral is constructed 
* from a bunch of these + the kinematic prefactor  */
template <std::size_t NPARTICLES>
class CollElem {

	// For now, no info about momenta in this class

public:
	CollElem() {}

	CollElem(const std::array<ParticleSpecies, NPARTICLES> &inputParticleSpecies) {
		particles = inputParticles;
	}

	inline Mandelstam calculateMandelstam(const FourVector& p1, const FourVector& p2, const FourVector& p3, const FourVector& p4) {
		Mandelstam m;
		m.s = (p1 + p2) * (p1 + p2);
		m.t = (p1 - p3) * (p1 - p3);
		m.u = (p1 - p4) * (p1 - p4);
		return m;
	}

	// TODO properly. Right now I've just hand-coded matrix elements for the top
	void makeMatrixElements(const Mandelstam &mandelstam) {
		double s = mandelstam.s;
		double t = mandelstam.t;
		double u = mandelstam.u;
		matrixElements.clear();
		
		// Thermal masses squared to use in matrix elements
		double mg2 = 3.01593;
		double mq2 = 0.251327;
		// Coupling 
		double gs = 1.2279920495357861;
		double gs4 = gs*gs*gs*gs;

		double tt_gg = -64./9. * gs4 * s*t / ((t-mq2) * (t-mq2));
        double tg_tg = -64./9. * gs4 * s*u / ((u-mq2) * (u-mq2)) + 16.*gs4 * (s*s + u*u) / ((t-mg2) * (t-mg2) );
        double tq_tq = 80./3. * gs4 * (s*s + u*u) / ((t-mg2) * (t-mg2));

		matrixElements.push_back(tt_gg);
		matrixElements.push_back(tg_tg);
		matrixElements.push_back(tq_tq);
	}



	// Evaluate eq (A3) in 2204.13120. See published version since arxiv v1 is wrong
	double populationFactor(const std::array<FourVector, NPARTICLES> &momenta) const {


		double f1 = particles[0].fEq( momenta[0].energy() );
		double f2 = particles[1].fEq( momenta[1].energy() );
		double f3 = particles[2].fEq( momenta[2].energy() );
		double f4 = particles[3].fEq( momenta[3].energy() );
		
		double res =  std::exp(momenta[1].energy()) * particles[0].getDeltaF() / (f1*f1)
					+ std::exp(momenta[0].energy()) * particles[1].getDeltaF() / (f2*f2)
					- std::exp(momenta[3].energy()) * particles[2].getDeltaF() / (f3*f3)
					- std::exp(momenta[2].energy()) * particles[3].getDeltaF() / (f4*f4);

		res = res * f1*f2*f3*f4;
		return res;
	}


	double evaluate(const std::array<FourVector, NPARTICLES> &momenta) {
		double res = 0.0;

		// FOR NOW: hardcode in: 
		// if (p[0] = top, p[1] = top, p[2] = gluon, p[3] = gluon) ETC. here p = particles array

		// TODO

		double E1 = p[0].energy();
		double E2 = p[1].energy();
		double E3 = p[2].energy();
		double E4 = p[3].energy();

		// Coupling 
		double gs = 1.2279920495357861;
		double gs4 = gs*gs*gs*gs;

		Mandelstam m = calculateMandelstam(p[0], p[1], p[2], p[3]);

		// tt_gg, so p[0] p[1] -> p[2] p[3]
		double mq2 = p[0].thermalMassSquared;
		double tt_gg = -64./9. * gs4 * m.s*m.t / ((m.t-mq2) * (m.t-mq2));


	}

private:

	// Particle 0 is the 'incoming' one whose momentum is kept fixed to p1

	std::array<ParticleSpecies, NPARTICLES> particles;
};


/*
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

*/






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