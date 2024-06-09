#ifndef COLLELEM_H_
#define COLLELEM_H_

#include <cmath>
#include <array>
#include <vector>
#include <string>
#include <iostream>
#include <memory>

#include "EnvironmentMacros.h"
#include "FourVector.h"
#include "ParticleSpecies.h"
#include "MatrixElement.h"

namespace wallgo
{

struct WALLGO_API Mandelstam
{
	double s, t, u;
};


/* CollElem class: This describes collision process with external particles types being fixed */
template <size_t NPARTICLES>
class WALLGO_API CollElem
{

public:

	// which deltaF terms are nonzero
	std::array<bool, NPARTICLES> bDeltaF;

	/* Use initialization list here for setting the particle species, 
	otherwise may run into compiler errors due to (lack of) copy constructors */
	CollElem(const std::array<std::shared_ptr<ParticleSpecies>, NPARTICLES> &inputParticleSpecies) : particles(inputParticleSpecies)
	{
		bool bAllUltrarelativistic = true;
		for (const auto& p : inputParticleSpecies)
		{
			if (!p->isUltrarelativistic()) 
			{
				bAllUltrarelativistic = false;
				break;	
			}
		}
		bUltrarelativistic = bAllUltrarelativistic;
	}


	inline Mandelstam calculateMandelstam(const FourVector& p1, const FourVector& p2, const FourVector& p3, const FourVector& p4) const
	{
		Mandelstam m;
		m.s = (p1 + p2) * (p1 + p2);
		m.t = (p1 - p3) * (p1 - p3);
		m.u = (p1 - p4) * (p1 - p4);
		return m;
	}

	// Calculate |M|^2 
	inline double evaluateMatrixElement(const std::array<FourVector, NPARTICLES> &momenta)
	{
		Mandelstam mandelstam = calculateMandelstam(momenta[0], momenta[1], momenta[2], momenta[3]);

		return matrixElement.evaluate(mandelstam.s, mandelstam.t, mandelstam.u);
	}

	/* Evaluate the statistical "population factor", eq (A3) in 2204.13120. See published version since arxiv v1 is wrong.
	delfaF[i] is the out-of-equilibrium part of distribution funct. of particle i */
	double evaluatePopulationFactor(const std::array<FourVector, NPARTICLES> &momenta, 
		const std::array<double, NPARTICLES> &deltaF)
	{

		const double f1 = particles[0]->fEq( momenta[0].energy() );
		const double f2 = particles[1]->fEq( momenta[1].energy() );
		const double f3 = particles[2]->fEq( momenta[2].energy() );
		const double f4 = particles[3]->fEq( momenta[3].energy() );
		
		double res =  static_cast<int>(bDeltaF[0]) * std::exp(momenta[1].energy()) * deltaF[0] / (f1*f1)
					+ static_cast<int>(bDeltaF[1]) * std::exp(momenta[0].energy()) * deltaF[1] / (f2*f2)
					- static_cast<int>(bDeltaF[2]) * std::exp(momenta[3].energy()) * deltaF[2] / (f3*f3)
					- static_cast<int>(bDeltaF[3]) * std::exp(momenta[2].energy()) * deltaF[3] / (f4*f4);

		res = res * f1*f2*f3*f4;
		return res;
	}

	// Calculate matrix element times population factor for this 2->2 process 
	inline double evaluate(const std::array<FourVector, NPARTICLES> &momenta,
		const std::array<double, NPARTICLES> &deltaF)
	{

		return evaluateMatrixElement(momenta) * evaluatePopulationFactor(momenta, deltaF);
	}

	inline bool isUltrarelativistic() const { return bUltrarelativistic; }

public:

	// Particle 0 is the 'incoming' one whose momentum is kept fixed to p1
	std::array<std::shared_ptr<ParticleSpecies>, NPARTICLES> particles;

	// Parsed matrix element for this process
	MatrixElement matrixElement;

	// Collision element is ultrarelativistic if all its EXTERNAL particles are
	bool bUltrarelativistic = false;
};

} // namespace

#endif // header guard