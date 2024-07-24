#pragma once

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

/* Describes collision process of N external particles; one term in a collision integral.
Contains a matrix element and info about statistics of the colliding particles. */
template <size_t NPARTICLES>
class WALLGO_API CollisionElement
{

public:

	/* Constructor, will store a copy, not reference, of the matrix element.
	The bDeltaF array specifies which delta_f terms are included when computing the statistical population factor. */
	CollisionElement(
		const std::array<const ParticleSpecies*, NPARTICLES>& externalParticles,
		const MatrixElement& matrixElement,
		const std::array<bool, NPARTICLES> bDeltaF) 
		: mExternalParticles(externalParticles),
		mMatrixElement(matrixElement),
		mDeltaF(bDeltaF)
	{
		bool bAllUltrarelativistic = true;
		for (const ParticleSpecies* p : mExternalParticles)
		{
			assert(p && "CollisionElement received nullptr particle");
			
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


	/* Evaluate the statistical "population factor", eq (A3) in 2204.13120. See published version since arxiv v1 is wrong.
	delfaF[i] is the out-of-equilibrium part of distribution funct. of particle i.
	TODO: need to template specialize this if generalizing the class to work also with 3 external particles (or any other number) */
	double evaluatePopulationFactor(
		const std::array<FourVector, NPARTICLES> &momenta,
		const std::array<double, NPARTICLES> &deltaF) const
	{

		const double f1 = mExternalParticles[0]->fEq( momenta[0].energy() );
		const double f2 = mExternalParticles[1]->fEq( momenta[1].energy() );
		const double f3 = mExternalParticles[2]->fEq( momenta[2].energy() );
		const double f4 = mExternalParticles[3]->fEq( momenta[3].energy() );
		
		double res =  static_cast<int>(mDeltaF[0]) * std::exp(momenta[1].energy()) * deltaF[0] / (f1*f1)
					+ static_cast<int>(mDeltaF[1]) * std::exp(momenta[0].energy()) * deltaF[1] / (f2*f2)
					- static_cast<int>(mDeltaF[2]) * std::exp(momenta[3].energy()) * deltaF[2] / (f3*f3)
					- static_cast<int>(mDeltaF[3]) * std::exp(momenta[2].energy()) * deltaF[3] / (f4*f4);

		res = res * f1*f2*f3*f4;
		return res;
	}

	// Calculate |M|^2 
	inline double evaluateMatrixElement(const std::array<FourVector, NPARTICLES>& momenta)
	{
		const Mandelstam mandelstam = calculateMandelstam(momenta[0], momenta[1], momenta[2], momenta[3]);

		return mMatrixElement.evaluate(mandelstam);
	}

	// Calculate matrix element times population factor for this process 
	inline double evaluate(
		const std::array<FourVector, NPARTICLES> &momenta,
		const std::array<double, NPARTICLES> &deltaF)
	{
		return evaluateMatrixElement(momenta) * evaluatePopulationFactor(momenta, deltaF);
	}

	inline bool isUltrarelativistic() const { return bUltrarelativistic; }

private:

	/* External particles. Order matters as we associate particles[i] with momentum p[i] when evaluating the CollisionElement.
	Note that we store raw pointers to particles: idea is that we obtain these from the model. */
	std::array<const ParticleSpecies*, NPARTICLES> mExternalParticles;

	// Parsed matrix element for this process
	MatrixElement mMatrixElement;

	// Which delta_f terms are nonzero in the population factor
	std::array<bool, NPARTICLES> mDeltaF;

	// Collision element is ultrarelativistic if all its EXTERNAL particles are
	bool bUltrarelativistic = false;

	// Collision integrand computations currently need direct access to our private members
	friend class CollisionIntegral4;
};

} // namespace
