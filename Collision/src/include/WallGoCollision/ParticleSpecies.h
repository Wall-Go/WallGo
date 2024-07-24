#pragma once

#include <string>
#include <cmath>
#include <functional>
#include <cassert>

#include "ModelParameters.h"
#include "EnvironmentMacros.h"

namespace wallgo
{

// Boson or fermion?
enum class WALLGO_API EParticleType
{
	eNone, eBoson, eFermion
};


// Data-only container for describing a particle species
struct ParticleDescription
{
	// Must be unique
	std::string name = "Unknown";
	// Must be unique
	uint32_t index = 0;

	EParticleType type = EParticleType::eNone;

	// Is the particle assumed to be in thermal equilibrium?
	bool bInEquilibrium = false;
	// Neglect mass in dispersion relations or not?
	bool bUltrarelativistic = true;

	/* Function f(x) -> double that calculates mass squared of the particle.
	Must be in units of the temperature, ie. this should return m^2 / T^2.
	This mass will be used in dispersion relations when computing particle energies for collision integration.
	If the bUltrarelativistic flag is set, the mass function will not be called. */
	std::function<double(const ModelParameters&)> massSqFunction;
};


class ParticleSpecies
{
public:

	ParticleSpecies() {}
	ParticleSpecies(const ParticleDescription& description)
		: mDescription(description) {}

	// Computes particle mass squared
	double computeMassSquared(const ModelParameters& params) const
	{
		assert(mDescription.massSqFunction && "Particle mass function has not been set");
		return mDescription.massSqFunction(params);
	}

	/* Computes particle mass squared, possibly utilizing the cached value to avoid the computation altogether.
	*/
	double computeMassSquared(
		const ModelParameters& params,
		bool bCanUseCachedValue,
		bool bShouldCache)
	{
		if (bCanUseCachedValue) return mCachedMass;

		double res = computeMassSquared(params);
		mCachedMass = bShouldCache ? res : mCachedMass;
		
		return res;
	}

	inline double getCachedMassSquared() const { return mCachedMass; }

	ParticleDescription getDescription() const { return mDescription; }
	inline bool isUltrarelativistic() const { return mDescription.bUltrarelativistic; }
	inline bool isInEquilibrium() const { return mDescription.bInEquilibrium; }
	inline std::string_view getName() const { return mDescription.name; }
	inline EParticleType getStatistics() const { return mDescription.type; }

	// Equilibrium distribution function for the particle species
	double fEq(double energy) const
	{
		assert(mDescription.type != EParticleType::eNone && "Particle statistics type was None");

		if (mDescription.type == EParticleType::eBoson)
		{
			return 1.0 / (std::exp(energy) - 1.0 + 1e-50);
		}
		else
		{
			return 1.0 / (std::exp(energy) + 1.0);
		}
	}

private:

	ParticleDescription mDescription;

	// Cache the mass to avoid unnecessary re-evaluations of the mass function
	double mCachedMass = 0.0;
};

} // namespace
