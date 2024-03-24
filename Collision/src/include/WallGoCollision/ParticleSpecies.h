#ifndef PARTICLESPECIES_H
#define PARTICLESPECIES_H

#include <string>
#include <cmath>
#include <iostream>

namespace wallgo
{

// Boson or fermion?
enum class EParticleType {
	BOSON, FERMION
};


class ParticleSpecies {

public: 

	ParticleSpecies(std::string speciesName, EParticleType particleType, bool speciesInEquilibrium, double msqVacuum, double msqThermal, bool ultrarelativistic)
		: type(particleType),
		name(speciesName),
		bInEquilibrium(speciesInEquilibrium),
		bUltrarelativistic(ultrarelativistic)
	{
		setVacuumMassSquared(msqVacuum);
		setThermalMassSquared(msqThermal);
	}

	bool isUltrarelativistic() const { return bUltrarelativistic; }
	bool isInEquilibrium() const { return bInEquilibrium; }

	std::string getName() const { return name; }

    double getVacuumMassSquared() const { return vacuumMassSquared; }
    double getThermalMassSquared() const { return thermalMassSquared; }

	// Set the non-thermal part of particle's mass squared. Needs to be in units of temperature
	void setVacuumMassSquared(double msq)
	{
		vacuumMassSquared = msq;
	}

	// Set thermal mass squared. Needs to be in units of temperature
	void setThermalMassSquared(double msq)
	{
		thermalMassSquared = msq;
	}
	

	// Equilibrium distribution function for the particle species
	double fEq(double energy) const
	{
		if (type == EParticleType::BOSON)
		{
			return 1.0 / (std::exp(energy) - 1.0 + 1e-50);
		} 
		else
		{
			return 1.0 / (std::exp(energy) + 1.0);
		}
	}

private:

	const EParticleType type;
	const std::string name;
	
	// Is the particle assumed to be in thermal equilibrium?
	const bool bInEquilibrium;
	// Neglect mass in dispersion relations or not?
	const bool bUltrarelativistic;

	double vacuumMassSquared;
	double thermalMassSquared;
};

} // namespace

#endif