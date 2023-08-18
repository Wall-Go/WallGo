#ifndef PARTICLESPECIES_H
#define PARTICLESPECIES_H

#include <string>
#include <cmath>
#include <iostream>

// Boson or fermion?
enum class EParticleType {
	BOSON, FERMION
};


class ParticleSpecies {

public: 

	ParticleSpecies(std::string speciesName, EParticleType particleType, bool speciesInEquilibrium, 
            double msqVacuum, double msqThermal) : type(particleType)  
	{
		name = speciesName;
        bInEquilibrium = speciesInEquilibrium;
		vacuumMassSquared = msqVacuum;
		thermalMassSquared = msqThermal;
	}

	inline bool isUltrarelativistic() const { return bUltrarelativistic; }
	inline bool isInEquilibrium() const { return bInEquilibrium; }

	inline std::string getName() const { return name; }

    inline double getVacuumMassSquared() const { return vacuumMassSquared; }
    inline double getThermalMassSquared() const { return thermalMassSquared; }


	// Equilibrium distribution function for the particle species
	double fEq(double energy) const {
		double res = 0.0;
		if (type == EParticleType::BOSON) {
			// TODO better cutoff
			res = 1.0 / (std::exp(energy) - 1.0 + 1e-6);
		} else {
			res = 1.0 / (std::exp(energy) + 1.0);
		}
		return res;
	}

private:
	// TODO setters for these 
	// Neglect mass in dispersion relations or not? (this flag is not used ATM)
	bool bUltrarelativistic = true;
	// Is the particle assumed to be in thermal equilibrium?
	bool bInEquilibrium;

	double vacuumMassSquared;
	double thermalMassSquared;

	std::string name;

	// Set this in constructor using initialization list
	const EParticleType type;
};

#endif