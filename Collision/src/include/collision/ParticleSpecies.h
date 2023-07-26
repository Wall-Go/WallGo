#ifndef PARTICLESPECIES_H
#define PARTICLESPECIES_H

#include <string>
#include <cmath>

// Boson or fermion?
enum class EParticleType {
	BOSON, FERMION
};


class ParticleSpecies {

public: 

	// We prolly don't want a default constructor - the user should at the very least specify particle type

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
			res = 1.0 / (std::exp(energy) - 1.0 + 1e-6);
		} else {
			res = 1.0 / (std::exp(energy) + 1.0);
		}
		return res;
	}

	inline double getDeltaF() const { return deltaF; }
	// Set out-of-equilibrium part of distribution function
	void setDeltaF(double df) {
		if (bInEquilibrium) {
			std::cerr << "Warning: setDeltaF() called with bInEquilibrium = true\n";
			return;
		}
		deltaF = df;
	}

public:
	// TODO setters for these 
	// Neglect mass in dispersion relations or not? (this flag is not used ATM)
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

#endif