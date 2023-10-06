/*******/

#include <iostream>
#include <cmath>
#include <chrono>

#include <cstring>
#include <getopt.h> // command line arguments

#include "Collision.h"
#include "CollElem.h"
#include "CollisionIntegral.h"
#include "hdf5Interface.h"
#include "gslWrapper.h"
#include "MatrixElement.h"

// Print a description of all supported options
void printUsage(FILE *fp, const char *path) {

	// Take only the last portion of the path
	const char *basename = std::strrchr(path, '/');
	basename = basename ? basename + 1 : path;

	fprintf (fp, "usage: %s [OPTIONS]\n", basename);
	fprintf (fp, "Available options:\n");
	fprintf (fp, "  -h\t\t"
				"Print this help and exit.\n");

	fprintf (fp, "  -w\t\t"
				"Test the hdf5 output routines by writing dummy data and exit.\n");
	fprintf (fp, "  -t\t\t"
				"Do a short test run and exit. Useful for profiling\n");
}


//***************

/* Test/example routine, calculates all collision integrals with QCD interactions. 
The structure here illustrates how the same could be done from Python with arbitrary inputs */ 
void collisionsQCD(uint N) {

	const double gs = 1.2279920495357861;

    //**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
	// Thermal
	const double mq2 = 0.251327; // quark
	const double mg2 = 3.01593; // SU(3) gluon
	// Vacuum, @TODO if needed
	const double msqVacuum = 0.0;

	// take top and gluon out-of-eq
    ParticleSpecies topQuark("top", EParticleType::FERMION, false, msqVacuum, mq2);
    ParticleSpecies gluon("gluon", EParticleType::BOSON, false, msqVacuum, mg2);
	ParticleSpecies lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2);

	// Main control object
	Collision collision(N);
	collision.addParticle(topQuark);
	collision.addParticle(gluon);
	collision.addParticle(lightQuark);

	collision.addCoupling(gs);

	collision.calculateCollisionIntegrals();

}


int main(int argc, char *argv[]) {

	//--------------- How this works 

	/* class CollElem : Describes a matrix element with fixed external particles (ordering matters!). 
	* This is the object that calculates |M|^2 and the statistical 'population factor' P once the external momenta are fixed. 
	* We need a separate CollElem object for each scattering process that contributes to the collision integral (tt->gg, tg->tg, tq->tq are separate CollElems)
	* Currently the matrix elements are just hard coded, in a more realistic setting they would probably be read from elsewhere. */

	/* class ParticleSpecies : Quite self-explanatory. Contains info about particle statistics, masses and whether the particle species stays in equilibrium, etc.
	* These are given as inputs to CollElem when constructing CollElem objects. 
	* The particle name property is important as it is used to read in the correct matrix element (this needs improvement in the future). */

	/* class CollisionIntegral4 : This describes the whole 2-by-2 collision integral for a given particle type (top quark in this case). 
	* IE: this object calculates eq. (A1) in 2204.13120, with delta f replace with Chebyshev polynomials.
	* Therefore the class it needs to know the size of our polynomial basis (N) and CollElem objects that make up the integral. 
	* The class calculates 5D integrals with same integration variables as Benoit had. See notes in the private repo. */

	/* Currently the interface between CollisionIntegral4 and CollElem is not optimal and there is some redundancy 
	* in how the CollisionIntegral4 obtains particle masses etc. This needs to be improved in next version before we 
	* get started with generic matrix elements */

	//---------------

	bool bDoTestRun = false;

	// Parse command line arguments
	int opt;
	while ((opt = getopt(argc, argv, "wt")) != -1) {
		switch (opt) {
			case 'h':
				// Print usage and exit
				printUsage(stderr, argv[0]);
				return 0;
			case 'w':
				std::cout << "== Running HDF5 output test ==\n";
				testHDF5();
				return 0;
			case 't':
				std::cout << "== Running short test run ==\n";
				bDoTestRun = true;
				break;
			case '?':
				if (isprint(opt))
						fprintf(stderr, "Unknown option `-%c'.\n", opt);
				else
						fprintf(stderr, "Unknown option character `\\x%x'.\n", opt);
				return 1;
			default:
				abort();
		}
	}

	gslWrapper::initializeRNG();

	if (bDoTestRun) {
		collisionsQCD(5);
		return 0;
	}



	collisionsQCD(20);

/*
	//-------------------- Measure wall clock time

	std::cout << "Running speed test: integral C[2,1,1,1]\n";
	auto startTime = std::chrono::steady_clock::now();
	collInt.evaluate(2, 1, 1, 1);

	auto endTime = std::chrono::steady_clock::now();

	auto elapsedTime = endTime - startTime;
	// Convert the elapsed time to milliseconds
	auto elapsedTimeMs = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count();

	// How long for all collision integrals
	auto totalTime = elapsedTime * nCollisionTerms;

	auto hours = std::chrono::duration_cast<std::chrono::hours>(totalTime).count();
	auto minutes = std::chrono::duration_cast<std::chrono::minutes>(totalTime).count() % 60;

	std::cout << "Test done, took " << elapsedTimeMs << "ms\n";

	std::cout << "Estimated time-per-thread for all " << nCollisionTerms << " collision integrals: " 
			<< hours << " hours " << minutes << " minutes\n";

	//--------------------
*/

/*

	// FOR PROFILING: just calculate a few terms and exit

	std::array<double, 2> resultMC;


	int m, n, j, k;

	m = 2; n = 1; j = 1; k = 1;

	resultMC = collInt.evaluate(m, n, j, k);
	printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);

	m = 6; n = 4; j = 11; k = 9;
	resultMC = collInt.evaluate(m, n, j, k);

	printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);
*/

	// Cleanup
	gslWrapper::clearRNG();
	return 0;
}
