/*******/

#include <iostream>
#include <cmath>
#include <chrono>

#include <cstring>
#include <getopt.h> // command line arguments

#include "CollElem.h"
#include "CollisionIntegral.h"
#include "hdf5Interface.h"
#include "gslWrapper.h"


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
}

// Count how many independent collision integrals there are for basis with N polynomials
long countIndependentIntegrals(int N) {
     long count = (N-1)*(N-1)*(N-1)*(N-1);
     // C[Tm(-x), Tn(y)] = (-1)^m C[Tm(x), Tn(y)]
     count = std::ceil(count / 2.0);
     // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
     if (N % 2 == 0) {
          // how many odd m?
          long mOdd = N / 2;
          count -= mOdd;
     }

     return count;
}

// Temporary routine for illustrating how we can generate all collision terms + write them to hdf5 file
void calculateAllCollisions(CollisionIntegral4 &collisionIntegral) {

	int gridSizeN = collisionIntegral.getPolynomialBasisSize();

	Array4D collGrid(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);
	Array4D collGridErrors(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);

	std::cout << "Now evaluating all collision integrals\n" << std::endl;
	// Note symmetry: C[Tm(-rho_z), Tn(rho_par)] = (-1)^m C[Tm(rho_z), Tn(rho_par)]
	// which means we only need j <= N/2

	// m,n = Polynomial indices
	#pragma omp parallel for collapse(4) firstprivate(collisionIntegral)
	for (int m = 2; m <= gridSizeN; ++m)
	for (int n = 1; n <= gridSizeN-1; ++n) {
		// j,k = grid momentum indices
		for (int j = 1; j <= gridSizeN/2; ++j)
		for (int k = 1; k <= gridSizeN-1; ++k) {

		// Monte Carlo result for the integral + its error
		std::array<double, 2> resultMC;

			// Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
			if (2*j == gridSizeN && m % 2 != 0) {
				resultMC[0] = 0.0;
				resultMC[1] = 0.0;
			} else {
				resultMC = collisionIntegral.evaluate(m, n, j, k);
			}

			collGrid[m-2][n-1][j-1][k-1] = resultMC[0];
			collGridErrors[m-2][n-1][j-1][k-1] = resultMC[1];

		printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);

		} // end j,k
	} // end m,n

	// Fill in the j > N/2 elements
	#pragma omp parallel for collapse(4)
	for (int m = 2; m <= gridSizeN; ++m)
	for (int n = 1; n <= gridSizeN-1; ++n) {
		for (int j = gridSizeN/2+1; j <= gridSizeN-1; ++j)
		for (int k = 1; k <= gridSizeN-1; ++k) {
			int jOther = gridSizeN - j;
			int sign = (m % 2 == 0 ? 1 : -1);
			collGrid[m-2][n-1][j-1][k-1] = sign * collGrid[m-2][n-1][jOther-1][k-1];
			collGridErrors[m-2][n-1][j-1][k-1] = sign * collGridErrors[m-2][n-1][jOther-1][k-1];
		}
	}

	// Create a new HDF5 file. H5F_ACC_TRUNC means we overwrite the file if it exists
	std::string filename = "collisions_N" + std::to_string(gridSizeN) + ".hdf5";
	H5::H5File h5File(filename, H5F_ACC_TRUNC);

	H5Metadata metadata;
	metadata.basisSize = gridSizeN;
	metadata.basisName = "Chebyshev";
	metadata.integrator = "Vegas Monte Carlo (GSL)";


	writeMetadata(h5File, metadata);

	writeDataSet(h5File, collGrid, "top");
	writeDataSet(h5File, collGridErrors, "top errors");

	h5File.close();
}

//***************


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

	// Parse command line arguments
	int opt;
	while ((opt = getopt(argc, argv, "w")) != -1) {
		switch (opt) {
			case 'h':
				// Print usage and exit
				printUsage(stderr, argv[0]);
				return 0;
			case 'w':
				std::cout << "== Running HDF5 output test ==\n";
				testHDF5();
				return 0;
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
	// 2->2 scatterings so 4 external particles
	using CollisionElement = CollElem<4>;

	//**** Masses squared. These need to be in units of temperature, ie. (m/T)^2 **//
	// Thermal
	double mq2 = 0.251327; // quark
	double mg2 = 3.01593; // SU(3) gluon
	// Vacuum
	// TODO if needed
	double msqVacuum = 0.0;


	// define particles that we include in matrix elements
	ParticleSpecies topQuark("top", EParticleType::FERMION, false, msqVacuum, mq2);
	ParticleSpecies lightQuark("quark", EParticleType::FERMION, true, msqVacuum, mq2);
	ParticleSpecies gluon("gluon", EParticleType::BOSON, true, msqVacuum, mg2);


	// Then create collision elements for 2->2 processes involving these.
	// By 'collision element' I mean |M|^2 * P[ij -> nm], where P is the population factor involving distribution functions.
	// Right now the correct matrix elements are hardcoded in for the top quark.
	// In a real model-independent calculation this needs to either calculate the matrix elements itself (hard, probably needs its own class)
	// or read them in from somewhere.
	CollisionElement tt_gg({ topQuark, topQuark, gluon, gluon });
	CollisionElement tg_tg({ topQuark, gluon, topQuark, gluon });
	CollisionElement tq_tq({ topQuark, lightQuark, topQuark, lightQuark });

	const int basisSizeN = 5;

	CollisionIntegral4 collInt(basisSizeN);
	collInt.addCollisionElement(tt_gg);
	collInt.addCollisionElement(tg_tg);
	collInt.addCollisionElement(tq_tq);

	// How many collision terms do we need in total
	int nCollisionTerms = countIndependentIntegrals(basisSizeN);

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

	// This will  calculate all required collision terms:
	calculateAllCollisions(collInt);

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
