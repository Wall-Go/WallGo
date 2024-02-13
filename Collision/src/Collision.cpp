#include "Collision.h"

#include <array>
#include <fstream>
#include <regex> // Reading matrix elements from file
#include <algorithm> // std::remove_if
#include <chrono>

#include "ConfigParser.h"

#if WITH_OMP
    #include <omp.h>
#endif



// Global function for this file only. Processes string of form "M[a,b,c,d] -> some funct" and stores in the arguments
void interpretMatrixElement(const std::string &inputString, std::vector<uint> &indices, std::string &mathExpression)
{
    // First split the string by "->""
    std::vector<std::string> tokens(2);
    
    std::string delimiter = "->";
    std::string lhs = inputString.substr(0, inputString.find(delimiter));

    // RHS
    mathExpression = inputString.substr(lhs.length() + delimiter.length());

    // remove whitespaces from lhs to avoid weirdness
    lhs.erase(std::remove_if(lhs.begin(), lhs.end(), isspace), lhs.end());

    // ---- Extract the abcd indices from M[a,b,c,d]
    std::size_t start = lhs.find('[');
    std::size_t end = lhs.find(']');

    // Ensure '[' and ']' are found and the start position is before the end position
    if (start != std::string::npos && end != std::string::npos && start < end) 
    {
        std::string values = lhs.substr(start + 1, end - start - 1);

        // Use stringstream to tokenize and extract integers
        std::istringstream ss(values);
        indices.clear();
        indices.reserve(4);
        int num;

        while (ss >> num) 
        {
            indices.push_back(num);

            // Check for the ',' separator and ignore it
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
    }
}


Collision::Collision(uint basisSize) : basisSizeN(basisSize)
{
}

void Collision::addParticle(const ParticleSpecies &particle)
{
    particles.push_back(particle);

    massSquares.push_back(particle.getThermalMassSquared() + particle.getVacuumMassSquared());

    if (bMatrixElementsDone) 
    {
        std::cerr << "Warning: New particle added after parsing of matrix elements. You probably do NOT want to do this." << std::endl;
    }
}

void Collision::addCoupling(double coupling) 
{
    couplings.push_back(coupling);
}


void Collision::evaluateCollisionTensor(CollisionIntegral4 &collisionIntegral, Array4D &results, Array4D &errors)
{
    const uint N = basisSizeN;
    results = Array4D(N-1, N-1, N-1, N-1, 0.0);
    errors = Array4D(N-1, N-1, N-1, N-1, 0.0);

    ConfigParser& config = ConfigParser::get();
    const bool bVerbose = config.getBool("Integration", "verbose");


    // Note symmetry: C[Tm(-rho_z), Tn(rho_par)] = (-1)^m C[Tm(rho_z), Tn(rho_par)]
	// which means we only need j <= N/2

    // each thread needs its collisionIntegral because the operations for computing the integrand are not thread safe!
	#pragma omp parallel firstprivate(collisionIntegral)
    {
        int numThreads = 1;
        int threadID = 0;
    #if WITH_OMP
        numThreads = omp_get_num_threads();
        threadID = omp_get_thread_num();
    #endif 

        // Progress tracking

        // How many we've calculated inside this function only (so for this out-of-eq pair) 
        int localIntegralCount = 0;
        // Report when thread0 has computed this many integrals. NB: totalIntegralCount is the full count including all out-of-eq pairs
        int standardProgressInterval = totalIntegralCount / 10 / numThreads; // every 10%
        standardProgressInterval = globalFuncts::clamp<int>(standardProgressInterval, initialProgressInterval, totalIntegralCount); // but not more frequently than this

        int progressReportInterval = ( bFinishedInitialProgressCheck ? standardProgressInterval : initialProgressInterval );

        #pragma omp for collapse(4) 
        // m,n = Polynomial indices
        for (uint m = 2; m <= N; ++m)
        for (uint n = 1; n <= N-1; ++n)
        {
            // j,k = grid momentum indices 
            for (uint j = 1; j <= N/2; ++j)
            for (uint k = 1; k <= N-1; ++k)
            {
            
                // Monte Carlo result for the integral + its error
                std::array<double, 2> resultMC;

                // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
                if (2*j == N && m % 2 != 0)
                {
                    resultMC[0] = 0.0;
                    resultMC[1] = 0.0;
                }
                else
                {
                    resultMC = collisionIntegral.evaluate(m, n, j, k);
                }

                results[m-2][n-1][j-1][k-1] = resultMC[0];
                errors[m-2][n-1][j-1][k-1] = resultMC[1];

                localIntegralCount++;

                if (bVerbose)
                {
                    printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);
                }

                if (threadID == 0 && (localIntegralCount % progressReportInterval == 0)) 
                {
                    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
                    elapsedTime = currentTime - startTime;

                    // HACK: could not figure out how to nicely sync the counts from all threads to correctly update computedIntegralCount. 
                    // Here I extrapolate from thread0 to estimate the progress, then undo the change afterwards. 
                    // Correct count is calculated at the end of this function
                    const int backupCount = computedIntegralCount;
                    computedIntegralCount += localIntegralCount * numThreads;
                    computedIntegralCount = globalFuncts::clamp<int>(computedIntegralCount, localIntegralCount, totalIntegralCount);

                    reportProgress();
                    computedIntegralCount = backupCount;

                    // Update report interval (it starts at initialProgressInterval)
                    progressReportInterval = standardProgressInterval;
                }

                // Check if we received instructions to stop
                if (!shouldContinueEvaluation())
                {
                    std::exit(20);
                }

            } // end j,k
        } // end m,n
        
    } // end #pragma omp parallel 

	// Fill in the j > N/2 elements
	#pragma omp parallel for collapse(4)
	for (uint m = 2; m <= N; ++m) 
	for (uint n = 1; n <= N-1; ++n) 
    {
		for (uint j = N/2+1; j <= N-1; ++j)
		for (uint k = 1; k <= N-1; ++k) 
        {

			uint jOther = N - j;
			int sign = (m % 2 == 0 ? 1 : -1);
            
			results[m-2][n-1][j-1][k-1] = sign * results[m-2][n-1][jOther-1][k-1];
			errors[m-2][n-1][j-1][k-1] = sign * errors[m-2][n-1][jOther-1][k-1];
		}
	}

    // How many we calculated in this function. 
    // Just recalculate this here instead of trying to combine counts from many threads and manually count j > N/2 cases etc
    computedIntegralCount += countIndependentIntegrals(basisSizeN, 1);        
}


void Collision::calculateCollisionIntegrals()
{
    // Particle list is assumed to be fixed from now on!
    findOutOfEquilibriumParticles();
    makeParticleIndexMap();

    // Initialize progress tracking 
    totalIntegralCount = countIndependentIntegrals(basisSizeN, outOfEqParticles.size());
    computedIntegralCount = 0;
    bFinishedInitialProgressCheck = false;
    startTime = std::chrono::steady_clock::now();

    // make rank 2 tensor that mixes out-of-eq particles (each element is a collision integral, so actually rank 6, but the grid indices are irrelevant here)
    for (const ParticleSpecies& particle1 : outOfEqParticles) 
    {
        for (const ParticleSpecies& particle2 : outOfEqParticles)
        {

            std::chrono::steady_clock::time_point pairStartTime = std::chrono::steady_clock::now();

            // integration results/errors on the grid
            Array4D results;
            Array4D errors;
            
            CollisionIntegral4 collisionIntegral(basisSizeN);
            std::vector<CollElem<4>> collisionElements = makeCollisionElements(particle1.getName(), particle2.getName());
            
            for (const CollElem<4> &elem : collisionElements)
            {
                collisionIntegral.addCollisionElement(elem);
            }

            evaluateCollisionTensor(collisionIntegral, results, errors);

            // Create a new HDF5 file. H5F_ACC_TRUNC means we overwrite the file if it exists
            std::string filename = "collisions_" + particle1.getName() + "_" + particle2.getName() + "_N" + std::to_string(basisSizeN) + ".hdf5";
            H5::H5File h5File(filename, H5F_ACC_TRUNC);

            H5Metadata metadata;
            metadata.basisSize = basisSizeN;
            metadata.basisName = "Chebyshev";
            metadata.integrator = "Vegas Monte Carlo (GSL)";


            writeMetadata(h5File, metadata);

            writeDataSet(h5File, results, particle1.getName() + ", " + particle2.getName());
            writeDataSet(h5File, errors, particle1.getName() + ", " + particle2.getName() + " errors");
            
            h5File.close();

            // How long did this all take
            std::chrono::duration<double> duration = std::chrono::steady_clock::now() - pairStartTime;
            double seconds = duration.count();
            int hours = seconds / 3600;
            // leftover mins
            int minutes = (seconds - hours * 3600) / 60;
            std::cout << "[" << particle1.getName() << ", " << particle2.getName() << "] done in " << hours << "h " << minutes << "min." << std::endl;

        }
    }
    
}

std::vector<CollElem<4>> Collision::makeCollisionElements(const std::string &particleName1, const std::string &particleName2)
{
    // Just for logging 
    std::string pairName = "[" + particleName1 + ", " + particleName2 + "]"; 

    // @todo should actually check if these are in outOfEquilibriumParticles vector
    if (particleIndex.count(particleName1) < 1 || particleIndex.count(particleName2) < 1 ) 
    {
        std::cerr << "Error: particles missing from list! Was looking for out-of-eq particles " << pairName << "\n";
        exit(9);
    }

    // file paths are of form MatrixElements/matrixElements_top_gluon etc
    const std::string fileName = ConfigParser::get().getString("MatrixElements", "fileName");
    const bool bVerbose = ConfigParser::get().getBool("MatrixElements", "verbose");

    if (bVerbose) std::cout << "\n" <<"Parsing matrix elements for off-equilibrium pair " << pairName << "\n";
    
    std::ifstream matrixElementFile(fileName);

    // M_ab -> cd, with a = particle1 and at least one of bcd is particle2. Suppose that for each out-of-eq pair, Mathematica gives these in form 
    // M[a, b, c, d] -> (some symbolic expression), where abcd are integer indices that need to match our ordering in particleIndex map
    // Here we parse the lhs to extract indices, then parse the rhs as a math expression (function of s,t,u and couplings/masses). 
    // For each of these we make a CollElem<4> object with correct deltaF structure which we infer from the indices 

    std::vector<CollElem<4>> collisionElements;

    if (!matrixElementFile.is_open()) {
        std::cerr << "!!! Error: Failed to open matrix element file " << fileName << std::endl;
        exit(10);
    }

    /* Now use regex to read all lines of form M[...] -> expr
    For each line we check if the first particle index matches that of particleName1
    and require that at least one other index matches that of particleName2.
    This is not optimal because we end up reading the full file for each off-eq pair.
    */
    std::string line;
    std::string expr;
    std::vector<uint> indices;
    indices.resize(4);
    
    while (std::getline(matrixElementFile, line)) {
        if (std::regex_search(line, std::regex("M\\[.*\\] -> (.*)"))) {

            interpretMatrixElement(line, indices, expr);
            
            if (indices[0] != particleIndex[particleName1]) continue;
            if ( std::find(indices.begin(), indices.end(), particleIndex[particleName2]) == indices.end() ) continue;
            
            CollElem<4> newElem = makeCollisionElement(particleName2, indices, expr);
            collisionElements.push_back(newElem);

            if (bVerbose)
            {
                std::cout << "Loaded matrix element:\n";
                std::cout << line << "\n";
            }
        }
    }

    matrixElementFile.close();

    std::cout << "\n";
    bMatrixElementsDone = true;

    return collisionElements;
}

CollElem<4> Collision::makeCollisionElement(const std::string &particleName2, const std::vector<uint> &indices, const std::string &expr)
{
    assert(indices.size() == 4);

    const ParticleSpecies p1 = particles[indices[0]];
    const ParticleSpecies p2 = particles[indices[1]];
    const ParticleSpecies p3 = particles[indices[2]];
    const ParticleSpecies p4 = particles[indices[3]];
    
    CollElem<4> collisionElement( { p1, p2, p3, p4} );

    collisionElement.matrixElement.initParser(couplings, massSquares);
    // Parses the RHS math expression so that we can evaluate it as a function of the symbols
    collisionElement.matrixElement.setExpression(expr);
    // Set deltaF flags: in general there can be 4 deltaF terms but we only take the ones with deltaF of particle2
    for (uint i = 0; i < collisionElement.bDeltaF.size(); ++i) 
    {
        collisionElement.bDeltaF[i] = (collisionElement.particles[i].getName() == particleName2);
    }
    
    return collisionElement;
}


long Collision::countIndependentIntegrals(uint basisSize, uint outOfEqCount)
{
    const uint N = basisSize;
    // How many independent integrals in each CollisionIntegral4
    long count = (N-1)*(N-1)*(N-1)*(N-1);
    // C[Tm(-x), Tn(y)] = (-1)^m C[Tm(x), Tn(y)]
    count = std::ceil(count / 2.0);
    // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
    if (N % 2 == 0) {
        // how many odd m?
        long mOdd = N / 2;
        count -= mOdd;
    } 

    // this was for 1 deltaF particle, for more than one we have mixing terms
    return count * outOfEqCount*outOfEqCount;
}


void Collision::reportProgress()
{
    if (totalIntegralCount > 0)
    {
        double elapsedSeconds = elapsedTime.count();
        double timePerIntegral = elapsedSeconds / computedIntegralCount;
        double timeRemaining = timePerIntegral * (totalIntegralCount - computedIntegralCount);

        int percentage = 100 * static_cast<double>(computedIntegralCount) / totalIntegralCount;
        std::cout << "Integral progress: " << computedIntegralCount << " / " << totalIntegralCount << " (" << percentage << "%). "; 
        std::cout << "Estimated time remaining: " << std::floor(timeRemaining / 3600) << "h " << (int(timeRemaining) % 3600 ) / 60 << "min" << std::endl;
    }
    bFinishedInitialProgressCheck = true;
}


void Collision::makeParticleIndexMap()
{
    particleIndex.clear();
    
    uint i = 0;
    for (const ParticleSpecies& particle : particles) 
    {
        particleIndex[particle.getName()] = i;
        i++; 
    }
}

void Collision::findOutOfEquilibriumParticles()
{
    outOfEqParticles.clear();

    for (const ParticleSpecies &particle : particles) 
    {
        if (!particle.isInEquilibrium())
        {
            outOfEqParticles.push_back(particle);
        }
    }
}