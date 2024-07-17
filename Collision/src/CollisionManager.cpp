#include "CollisionManager.h"

#include <array>
#include <fstream>
#include <sstream>
#include <regex> // Reading matrix elements from file
#include <algorithm> // std::remove_if
#include <chrono>
#include <filesystem>

namespace wallgo
{

// Function for this file only. Processes string of form "M[a,b,c,d] -> some funct" and stores in the arguments
void interpretMatrixElement(const std::string &inputString, std::vector<size_t> &indices, std::string &mathExpression)
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
        size_t num;

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

CollisionManager::CollisionManager()
{
    // Set default options

    basisSize = 0;

    outputDirectory = std::filesystem::current_path();
    matrixElementFile = std::filesystem::path("MatrixElements.txt");

    integrationOptions.calls = 50000;
    integrationOptions.maxIntegrationMomentum = 20;
    integrationOptions.maxTries = 50;
    integrationOptions.relativeErrorGoal = 1e-1;
    integrationOptions.absoluteErrorGoal = 1e-8;
    integrationOptions.bOptimizeUltrarelativistic = true;
}

CollisionManager::CollisionManager(size_t basisSize) : CollisionManager()
{
    changePolynomialBasis(basisSize);
}

void CollisionManager::addParticle(const ParticleSpecies &particle)
{
    const std::string name = particle.getName();
    
    if (particleRegistered(particle))
    {
        std::cout << "Particle " << name << " already registered with CollisionManager, doing nothing\n";
        return;
    }

    // TODO let the user specify the index themselves?
    particles.push_back(std::make_shared<ParticleSpecies>(particle));

    particleIndex[name] = particles.size() - 1;

    if (!particle.isInEquilibrium())
    {
        outOfEqParticles.push_back(particles.back());
    }
}

void CollisionManager::clear()
{
    clearCollisionIntegrals();
    modelParameters.clear();
    particleIndex.clear();
    outOfEqParticles.clear();
    particles.clear();
}


void CollisionManager::updateParticleMasses(const std::map<std::string, double> &msqVacuum, const std::map<std::string, double> &msqThermal)
{
    // TODO proper validation. The .at() will crash if the key is not found 
    for (const auto &[name, msq] : msqVacuum)
    {
        const size_t idx = particleIndex.at(name);
        particles[idx]->setVacuumMassSquared(msq);
    }

    for (const auto &[name, msq] : msqThermal)
    {
        const size_t idx = particleIndex.at(name);
        particles[idx]->setThermalMassSquared(msq);
    }
}


void CollisionManager::changePolynomialBasis(size_t newBasisSize)
{
    basisSize = newBasisSize;
    for (auto &[key, integral] : integrals)
    {
        integral.changePolynomialBasis(newBasisSize);
    }
}

void CollisionManager::defineVariable(const std::string &name, double value)
{
    if (modelParameters.count(name) > 0)
    {
        std::cerr << "Error: variable " << name << " has already been defined\n";
    }
    else
    {
        modelParameters[name] = value;
    }
}

void CollisionManager::defineVariables(const std::map<std::string, double> &variables)
{
    for (const auto& [name, value] : variables)
    {
        defineVariable(name, value);
    }
}

void CollisionManager::setVariable(const std::string &name, double value)
{
    if (modelParameters.count(name) < 1)
    {
        std::cerr << "Error: can't change value of undefined variable " << name << "\n";
        return;
    }
    modelParameters[name] = value;

    // Sync collision elements    
    for (auto& [_, integral] : integrals)
    {
        integral.updateModelParameter(name, value);
    }
}

void CollisionManager::setVariables(const std::map<std::string, double> &newValues)
{
    for (const auto &[key, val] : newValues)
    {
        setVariable(key, val);
    }
}

CollisionIntegral4 CollisionManager::setupCollisionIntegral(const std::shared_ptr<ParticleSpecies>& particle1, const std::shared_ptr<ParticleSpecies>& particle2, 
    const std::string &inMatrixElementFile, size_t inBasisSize, bool bVerbose)
{
    CollisionIntegral4 collisionIntegral(inBasisSize);
    std::vector<CollElem<4>> collisionElements = parseMatrixElements(particle1->getName(), particle2->getName(), inMatrixElementFile, bVerbose);
    
    for (const CollElem<4> &elem : collisionElements)
    {
        collisionIntegral.addCollisionElement(elem);
    }

    return collisionIntegral;
}

void CollisionManager::setupCollisionIntegrals(bool bVerbose)
{
    clearCollisionIntegrals();

    for (const std::shared_ptr<ParticleSpecies>& particle1 : outOfEqParticles) 
    for (const std::shared_ptr<ParticleSpecies>& particle2 : outOfEqParticles)
    {
        const std::string name1 = particle1->getName();
        const std::string name2 = particle2->getName();

        const auto namePair = std::make_pair(name1, name2);

        CollisionIntegral4 newIntegral = setupCollisionIntegral(particle1, particle2, matrixElementFile.string(), basisSize, bVerbose);

        integrals.insert( {namePair, newIntegral} );
    }

}

void CollisionManager::clearCollisionIntegrals()
{
    integrals.clear();
}

CollisionTensor CollisionManager::evaluateCollisionTensor(CollisionIntegral4 &collisionIntegral, const IntegrationOptions &options, bool bVerbose)
{

    const size_t N = collisionIntegral.getPolynomialBasisSize();

    CollisionTensor result(collisionIntegral.getPolynomialBasisSize(), true);

    // Note symmetry: C[Tm(-rho_z), Tn(rho_par)] = (-1)^m C[Tm(rho_z), Tn(rho_par)]
	// which means we only need j <= N/2

    /* Each thread needs its collisionIntegral because the operations for computing the integrand are not thread safe!
    * Take copy here because using firstprivate(...) on a reference type */

    CollisionIntegral4 workIntegral(collisionIntegral.getPolynomialBasisSize());

	#pragma omp parallel private(workIntegral)
    {
        workIntegral = collisionIntegral;

        int numThreads = 1;
        int threadID = 0;
    #if WITH_OMP
        numThreads = omp_get_num_threads();
        threadID = omp_get_thread_num();
    #endif
        const int lastThreadID = numThreads - 1;

        // ---- Progress tracking

        // How many we've calculated inside this function only (so for this out-of-eq pair) 
        size_t localIntegralCount = 0;
        // Report when thread0 has computed this many integrals. NB: totalIntegralCount is the full count including all out-of-eq pairs
        size_t standardProgressInterval = totalIntegralCount / 25 / numThreads; // every 25%
        standardProgressInterval = wallgo::clamp<size_t>(standardProgressInterval, initialProgressInterval, totalIntegralCount); // but not more frequently than this

        size_t progressReportInterval = ( bFinishedInitialProgressCheck ? standardProgressInterval : initialProgressInterval );

        // VS OMP limitation: for loops must use signed integer indices, size_t apparently doesn't work

    #if WG_OMP_SUPPORTS_COLLAPSE
        #pragma omp for collapse(4)
    #else
        #pragma omp for
    #endif
        // m,n = Polynomial indices
        for (int64_t m = 2; m <= (int64_t)N; ++m)
        for (int64_t n = 1; n <= (int64_t)(N-1); ++n)
        {
            // j,k = grid momentum indices 
            for (int64_t j = 1; j <= (int64_t)(N/2); ++j)
            for (int64_t k = 1; k <= (int64_t)(N-1); ++k)
            {
            
                IntegrationResult localResult;

                // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
                if (2*j == (int64_t)N && m % 2 != 0)
                {
                    localResult.result = 0.0;
                    localResult.error = 0.0;
                }
                else
                {
                    localResult = workIntegral.integrate(m, n, j, k, options);
                }

                result.valueAt(m-2, n-1, j-1, k-1) = localResult.result;
                result.errorAt(m-2, n-1, j-1, k-1) = localResult.error;

                localIntegralCount++;

                if (bVerbose)
                {   
                    std::cout << "m=" << m << " n=" << n << " j=" << j << " k=" << k << " : "
                        << localResult.result << " +/- " << localResult.error << "\n";
                }

                // Report progress from a thread that is NOT the main thread, because that one tends to need less initialization and can be ahead
                if (threadID == lastThreadID && (localIntegralCount % progressReportInterval == 0)) 
                {
                    std::chrono::steady_clock::time_point currentTime = std::chrono::steady_clock::now();
                    elapsedTime = currentTime - startTime;

                    // HACK: could not figure out how to nicely sync the counts from all threads to correctly update computedIntegralCount. 
                    // Here I extrapolate from thread0 to estimate the progress, then undo the change afterwards. 
                    // Correct count is calculated at the end of this function
                    const size_t backupCount = computedIntegralCount;
                    computedIntegralCount += localIntegralCount * numThreads;
                    computedIntegralCount = wallgo::clamp<size_t>(computedIntegralCount, localIntegralCount, totalIntegralCount);

                    // TODO process tracking is not correct now
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
#if WG_OMP_SUPPORTS_COLLAPSE
	#pragma omp for collapse(4)
#else
    #pragma omp for
#endif
    for (int64_t m = 2; m <= (int64_t)N; ++m)
	for (int64_t n = 1; n <= (int64_t)(N-1); ++n)
    {
		for (int64_t j = N/2+1; j <= (int64_t)(N-1); ++j)
		for (int64_t k = 1; k <= (int64_t)(N-1); ++k)
        {

			const int64_t jOther = N - j;
			const int sign = (m % 2 == 0 ? 1 : -1);
            
			result.valueAt(m-2, n-1, j-1, k-1) = sign * result.valueAt(m-2, n-1, jOther-1, k-1);
			result.errorAt(m-2, n-1, j-1, k-1) = sign * result.errorAt(m-2, n-1, jOther-1, k-1);
		}
	}

    // How many we calculated in this function. 
    // Just recalculate this here instead of trying to combine counts from many threads and manually count j > N/2 cases etc
    computedIntegralCount += countIndependentIntegrals(N, 1);  

    return result;      
}

CollisionTensor CollisionManager::evaluateCollisionTensor(CollisionIntegral4 &collisionIntegral, bool bVerbose)
{
    return evaluateCollisionTensor(collisionIntegral, integrationOptions, bVerbose);
}

CollisionTensor CollisionManager::evaluateCollisionTensor(const std::string &particle1,
    const std::string &particle2, const IntegrationOptions &options, bool bVerbose)
{
    const auto namePair = std::make_pair(particle1, particle2);

    if (integrals.count(namePair) < 1)
    {
        std::cerr << "Error: no collisions defined for particle pair ["
            << namePair.first << ", " << namePair.second << "]\n!";
        return CollisionTensor(1, false);
    }

    return evaluateCollisionTensor(integrals.at(namePair), options, bVerbose);
}

CollisionTensor CollisionManager::evaluateCollisionTensor(const std::string &particle1,
    const std::string &particle2, bool bVerbose)
{
    return evaluateCollisionTensor(particle1, particle2, integrationOptions, bVerbose);
}

void CollisionManager::configureIntegration(const IntegrationOptions &options)
{
    integrationOptions = options;
}

void CollisionManager::setOutputDirectory(const std::string &directoryName)
{
    namespace fs = std::filesystem;

    // Create the directory if it doesn't exist
    fs::path dir(directoryName);
    if (!fs::exists(dir))
    {
        try
        {
            fs::create_directory(dir);
        }
        catch (const fs::filesystem_error& e)
        {
            std::cerr << "Failed to create collision output dir: " << dir.string() 
                << ". Error was: " << e.what() << std::endl;
            return;
        }
    }

    outputDirectory = dir;
}

bool CollisionManager::setMatrixElementFile(const std::string &filePath)
{
    matrixElementFile = std::filesystem::path(filePath);
    // Check that the file exists
    if (!std::filesystem::exists(matrixElementFile))
    {
        std::cerr << "Error: Can't find matrix element file " << matrixElementFile.string() << std::endl;
        return false;
    }
    return true;
}

void CollisionManager::calculateAllIntegrals(bool bVerbose)
{

    if (integrals.size() < 1)
    {
        std::cout << "Warning: calculateCollisionIntegrals() called, but no integrals have been initialized." 
            << "Please call setupCollisionIntegrals() first." << std::endl;
    }

    // Initialize progress tracking 
    totalIntegralCount = countIndependentIntegrals(basisSize, outOfEqParticles.size());
    computedIntegralCount = 0;
    bFinishedInitialProgressCheck = false;
    startTime = std::chrono::steady_clock::now();

    // make rank 2 tensor that mixes out-of-eq particles (each element is a collision integral, so actually rank 6, but the grid indices are irrelevant here)

    for (auto & [namePair, integral] : integrals)
    {
        std::chrono::steady_clock::time_point pairStartTime = std::chrono::steady_clock::now();

        const std::string name1 = namePair.first;
        const std::string name2 = namePair.second;

        CollisionTensor result = evaluateCollisionTensor(integral, integrationOptions, bVerbose);

        // Create a new HDF5 file. H5F_ACC_TRUNC means we overwrite the file if it exists
        const std::string fileNameBase = "collisions_" + name1 + "_" + name2 + ".hdf5";
        std::filesystem::path outputPath(outputDirectory);
        
        outputPath = outputPath / fileNameBase;

        H5::H5File h5File(outputPath.string(), H5F_ACC_TRUNC);

        H5Metadata metadata;
        metadata.basisSize = basisSize;
        metadata.basisName = "Chebyshev";
        metadata.integrator = "Vegas Monte Carlo (GSL)";

        writeMetadata(h5File, metadata);

// TODO FIXME
/*
        writeDataSet(h5File, result.results, name1 + ", " + name2);
        writeDataSet(h5File, result.errors, name1 + ", " + name2 + " errors");
*/      
        h5File.close();

        // How long did this all take
        std::chrono::duration<double> duration = std::chrono::steady_clock::now() - pairStartTime;
        double seconds = duration.count();
        int hours = static_cast<int>(seconds / 3600);
        // leftover mins
        int minutes = static_cast<int>(seconds - hours * 3600 / 60);
        std::cout << "[" << name1 << ", " << name2 << "] done in " << hours << "h " << minutes << "min." << std::endl;

    }

}


CollElem<4> CollisionManager::makeCollisionElement(const std::string &particleName2, const std::vector<size_t> &indices, 
    const std::string &expr, const std::vector<std::string> &symbols)
{
    assert(indices.size() == 4);

    const std::shared_ptr<ParticleSpecies> p1 = particles[indices[0]];
    const std::shared_ptr<ParticleSpecies> p2 = particles[indices[1]];
    const std::shared_ptr<ParticleSpecies> p3 = particles[indices[2]];
    const std::shared_ptr<ParticleSpecies> p4 = particles[indices[3]];
    
    CollElem<4> collisionElement( { p1, p2, p3, p4} );

    std::map<std::string, double> variables;
    for (const std::string& s : symbols)
    {
        if (modelParameters.count(s) < 1)
        {
            std::cerr << "CollisionManager error: symbol " << s << " not found in modelParameters map, can't parse matrix element " << expr << "\n";
            return collisionElement;
        }
        variables[s] = modelParameters.at(s);
    }

    collisionElement.matrixElement.initSymbols(variables);
    // Parse expr as a math function
    collisionElement.matrixElement.setExpression(expr);

    // Set deltaF flags: in general there can be 4 deltaF terms but we only take the ones with deltaF of particle2
    for (size_t i = 0; i < collisionElement.bDeltaF.size(); ++i) 
    {
        collisionElement.bDeltaF[i] = (collisionElement.particles[i]->getName() == particleName2);
    }
    
    return collisionElement;
}


std::vector<CollElem<4>> CollisionManager::parseMatrixElements(const std::string &particleName1, const std::string &particleName2,
                                                                 const std::string &inMatrixElementFile, bool bVerbose)
{
    // Just for logging 
    const std::string pairName = "[" + particleName1 + ", " + particleName2 + "]"; 

    // @todo should actually check if these are in outOfEquilibriumParticles vector
    if (particleIndex.count(particleName1) < 1 || particleIndex.count(particleName2) < 1 ) 
    {
        std::cerr << "Error: particles missing from list! Was looking for out-of-eq particles " << pairName << "\n";
        exit(9);
    }

    if (bVerbose) std::cout << "\n" <<"Parsing matrix elements for off-equilibrium pair " << pairName << "\n";
    
    std::ifstream file(inMatrixElementFile);

    // M_ab -> cd, with a = particle1 and at least one of bcd is particle2. Suppose that for each out-of-eq pair, Mathematica gives these in form 
    // M[a, b, c, d] -> (some symbolic expression), where abcd are integer indices that need to match our ordering in particleIndex map
    // Here we parse the lhs to extract indices, then parse the rhs as a math expression (function of s,t,u and other symbols that we infer).
    // For each of these we make a CollElem<4> object with correct deltaF structure which we infer from the indices 

    std::vector<CollElem<4>> collisionElements;

    if (!file.is_open()) {
        std::cerr << "!!! Error: Failed to open matrix element file " << inMatrixElementFile << std::endl;
        return collisionElements;
    }

    /* Now use regex to read all lines of form M[...] -> expr
    For each line we check if the first particle index matches that of particleName1
    and require that at least one other index matches that of particleName2.
    This is not optimal because we end up reading the full file for each off-eq pair.
    */
    std::string line;
    std::string expr;
    std::vector<size_t> indices;
    indices.resize(4);
    
    while (std::getline(file, line)) {
        if (std::regex_search(line, std::regex("M\\[.*\\] -> (.*)"))) {

            /* Big TODO. Change matrix element file format so that 
            1. It's easier to parse without regex hacks. Eg: JSON format
            2. Each matrix element could be associated with a list of symbols needed to evaluate it.
            This would make it possible to safely define just enough symbols needed for each matrix element.
            */
            
            interpretMatrixElement(line, indices, expr);
            
            if (indices[0] != particleIndex[particleName1]) continue;
            if ( std::find(indices.begin(), indices.end(), particleIndex[particleName2]) == indices.end() ) continue;
            
            /* Since we don't have per-expression symbol lists implemented in the files,
            here I just define each symbol in our modelParameters list as a variable. This wastes memory.
            Alternatively, could try to implicitly find free symbols with the parser, but this seems error prone. */ 

            std::vector<std::string> symbols;
            symbols.reserve(modelParameters.size());
            for (const auto& [key, _] : modelParameters)
            {
                symbols.push_back(key);
            }

            CollElem<4> newElem = makeCollisionElement(particleName2, indices, expr, symbols);
            collisionElements.push_back(newElem);

            if (bVerbose)
            {
                std::cout << "Loaded matrix element:\n";
                std::cout << line << "\n";
            }
        }
    }

    file.close();

    std::cout << "\n";

    return collisionElements;
}

size_t CollisionManager::countIndependentIntegrals(size_t basisSize, size_t outOfEqCount)
{
    const size_t N = basisSize;
    // How many independent integrals in each CollisionIntegral4
    size_t count = (N-1)*(N-1)*(N-1)*(N-1);
    // C[Tm(-x), Tn(y)] = (-1)^m C[Tm(x), Tn(y)]
    count = count / 2;
    // Take ceiling (but this avoids type casts)
    if (count % 2 != 0) count += 1;

    // Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
    if (N % 2 == 0) {
        // how many odd m?
        size_t mOdd = N / 2;
        count -= mOdd;
    } 

    // this was for 1 deltaF particle, for more than one we have mixing terms
    return count * outOfEqCount*outOfEqCount;
}


void CollisionManager::reportProgress()
{
    if (totalIntegralCount > 0)
    {
        double elapsedSeconds = elapsedTime.count();
        double timePerIntegral = elapsedSeconds / computedIntegralCount;
        double timeRemaining = timePerIntegral * (totalIntegralCount - computedIntegralCount);

        int percentage = int(100 * static_cast<double>(computedIntegralCount) / totalIntegralCount);
        std::cout << "Integral progress: " << computedIntegralCount << " / " << totalIntegralCount << " (" << percentage << "%). "; 
        std::cout << "Estimated time remaining: " << std::floor(timeRemaining / 3600) << "h " << (int(timeRemaining) % 3600 ) / 60 << "min" << std::endl;
    }
    bFinishedInitialProgressCheck = true;
}

bool CollisionManager::particleRegistered(const ParticleSpecies& particle)
{
    return particleIndex.count(particle.getName()) > 0;
}


} // namespace