#include "Collision.h"

#include <array>

#include <cstdio> // for sprintf, probs remove later
#include <algorithm> // std::remove_if

Collision::Collision(uint basisSize) : basisSizeN(basisSize)
{
}

void Collision::addParticle(const ParticleSpecies &particle)
{
    particles.push_back(particle);

    massSquares.push_back(particle.getThermalMassSquared() + particle.getVacuumMassSquared());

    if (bMatrixElementsDone) {
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

    // Note symmetry: C[Tm(-rho_z), Tn(rho_par)] = (-1)^m C[Tm(rho_z), Tn(rho_par)]
	// which means we only need j <= N/2

	// m,n = Polynomial indices
	#pragma omp parallel for collapse(4) firstprivate(collisionIntegral)
	for (uint m = 2; m <= N; ++m) 
	for (uint n = 1; n <= N-1; ++n) {
		// j,k = grid momentum indices 
		for (uint j = 1; j <= N/2; ++j)
		for (uint k = 1; k <= N-1; ++k) {

		// Monte Carlo result for the integral + its error
		std::array<double, 2> resultMC;

			// Integral vanishes if rho_z = 0 and m = odd. rho_z = 0 means j = N/2 which is possible only for even N
			if (2*j == N && m % 2 != 0) {
				resultMC[0] = 0.0;
				resultMC[1] = 0.0;
			} else {
				resultMC = collisionIntegral.evaluate(m, n, j, k);
			}

			results[m-2][n-1][j-1][k-1] = resultMC[0];
			errors[m-2][n-1][j-1][k-1] = resultMC[1];

		    printf("m=%d n=%d j=%d k=%d : %g +/- %g\n", m, n, j, k, resultMC[0], resultMC[1]);

		} // end j,k
	} // end m,n

	// Fill in the j > N/2 elements
	#pragma omp parallel for collapse(4)
	for (uint m = 2; m <= N; ++m) 
	for (uint n = 1; n <= N-1; ++n) {
		for (uint j = N/2+1; j <= N-1; ++j)
		for (uint k = 1; k <= N-1; ++k) {
			uint jOther = N - j;
			uint sign = (m % 2 == 0 ? 1 : -1);
			results[m-2][n-1][j-1][k-1] = sign * results[m-2][n-1][jOther-1][k-1];
			errors[m-2][n-1][j-1][k-1] = sign * errors[m-2][n-1][jOther-1][k-1];
		}
	}
	

}


void Collision::calculateCollisionIntegrals()
{
    // Particle list is assumed to be fixed from now on!
    findOutOfEquilibriumParticles();
    makeParticleIndexMap();

    // make rank 2 tensor that mixes out-of-eq particles (each element is a collision integral, so actually rank 6, but the grid indices are irrelevant here)
    for (const ParticleSpecies& particle1 : outOfEqParticles) 
    {
        for (const ParticleSpecies& particle2 : outOfEqParticles)
        {
            // integration results/errors on the grid
            Array4D results;
            Array4D errors;
            
            CollisionIntegral4 collisionIntegral(basisSizeN);
            collisionIntegral.collisionElements = makeCollisionElements(particle1.getName(), particle2.getName());

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
        }
    }
    
}

std::vector<CollElem<4>> Collision::makeCollisionElements(const std::string &particleName1, const std::string &particleName2)
{
    // Just for logging 
    std::string pairName = "[" + particleName1 + ", " + particleName2 + " ]"; 

    // @todo should actually check if these are in outOfEquilibriumParticles vector
    if (particleIndex.count(particleName1) < 1 || particleIndex.count(particleName2) < 1 ) 
    {
        std::cerr << "Error: particles missing from list! Was looking for out-of-eq particles " << pairName << "\n";
        return std::vector<CollElem<4>>();
    }

    std::cout << "Parsing matrix elements for out-of-equilibrium pair " << pairName << "\n";
    
    std::vector<CollElem<4>> collisionElements;

    // M_ab -> cd, with a = particle1 and at least one of bcd is particle2. Suppose that for each out-of-eq pair, Mathematica gives these in form 
    // M[a, b, c, d] -> (some symbolic expression), where abcd are integer indices that need to match our ordering in particleIndex map
    // Here we parse the lhs to extract indices, then parse the rhs as a math expression (function of s,t,u and couplings/masses). 
    // For each of these we make a CollElem<4> object with correct deltaF structure which we infer from the indices 

    // readCollisions(particleName1, particleName2);
    // @todo these would now be read from file. But since we don't have exported files yet, just hardcode some example expressions  

    // hardcoding t,t since we have benchmarks for that
    if (particleName1 == "top" && particleName2 == "top")
    {
        std::string expr;
        char buf[1024];
        // there is tt -> tt but Benoit didn't have that... so skipping for now

        {
            // tq -> tq would look like this
            sprintf(buf, "M[%d,%d,%d,%d] -> 5 * 16./3. * couplings[0]^4 * (s^2 + u^2) / (msq[1] - t)^2", 
                            particleIndex[particleName1], particleIndex["quark"], particleIndex[particleName1], particleIndex["quark"]);
            expr = std::string(buf);

            collisionElements.push_back( makeCollisionElement(particleName1, particleName2, expr) );
        }
        {
            // tg -> tg would look like this
            sprintf(buf, "M[%d,%d,%d,%d] -> 16./9. * couplings[0]^4 * (9 * (s^2 + t^2) / (msq[1] - t)^2 - 4*s*u / (msq[2] - u)^2)", 
                            particleIndex[particleName1], particleIndex["gluon"], particleIndex[particleName1], particleIndex["gluon"]);
            expr = std::string(buf);

            collisionElements.push_back( makeCollisionElement(particleName1, particleName2, expr) );
        }
        {
            // tt -> gg would look like this
            sprintf(buf, "M[%d,%d,%d,%d] -> 32./9. * couplings[0]^4 * t * u * ( 1 / (msq[2] - t)^2 - 1 / (msq[2] - u)^2)", 
                            particleIndex[particleName1], particleIndex[particleName1], particleIndex["gluon"], particleIndex["gluon"]);
            expr = std::string(buf);

            collisionElements.push_back( makeCollisionElement(particleName1, particleName2, expr) );
        }
        
    }


    // then doing t,g for illustration. After symmetrization it only has tg -> tg, tt -> gg
    if (particleName1 == "top" && particleName2 == "gluon")
    {
        std::string expr;
        char buf[1024];
        { 
            // tg -> tg would look like this
            sprintf(buf, "M[%d,%d,%d,%d] -> 16./9. * couplings[0]^4 * (9 * (s^2 + t^2) / (msq[1] - t)^2 - 4 *s*u / (msq[0] - u)^2)", 
                            particleIndex[particleName1], particleIndex[particleName2], particleIndex[particleName1], particleIndex[particleName2]);
            expr = std::string(buf);

            collisionElements.push_back( makeCollisionElement(particleName1, particleName2, expr) );
        }
        { 
            // And tt -> gg would look like this
            sprintf(buf, "M[%d,%d,%d,%d] -> 32./9. * couplings[0]^4 * t*u * (1 / (msq[0] - t)^2 + 1 / (msq[0] - u)^2 )", 
                            particleIndex[particleName1], particleIndex[particleName1], particleIndex[particleName2], particleIndex[particleName2]);
            expr = std::string(buf);

            collisionElements.push_back( makeCollisionElement(particleName1, particleName2, expr) );
        }
    }

    bMatrixElementsDone = true;

    return collisionElements;
}

CollElem<4> Collision::makeCollisionElement(const std::string &particleName1, const std::string &particleName2, const std::string &readMatrixElement)
{
    // Assume the read matrix element is of form "M[a,b,c,d] -> some funct"
    // Split this so that we get the indices, and the RHS goes to rhsString
    std::vector<uint> indices(4);
    std::string rhsString;

    extractSymbolicMatrixElement(readMatrixElement, indices, rhsString);

    if (indices[0] != particleIndex[particleName1])
    {
        std::cerr << "Warning: first index in matrix element does not match name [" << particleName1 << "]\n";
    }

    const ParticleSpecies p1 = particles[indices[0]];
    const ParticleSpecies p2 = particles[indices[1]];
    const ParticleSpecies p3 = particles[indices[2]];
    const ParticleSpecies p4 = particles[indices[3]];
    
    CollElem<4> collisionElement( { p1, p2, p3, p4} );

    collisionElement.matrixElement.initParser(couplings, massSquares);
    // Parses the RHS math expression so that we can evaluate it as a function of the symbols
    collisionElement.matrixElement.setExpression(rhsString);
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

// Processes string of form "M[a,b,c,d] -> some funct" and stores in the arguments
void Collision::extractSymbolicMatrixElement(const std::string &inputString, std::vector<uint> &indices, std::string &mathExpression)
{   
    // First split the string by "->""
    std::vector<std::string> tokens(2);
    
    std::string delimiter = "->";
    std::string lhs = inputString.substr(0, inputString.find(delimiter));

    // RHS
    mathExpression = inputString.substr(lhs.length() + delimiter.length());

    // remove whitespaces from lhs to avoid weirdness
    lhs.erase(std::remove_if(lhs.begin(), lhs.end(), isspace), lhs.end());

    // Do annoying stuff to extract the abcd indices
    std::size_t start = lhs.find('[');
    std::size_t end = lhs.find(']');

    // Ensure '[' and ']' are found and the start position is before the end position
    if (start != std::string::npos && end != std::string::npos && start < end) {
        std::string values = lhs.substr(start + 1, end - start - 1);

        // Use stringstream to tokenize and extract integers
        std::istringstream ss(values);
        indices.clear();
        indices.reserve(4);
        int num;

        while (ss >> num) {
            indices.push_back(num);

            // Check for the ',' separator and ignore it
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
    }

}

