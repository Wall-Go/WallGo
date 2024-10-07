#pragma once

#include <filesystem>
#include <map>

#include "Common.h"
#include "MatrixElement.h"

namespace wallgo
{

namespace utils
{

struct ReadParticle
{
    std::string name;
    int32_t index;
};

struct ReadMatrixElement
{
    std::vector<int32_t> particleIndices;
    std::vector<std::string> parameters;
    std::string expression;
};

/* Reads matrix elements from file and groups them for each out-of-equilibrium particle pair.
The particleIndices input specifies which particles are valid external particles.
The function ignores processes that contain unknown particles, but warns about them with a message to stdout.
The offEqParticleIndices inputs specifies which particles are out-of-equilibrium.
Returns false if something goes wrong. */
bool buildMatrixElementsFromFile(
    const std::filesystem::path& matrixElementFile,
    const std::vector<int32_t>& particleIndices,
    const std::vector<int32_t>& offEqParticleIndices,
    const std::unordered_map<std::string, double>& symbols,
    std::map<IndexPair, std::vector<MatrixElement>>& outMatrixElements);


bool parseMatrixElementsJson(
    const std::filesystem::path& matrixElementFile,
    std::vector<ReadParticle>& outParticles,
    std::vector<ReadMatrixElement>& outMatrixElements);

/* Legacy parsing of.txt based matrix elements using regex.Prefer.json format whenever possible.
* Note that this does NOT parse particle or parameter info, just external particle indices and the associated expression.
*/
bool parseMatrixElementsRegexLegacy(
    const std::filesystem::path& matrixElementFile,
    std::vector<ReadMatrixElement>& outMatrixElements);

/* Construct valid MatrixElement objects by matching read data with properties defined in a model.
Returns false if something goes critically wrong. */
bool buildMatrixElements(
    const std::vector<int32_t>& modelParticleIndices,
    const std::vector<int32_t>& modelOffEqParticleIndices,
    const std::unordered_map<std::string, double>& modelSymbols,
    const std::vector<ReadParticle>& parsedParticles,
    const std::vector<ReadMatrixElement>& parsedMatrixElements,
    std::map<IndexPair, std::vector<MatrixElement>>& outMatrixElements);

} // namespace utils
} // namespace wallgo
