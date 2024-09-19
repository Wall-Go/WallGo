#pragma once

#include <filesystem>
#include <map>

#include "Common.h"
#include "MatrixElement.h"

namespace wallgo
{

namespace utils
{

/* Reads matrix elements from file and groups them for each out - of - equilibrium particle pair.
Returns false if something goes wrong. */
bool parseMatrixElements(
    const std::filesystem::path& matrixElementFile,
    std::vector<uint32_t> offEqParticleIndices,
    const std::unordered_map<std::string, double>& symbols,
    std::map<IndexPair, std::vector<MatrixElement>>& outMatrixElements);


} // namespace utils
} // namespace wallgo
