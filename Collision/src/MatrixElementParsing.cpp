#include <fstream>
#include <sstream>
#include <regex>
#include <map>
#include <algorithm>

#include "nlohmann/json.hpp"

#include "MatrixElementParsing.h"
#include "Common.h"

using json = nlohmann::json;

namespace wallgo
{
namespace utils
{

// Naive check based on file extension if it's likely a .json file
bool isLikelyJsonFile(const std::filesystem::path& filePath)
{
    std::string extension = filePath.extension().string();
    // make all lowercase
    std::transform(extension.begin(), extension.end(), extension.begin(), ::tolower);

    return extension == ".json";
}

bool buildMatrixElementsFromFile(
    const std::filesystem::path& matrixElementFile,
    const std::vector<int32_t>& particleIndices,
    const std::vector<int32_t>& offEqParticleIndices,
    const std::unordered_map<std::string, double>& symbols,
    std::map<IndexPair, std::vector<MatrixElement>>& outMatrixElements)
{
    if (offEqParticleIndices.size() < 1) return false;

    outMatrixElements.clear();

    std::vector<ReadParticle> parsedParticles;
    std::vector<ReadMatrixElement> parsedMatrixElements;

    bool bReadSuccess = false;

    if (isLikelyJsonFile(matrixElementFile))
    {
        bReadSuccess = parseMatrixElementsJson(
            matrixElementFile,
            parsedParticles,
            parsedMatrixElements);
    }
    else
    {
        std::cout << "Warning: using legacy matrix element parsing. Consider using .json file format for better data validation.\n";

        bReadSuccess = parseMatrixElementsRegexLegacy(
            matrixElementFile,
            parsedMatrixElements
        );
    }

    if (!bReadSuccess)
    {
        return false;
    }

    bool bBuildSuccess = buildMatrixElements(
        particleIndices,
        offEqParticleIndices,
        symbols,
        parsedParticles,
        parsedMatrixElements,
        outMatrixElements
    );

    return bBuildSuccess;
}

bool readParticlesJson(const json& data, std::vector<ReadParticle>& outParticles)
{
    bool bSuccess = true;
    std::vector<std::string> errorReasons;
    outParticles.clear();

    if (data.contains("particles") && data["particles"].is_array())
    {
        for (const auto& particleEntry : data["particles"])
        {

            bool bHasName = particleEntry.contains("name") && particleEntry["name"].is_string();
            bool bHasIndex = particleEntry.contains("index") && particleEntry["index"].is_number_integer();
            
            if (!bHasName)
            {
                errorReasons.push_back("Invalid name (missing or incomprehensible)");
            }

            if (!bHasIndex)
            {
                errorReasons.push_back("Invalid index (missing or incomprehensible)");
            }

            if (!bHasName || !bHasIndex)
            {
                std::cerr << "Invalid particle entry: " << particleEntry.dump() << std::endl;
                bSuccess = false;
                break;
            }

            ReadParticle particle;
            particle.name = particleEntry["name"];
            particle.index = particleEntry["index"];

            outParticles.push_back(particle);
        }
    }
    else
    {
        errorReasons.push_back("Missing or invalid entry: \"particles\"");
        bSuccess = false;
    }

    if (!bSuccess)
    {
        std::cerr << "\nParticle parsing from JSON failed. Reasons:\n";
        for (const std::string& errorMsg : errorReasons)
        {
            std::cerr << errorMsg << std::endl;
        }
    }

    return bSuccess;
}

bool readExpressionsJson(const json& data, std::vector<ReadMatrixElement>& outMatrixElements)
{
    std::vector<std::string> errorReasons;
    outMatrixElements.clear();

    if (data.contains("matrixElements") && data["matrixElements"].is_array())
    {
        for (const auto& entry : data["matrixElements"])
        {
            bool bHasIndices = entry.contains("externalParticles") && entry["externalParticles"].is_array();
            // lack of parameters is non-fatal
            bool bHasParameters = entry.contains("parameters") && entry["parameters"].is_array();
            bool bHasExpression = entry.contains("expression") && entry["expression"].is_string();

            ReadMatrixElement newElement;

            if (!bHasIndices)
            {
                errorReasons.push_back("Invalid particle indices (missing or incomprehensible)");            }
            else
            {
                for (const auto& idx : entry["externalParticles"])
                {
                    if (!idx.is_number_integer())
                    {
                        errorReasons.push_back("Non-integer particle index");
                        newElement.particleIndices.clear();
                        break;
                    }

                    newElement.particleIndices.push_back(idx);
                }
            }

            if (!bHasExpression)
            {
                errorReasons.push_back("Invalid matrix element expression (missing or incomprehensible");
            }

            if (errorReasons.size() > 0)
            {
                std::cerr << "Invalid matrix element entry: " << entry.dump() << std::endl;
                break;
            }

            newElement.expression = entry["expression"];

            // Read parameter symbols associated with the expression
            if (bHasParameters)
            {
                for (const auto& param : entry["parameters"])
                {
                    if (!param.is_string())
                    {
                        errorReasons.push_back("Invalid model parameter (must be string)");
                        newElement.parameters.clear();
                        break;
                    }

                    newElement.parameters.push_back(param);
                }
            }

            outMatrixElements.push_back(newElement);
        }
    }
    else
    {
        errorReasons.push_back("Missing or invalid entry: \"matrixElements\"");
    }

    bool bSuccess = errorReasons.size() == 0;
    if (!bSuccess)
    {
        std::cerr << "\nMatrix element parsing from JSON failed. Reasons:\n";
        for (const std::string& errorMsg : errorReasons)
        {
            std::cerr << errorMsg << std::endl;
        }
    }

    return bSuccess;
}

bool parseMatrixElementsJson(
    const std::filesystem::path& matrixElementFile,
    std::vector<ReadParticle>& outParticles,
    std::vector<ReadMatrixElement>& outMatrixElements)
{
    std::ifstream file(matrixElementFile);
    
    if (!file.is_open())
    {
        std::cerr << "Failed to open matrix element file: " << matrixElementFile << std::endl;
        return false;
    }

    json data = json::parse(file);
    file.close();

    bool bSuccess = readParticlesJson(data, outParticles);
    if (!bSuccess) return false;

    bSuccess &= readExpressionsJson(data, outMatrixElements);
    if (!bSuccess) return false;

    return bSuccess;
}


// Legacy function for this file only. Processes string of form "M[a,b,c,d] -> some funct" and stores in the arguments
void interpretMatrixElement(const std::string& inputString, std::vector<int32_t>& indices, std::string& mathExpression)
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
            indices.push_back(static_cast<int32_t>(num));

            // Check for the ',' separator and ignore it
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
    }
}


bool parseMatrixElementsRegexLegacy(const std::filesystem::path& matrixElementFile, std::vector<ReadMatrixElement>& outMatrixElements)
{
    outMatrixElements.clear();

    std::ifstream file(matrixElementFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open matrix element file: " << matrixElementFile << std::endl;
        return false;
    }

    // Use regex to read all lines of form M[a,b,c,d] -> expr
    std::string line;

    while (std::getline(file, line))
    {
        if (std::regex_search(line, std::regex("M\\[.*\\] -> (.*)")))
        {
            std::string expr;
            std::vector<int32_t> indices;
            interpretMatrixElement(line, indices, expr);

            if (indices.size() < 1 || expr.empty())
            {
                std::cerr << "Invalid matrix element: " << line << std::endl;
                return false;
            }

            ReadMatrixElement newElement;
            newElement.expression = expr;
            newElement.particleIndices = indices;

            outMatrixElements.push_back(newElement);
        }
    }

    return true;
}

bool buildMatrixElements(
    const std::vector<int32_t>& modelParticleIndices,
    const std::vector<int32_t>& modelOffEqParticleIndices,
    const std::unordered_map<std::string, double>& modelSymbols,
    const std::vector<ReadParticle>& parsedParticles,
    const std::vector<ReadMatrixElement>& parsedMatrixElements,
    std::map<IndexPair, std::vector<MatrixElement>>& outMatrixElements)
{
    outMatrixElements.clear();

    // parsedParticles currently not used as the legacy parsing does not support it.
    // It could be used for model <-> parsed expr consistency validations, but this is TODO
    WG_UNUSED(parsedParticles);

    // Iterate over all (p1, p2) off-eq particle combinations and find and init matrix elements that contribute to their Boltzmann mixing
    for (int32_t idx1 : modelOffEqParticleIndices) for (int32_t idx2 : modelOffEqParticleIndices)
    {
        const IndexPair offEqPair(idx1, idx2);
        outMatrixElements.insert({ offEqPair, std::vector<MatrixElement>() });

        for (const ReadMatrixElement& readElement : parsedMatrixElements)
        {
            assert(!readElement.particleIndices.empty() && "Matrix element missing external particle info");

            const std::vector<int32_t>& indices = readElement.particleIndices;

            if (indices.front() != idx1) continue;
            // Any other index needs to match idx2
            if (std::find(indices.begin(), indices.end(), idx2) == indices.end()) continue;

            // Off-eq indices found, next verify that all other external particles are also allowed (as specified by the input index list).
            bool bExternalsOK = true;
            for (int32_t externalIdx : indices)
            {
                if (std::find(modelParticleIndices.begin(), modelParticleIndices.end(), externalIdx) == modelParticleIndices.end())
                {
                    std::cout << "Warning: found process with external particles [";
                    for (size_t i = 0; i < indices.size(); ++i)
                    {
                        if (i > 0) std::cout << ", ";
                        std::cout << indices[i];
                    }
                    std::cout << "], but particle with index " << externalIdx << " has not been defined. This process will be ignored.\n";
                    bExternalsOK = false;
                    break;
                }
            }

            if (!bExternalsOK) continue;

            // All external particles OK, so the matrix element contributes. Now create it
            MatrixElement newElement;

            // Define the numerical matrix element to only depend on symbols that were parsed together with the expression.
            // If empty, define it to depend on all model parameters
            std::unordered_map<std::string, double> symbols;

            for (const std::string& s : readElement.parameters)
            {
                if (modelSymbols.count(s) < 1)
                {
                    std::cerr << "Error: matrix element was defined to depend on symbol '" << s << "', but this symbol has not been defined in PhysicsModel.\n";
                    std::cerr << "This is very likely caused by invalid user input and almost certainly fatal, but we try to continue anyway.\n";
                    std::cerr << "The problematic matrix element was:\n" << readElement.expression << std::endl;
                    continue;
                }

                symbols.insert({ s, modelSymbols.at(s) });
            }
            if (symbols.empty() or true)
            {
                symbols = modelSymbols;
            }

            if (!newElement.init(readElement.expression, readElement.particleIndices, symbols))
            {
                // Failed to make the matrix element evaluatable. This is fatal, so give up here and let the caller decide what to do
                return false;
            }

            outMatrixElements.at(offEqPair).push_back(newElement);
        }
    }

    return true;
}

} // namespace utils
} // namespace wallgo
