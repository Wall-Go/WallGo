#include <fstream>
#include <sstream>
#include <regex>
#include <map>

#include "MatrixElementParsing.h"
#include "Common.h"

namespace wallgo
{
namespace utils
{

// Function for this file only. Processes string of form "M[a,b,c,d] -> some funct" and stores in the arguments
void interpretMatrixElement(const std::string& inputString, std::vector<uint32_t>& indices, std::string& mathExpression)
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
            indices.push_back(static_cast<uint32_t>(num));

            // Check for the ',' separator and ignore it
            if (ss.peek() == ',')
            {
                ss.ignore();
            }
        }
    }
}

bool parseMatrixElements(
    const std::filesystem::path& matrixElementFile,
    std::vector<uint32_t> offEqParticleIndices,
    const std::unordered_map<std::string, double>& symbols,
    std::map<IndexPair, std::vector<MatrixElement>> outMatrixElements)
{
    outMatrixElements.clear();

    std::ifstream file(matrixElementFile);
    if (!file.is_open()) {
        std::cerr << "Failed to open matrix element file: " << matrixElementFile << std::endl;
        return false;
    }

    // Use regex to read all lines of form M[a,b,c,d] -> expr

    /* Big TODO. Change matrix element file format so that
    1. It's easier to parse without regex hacks. Eg: JSON format
    2. Each matrix element could be associated with a list of symbols needed to evaluate it.
    This would make it possible to safely define just enough symbols needed for each matrix element.
    */

    std::string line;

    // temp arrays
    std::vector<std::string> readExpressions;
    std::vector<std::vector<uint32_t>> readIndices;

    while (std::getline(file, line))
    {
        if (std::regex_search(line, std::regex("M\\[.*\\] -> (.*)")))
        {
            std::string expr;
            std::vector<uint32_t> indices;
            interpretMatrixElement(line, indices, expr);

            if (indices.size() < 1 || expr.empty())
            {
                std::cerr << "Invalid matrix element: " << line << std::endl;
                return false;
            }

            readExpressions.push_back(expr);
            readIndices.push_back(indices);
        }
    }

    file.close();

    /* Now create the MatrixElement objects and group them by their external off-eq indices
    * so that we know which elements are needed for collisions of particle pair (a,b).
    */
    for (uint32_t idx1 : offEqParticleIndices) for (uint32_t idx2 : offEqParticleIndices)
    {
        const IndexPair offEqPair(idx1, idx2);
        outMatrixElements.insert({ offEqPair, std::vector<MatrixElement>() });

        for (uint32_t elementIdx = 0; elementIdx < readExpressions.size(); ++elementIdx)
        {
            const auto& indices = readIndices[elementIdx];
            if (indices[0] != idx1) continue;
            // Any other index needs to match idx2
            if (std::find(indices.begin(), indices.end(), idx2) == indices.end()) continue;

            MatrixElement newElement;
            newElement.init(readExpressions[elementIdx], indices, symbols);

            outMatrixElements.at(offEqPair).push_back(newElement);
        }
    }

    return true;
}

} // namespace utils
} // namespace wallgo
