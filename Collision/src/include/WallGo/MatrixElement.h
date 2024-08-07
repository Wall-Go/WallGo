#pragma once

#include "Common.h"
#include "ModelParameters.h"

#include "muParser.h" // math expression parser

#include <unordered_map>
#include <vector>
#include <string>
#include <type_traits>


namespace wallgo
{

template<typename T>
struct WALLGO_API TMandelstam
{
    static_assert(std::is_floating_point<T>::value, "T must be floating point type (in TMandelstam)");
    TMandelstam() {}
    TMandelstam(T sIn, T tIn, T uIn) : s(sIn), t(tIn), u(uIn) {}
    T s, t, u;
};

using Mandelstam = TMandelstam<double>;


/* Class MatrixElement - Describes a single symbolic matrix element (abs value squared). Uses muparser as the math parsing backend.
Logic here is that the default constructor just constructs a dummy element with expression equal to 0
and defines the Mandelstam variables s,t,u as symbols whose numerical values are tied to our private variables. 
Calling initSymbols(parameters) defines a symbol for each string key in the 'parameters' map and fills an internal parameters map
for setting numerical values for these symbols. Note that it is currently NOT possible to define a symbol without specifying also an initial value.
Here 'parameters' refers to model-dependent variables like couplings, masses etc.
*/
class MatrixElement
{

public:

    MatrixElement();
    ~MatrixElement();

    // Need custom copy and assignment operators to properly handle parser symbol bindings
    MatrixElement(const MatrixElement& other);
    MatrixElement& operator=(const MatrixElement& other);

    /* Initializes the matrix element with given math expression.
    * 'symbols' must contain all non-constant symbols that appear in the expression
    * and their initial values, however the letters s, t, u are reserved for internal use.
    * Extra symbols that do not actually appear in the expression are OK but will waste memory.
    * NB: This will clear any existing state from the MatrixElement object.
    * Return value is false if something goes wrong in either symbol definitions or expression test evaluation,
    * and will leave the matrix element in an invalid state. */
    bool init(
        const std::string& expression,
        const std::vector<uint32_t> externalParticleIndices,
        const std::unordered_map<std::string, double>& symbols);
    
    /* Sets a new math expression, requires re-parse.
    Return value is false if the expression cannot be evaluated. NB: failure will put this matrix element in an invalid state. */
    bool setExpression(const std::string& expressionIn);

    std::string getExpression() const { return mExpression; }
    std::vector<uint32_t> getParticleIndices() const { return mParticleIndices; }

    // Must be called to update values of model-specific symbols in matrix elements
    void updateModelParameters(const ModelParameters& parameters);
    void updateModelParameter(const std::string& name, double newValue);

    /* Evaluate the matrix element at s, t, u, using cached couplings and masses.
    NB: No easy way of making this const because evaluating the parsed expression requires we set internal variables. */
    double evaluate(const Mandelstam& mandelstams);

private:

    mu::Parser parser;

    // Binds Mandelstam variables to the parser
    Mandelstam mMandelstam;
    
    /* We mirror model parameters and bind the parser to these instead of binding directly to parameter structs in a model class.
    This is safer and not a performance issue as long as the parameters need not be updated too frequently.*/
    std::unordered_map<std::string, double> mSymbols;

    // Metadata about what external particles this matrix element describes
    std::vector<uint32_t> mParticleIndices;

    std::string mExpression;

    // Tests that our expression is valid and can be evaluated by the parser
    bool testExpression();

    bool defineSymbol(const std::string& symbol, double initValue);
    void initParser();
    void clearParser();
};

} // namespace
