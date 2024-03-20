#ifndef MATRIXELEMENTPARSER_H
#define MATRIXELEMENTPARSER_H

#include <map>
#include <vector>
#include <string>

#include "muParser.h" // math expression parser

namespace wallgo
{

/* Class MatrixElement - Describes a single symbolic matrix element (abs value squared). Uses muparser as the math parsing backend.
Logic here is that the default constructor just constructs a dummy element with expression equal to 0, 
and defines the Mandelstam variables s,t,u as symbols whose numerical values are tied to our private variables. 
Calling initSymbols(parameters) defines a symbol for each string key in the 'parameters' map and fills an internal parameters map
for setting numerical values for these symbols. Note that it is currently NOT possible to define a symbol without specifying also an initial value.
Here 'parameters' refers to model-dependent variables like couplings, masses etc.

Therefore, once you have a math expression in string format to describe a matrix element, usage should be: 
    1. Call initSymbols(parameters) with a std::map that has all symbols (apart from s,t,u) needed to evaluate the wanted expression, with some initial values.
    Does not matter if it contains "extra" symbols that the expression does not actually depend on, but this wastes memory.
    2. Call setExpression(expr) with your string expression. This does the parsing and tests that the expression can be computed numerically.
    3. Evaluate with evaluate(s, t, u) with Mandelstam variables of your choice.
    4. Can change parameter values with the setParameters function without having to redefine or re-parse anything.
*/
class MatrixElement {

public:

    MatrixElement();
    ~MatrixElement();

    // Need deep copying constructors/assignments because of the parser
    MatrixElement(const MatrixElement& other);
    void operator=(const MatrixElement& another);
    
    void setExpression(const std::string& expressionIn);

    std::string getExpression() const { return expression; }

    /* Initialize symbols for math expression parser. This defines a symbol for each string in 'parameters'
    and associates them with variables in parametersInternal. Needs to be called before actual parsing. */
    void initSymbols(const std::map<std::string, double>& parameters);

    // This gives numerical values to all our internal symbols except for (s,t,u), as defined in initParser.
    void setParameters(const std::map<std::string, double>& parameters);

    // Evaluate the matrix element at s,t,u, using cached couplings and masses
    double evaluate(double s, double t, double u);

    /* Math expression parser. Might as well make it a pointer - had issues with this breaking when passing MatrixElements around with default constructors */
    mu::Parser *parser = nullptr;
    std::string expression;

private:

    double s_internal, t_internal, u_internal;

    // Internal map of model-specific symbols that are not (s,t,u), and their current values
    std::map<std::string, double> parametersInternal;

    // Tests that our expression is valid and can be evaluated by the parser
    void testExpression();

    void defineSymbol(const std::string& symbol, double initValue);
};

} // namespace

#endif // header guard