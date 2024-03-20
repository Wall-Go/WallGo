#ifndef MATRIXELEMENTPARSER_H
#define MATRIXELEMENTPARSER_H

#include <map>
#include <vector>
#include <string>

#include "muParser.h" // math expression parser

namespace wallgo
{

/* Class MatrixElement - Describes a single symbolic matrix element (squared). Uses muparser for handling symbolic expressions.
Logic here is that the default constructor just constructs a dummy element with expression equal to 0, and does not define any symbols. 
Calling initParser(parameters) defines the Mandelstam (s,t,u) as free symbols that are set later during integration, 
and also defines a symbol for each key in the 'parameters' std::map.
The parameters map should contain a definition for each symbol that appears in the matrix element expression.
Calling setParameters(parameters) fixes numerical values for all these symbols. Currently this setting happens automatically in initParser. */
class MatrixElement {

public:

    MatrixElement();
    ~MatrixElement();

    // Gonna need deep copying constructors/assignments because of the parser
    MatrixElement(const MatrixElement& other);
    void operator=(const MatrixElement& another);
    
    void setExpression(const std::string& expressionIn);

    std::string getExpression() const { return expression; }

    /* Initialize the math expression parser. This defines symbolic variables (s,t,u) and a symbol for each string in 'parameters'.
    Needs to be called before attempting evaluation */
    void initParser(const std::map<std::string, double>& parameters);

    // This gives numerical values to all our internal symbols except for (s,t,u), as defined in initParser.
    void setParameters(const std::map<std::string, double>& parameters);

    // Evaluate the matrix element at s,t,u, using cached couplings and masses
    double evaluate(double s, double t, double u);

    void defineSymbol(const std::string& symbol);


    /* Math expression parser. Might as well make it a pointer - had issues with this breaking when passing MatrixElements around with default constructors */
    mu::Parser *parser = nullptr;
    std::string expression;

private:

    double s_internal, t_internal, u_internal;

    // Internal map of model-specific symbols that are not (s,t,u), and their current values
    std::map<std::string, double> parametersInternal;

    // Tests that our expression is valid and can be evaluated by the parser
    void testExpression();
};

} // namespace

#endif // header guard