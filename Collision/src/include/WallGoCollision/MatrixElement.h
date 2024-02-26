#ifndef MATRIXELEMENTPARSER_H
#define MATRIXELEMENTPARSER_H

#include <map>
#include <vector>
#include <string>

#include "muParser.h" // math expression parser

namespace wallgo
{

class MatrixElement {

public:

    MatrixElement();
    ~MatrixElement();

    // Gonna need deep copying constructors/assignments because of the parser
    MatrixElement(const MatrixElement& other);
    void operator=(const MatrixElement& another);
    
    void setExpression(const std::string& expressionIn);

    std::string getExpression() const { return expression; }

    /* Initialize the math expression parser. This defines symbolic variables s,t,u ; couplings[i] and msq[i]. 
    Needs to be called before attempting evaluation */
    void initParser(const std::vector<double>& couplings, const std::vector<double>& massSquares);

    // This fixes couplings and masses, caching them internally
    void setConstants(const std::vector<double>& couplings, const std::vector<double>& massSquares);

    // Evaluate the matrix element at s,t,u, using cached couplings and masses
    double evaluate(double s, double t, double u);


    /* Math expression parser. Might as well make it a pointer - had issues with this breaking when passing MatrixElements around with default constructors */
    mu::Parser *parser = nullptr;
    std::string expression;

private:

    double s_internal, t_internal, u_internal;

    std::vector<double> couplings_internal;
    std::vector<double> msq_internal;

    // Tests that our expression is valid and can be evaluated by the parser
    void testExpression();
};

} // namespace

#endif // header guard