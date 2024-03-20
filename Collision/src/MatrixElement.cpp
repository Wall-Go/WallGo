#include <iostream>
#include <functional>
#include <array>

#include "muParser.h" // math expression parser
#include "MatrixElement.h"
#include "ParticleSpecies.h"
#include "CollElem.h"
#include "CollisionIntegral.h"

namespace wallgo
{

MatrixElement::MatrixElement() 
{
    parser = new mu::Parser();
    s_internal = 0; t_internal = 0; u_internal = 0;

    parser->SetExpr("0");

    parser->DefineVar("s", &s_internal);
    parser->DefineVar("t", &t_internal);
    parser->DefineVar("u", &u_internal);

    // To allow variable names like msq[2] we need to add [] to parser's character list
    parser->DefineNameChars("0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]");
}

MatrixElement::~MatrixElement()
{
    delete parser;
}


MatrixElement::MatrixElement(const MatrixElement& other)
{
    parametersInternal = other.parametersInternal;
    parametersInternal = other.parametersInternal;
    expression = other.getExpression();
    parser = new mu::Parser();
    initParser(couplings_internal, msq_internal);
    parser->SetExpr(expression);
}

void MatrixElement::operator=(const MatrixElement &other)
{
    couplings_internal = other.couplings_internal;
    msq_internal = other.msq_internal;
    expression = other.getExpression();
    // deleting this just to make sure we clear everything
    delete parser;
    parser = new mu::Parser();
    initParser(couplings_internal, msq_internal);
    parser->SetExpr(expression);
}

void MatrixElement::initParser(const std::vector<double> &couplings, const std::vector<double> &massSquares)
{
    setConstants(couplings, massSquares);

    
}

void MatrixElement::setExpression(const std::string &expressionIn)
{
    expression = expressionIn;
    parser->SetExpr(expression);

    /* Do checks here so that we don't need to have slow try...catch in evaluate() function */
    testExpression();
}

void MatrixElement::initParser(const std::map<std::string, double> &parameters, const std::map<std::string, double> &massSquares)
{
}

void MatrixElement::setConstants(const std::vector<double> &couplings, const std::vector<double> &massSquares)
{
    couplings_internal = couplings;
    msq_internal = massSquares;
}

double MatrixElement::evaluate(double s, double t, double u) 
{
    s_internal = s;
    t_internal = t;
    u_internal = u;

    return parser->Eval();
}

void MatrixElement::defineSymbol(const std::string &symbol)
{
    try
    {
        parser->DefineVar(symbol, )
    }
    catch (mu::Parser::exception_type &parserException) 
    {
        std::cerr << "=== Error when initializing symbol '" << symbol << "'. Parser threw error: \n"; 
        std::cerr << parserException.GetMsg() << std::endl;
    }
}

void MatrixElement::testExpression() 
{
    // try evaluate at some random values
    try {
        evaluate(-4.2, 2.9, 0);
    } catch (mu::Parser::exception_type &parserException) {
        std::cerr << "=== Error when evaluating matrix element. Parser threw error: \n"; 
        std::cerr << parserException.GetMsg() << std::endl;
        std::cerr << "The expression was: \n";
        std::cerr << expression << "\n";
    }
}

} // namespace