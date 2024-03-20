#include <iostream>
#include <functional>
#include <array>
#include <assert.h>

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
    const std::map<std::string, double> parameters = other.parametersInternal;
    expression = other.getExpression();

    parser = new mu::Parser();
    initSymbols(parameters);
    parser->SetExpr(expression);
}

void MatrixElement::operator=(const MatrixElement &other)
{
    const std::map<std::string, double> parameters = other.parametersInternal;
    expression = other.getExpression();

    // deleting this just to make sure we clear everything
    delete parser;

    parser = new mu::Parser();
    initSymbols(parameters);
    parser->SetExpr(expression);
}


void MatrixElement::setExpression(const std::string &expressionIn)
{
    expression = expressionIn;
    parser->SetExpr(expression);

    /* Do checks here so that we don't need to have slow try...catch in evaluate() function */
    testExpression();
}


void MatrixElement::initSymbols(const std::map<std::string, double> &parameters)
{
    parametersInternal.clear();

    for (auto const& [symbol, value] : parameters)
    {
        defineSymbol(symbol, value);
    }
}

void MatrixElement::setParameters(const std::map<std::string, double> &parameters)
{
    for (auto const& [key, val] : parametersInternal)
    {
        // NB: check only in debug builds, otherwise just assume that the key exists
        assert(parameters.count(key) > 0);

        parametersInternal[key] = parameters.at(key);
    }
}

double MatrixElement::evaluate(double s, double t, double u)
{
    s_internal = s;
    t_internal = t;
    u_internal = u;

    return parser->Eval();
}

void MatrixElement::defineSymbol(const std::string &symbol, double initValue)
{
    try
    {
        parametersInternal[symbol] = initValue;
        parser->DefineVar(symbol, &parametersInternal.at(symbol));
    }
    catch (mu::Parser::exception_type &parserException) 
    {
        std::cerr << "=== Error when defining symbol '" << symbol << "'. Parser threw error: \n"; 
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