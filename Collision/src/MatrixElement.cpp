#include <iostream>
#include <functional>

#include "muParser.h" // math expression parser
#include "MatrixElement.h"
#include "hdf5Interface.h"
#include "ParticleSpecies.h"
#include "CollElem.h"
#include "CollisionIntegral.h"
#include <array>

MatrixElement::MatrixElement() {

    parser = new mu::Parser();
    s_internal = 0; t_internal = 0; u_internal = 0;

    parser->SetExpr("0"); 
}

MatrixElement::~MatrixElement()
{
    delete parser;
}

MatrixElement::MatrixElement(const MatrixElement& other)
{
    couplings_internal = other.couplings_internal;
    msq_internal = other.msq_internal;
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
    parser->DefineVar("s", &s_internal);
    parser->DefineVar("t", &t_internal);
    parser->DefineVar("u", &u_internal);

    setConstants(couplings, massSquares);

    // To allow variable names like msq[2] we need to add [] to parser's character list
    parser->DefineNameChars("0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]");

    try {
        for (uint i = 0; i < couplings_internal.size(); ++i) {
            std::string coupling_str = "couplings[" + std::to_string(i) + "]";
            parser->DefineVar(coupling_str, &couplings_internal[i]);
        }

        for (uint i = 0; i < msq_internal.size(); ++i) {
            std::string coupling_str = "msq[" + std::to_string(i) + "]";
            parser->DefineVar(coupling_str, &msq_internal[i]);
        }

    } catch (mu::Parser::exception_type &parserException) {
        std::cout << "=== Error when initializing symbols. Parser threw error: \n"; 
        std::cout << parserException.GetMsg() << std::endl;
    }
}

void MatrixElement::setExpression(const std::string &expressionIn)
{
    expression = expressionIn;
    parser->SetExpr(expression);

    /* Do checks here so that we don't need to have slow try...catch in evaluate() function */
    testExpression();
}

void MatrixElement::setConstants(const std::vector<double> &couplings, const std::vector<double> &massSquares)
{
    couplings_internal = couplings;
    msq_internal = massSquares;
}

double MatrixElement::evaluate(double s, double t, double u) {

    s_internal = s;
    t_internal = t;
    u_internal = u;

    return parser->Eval();
}

void MatrixElement::testExpression() {

    // try evaluate at some random values
    try {
        evaluate(-4.2, 2.9, 0);
    } catch (mu::Parser::exception_type &parserException) {
        std::cout << "=== Error when evaluating matrix element. Parser threw error: \n"; 
        std::cout << parserException.GetMsg() << std::endl;
        std::cout << "The expression was: \n";
        std::cout << expression << "\n";
    }
}