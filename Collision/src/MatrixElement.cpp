#include <iostream>
#include <cassert>

#include "muParser.h" // math expression parser
#include "MatrixElement.h"
#include "ModelParameters.h"


namespace wallgo
{

const std::vector<std::string> MatrixElement::RESERVED_SYMBOLS = { "_s", "_t", "_u" };

MatrixElement::MatrixElement() 
    : mMandelstam(0, 0, 0)
{
    initParser();
}

MatrixElement::~MatrixElement()
{
    clearParser();
}

MatrixElement::MatrixElement(const MatrixElement& other)
{
    init(other.mExpression, other.mParticleIndices, other.mSymbols);
}

MatrixElement& MatrixElement::operator=(const MatrixElement &other)
{
    if (this == &other) return *this;

    init(other.mExpression, other.mParticleIndices, other.mSymbols);
    return *this;
}

bool MatrixElement::init(
    const std::string& expression,
    const std::vector<int32_t> externalParticleIndices,
    const std::unordered_map<std::string, double>& symbols)
{
    mSymbols.clear();
    clearParser();

    mParticleIndices = externalParticleIndices;

    initParser();

    bool bInitOK = true;

    for (const auto& [symbol, value] : symbols)
    {
        bInitOK &= defineSymbol(symbol, value);
    }

    bInitOK &= setExpression(expression);
    
    return bInitOK;
}

bool MatrixElement::setExpression(const std::string &expressionIn)
{
    mExpression = expressionIn;
    parser.SetExpr(mExpression);

    // Do sensibility checks here so that we can skip them in performance critical sections
    return testExpression();
}


void MatrixElement::updateModelParameters(const ModelParameters& parameters)
{
    for (auto const& [name, newValue] : parameters.getParameterMap())
    {
        // Change only keys that have been defined as symbols
        if (mSymbols.count(name) > 0)
        {
            updateModelParameter(name, newValue);
        }
    }
}

void MatrixElement::updateModelParameter(const std::string &name, double newValue)
{
    if (mSymbols.count(name) > 0)
    {
        mSymbols[name] = newValue;
    }
    else
    {
        std::cerr << "MatrixElement::updateModelParameter called with parameter " << name << ", but the parameter has not been defined" << std::endl;
    }
}

double MatrixElement::evaluate(const Mandelstam& mandelstams)
{
    mMandelstam = mandelstams;
    return parser.Eval();
}

bool MatrixElement::defineSymbol(const std::string &symbol, double initValue)
{
    try
    {
        mSymbols[symbol] = initValue;
        parser.DefineVar(symbol, &mSymbols.at(symbol));
    }
    catch (mu::Parser::exception_type &parserException) 
    {
        std::cerr << "=== Error when defining symbol '" << symbol << "'. Parser threw error: \n"; 
        std::cerr << parserException.GetMsg() << std::endl;
        return false;
    }
    return true;
}

void MatrixElement::initParser()
{
    parser.SetExpr("0");

    // To allow variable names like msq[2] we need to add [] to parser's character list
    parser.DefineNameChars("0123456789_abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]");

    parser.DefineVar("_s", &mMandelstam.s);
    parser.DefineVar("_t", &mMandelstam.t);
    parser.DefineVar("_u", &mMandelstam.u);
}

void MatrixElement::clearParser()
{
    parser.SetExpr("0");
    parser.ClearVar();
}

bool MatrixElement::testExpression() 
{
    // try evaluate at some random values
    try
    {
        evaluate(Mandelstam(-4.2, 2.9, 0));
    }
    catch (mu::Parser::exception_type &parserException)
    {
        std::cerr << "=== Error when evaluating matrix element. Parser threw error: \n"; 
        std::cerr << parserException.GetMsg() << std::endl;
        std::cerr << "The expression was: \n";
        std::cerr << mExpression << "\n";

        return false;
    }
    return true;
}

} // namespace