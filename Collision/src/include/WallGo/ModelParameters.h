#pragma once

#include <string>
#include <unordered_map>
#include <type_traits>
#include <cassert>
#include <vector>

namespace wallgo
{

/* NOTE: in the future we could extend TModelParameters to allow for grid-dependent parameters.
These could be std::functions, or just arrays of doubles that the user must manually compute elsewhere.
A function is probably better because then CollisionIntegrals can handle computations on their own grids. */

/* Holds physics model specific parameters that enter matrix elements.
Wraps around std::unordered_map. */
template<typename Value_t>
class TModelParameters
{
public:

    /* Adds a new parameter with specified name and value, or modifies an existing parameter. */
    void add(const std::string& paramName, const Value_t& newValue);

    [[deprecated("TModelParameters::addOrModifyParameter() is deprecated. Use TModelParameters::add()")]]
    void addOrModifyParameter(const std::string& paramName, const Value_t& newValue) { add(paramName, newValue); }

    /* Returns reference to the specified parameter. Throws std::out_of_range if the parameter is not found.
    This is similar to std::unordered_map::at() but we throw a more informative error message. */
    Value_t& at(const std::string& paramName);

    /* Const version of getParameter() */
    const Value_t& at(const std::string& paramName) const;

    /* Returns reference to the specified parameter. If not found, adds that parameter with a default value.
    Can be used as: myParams["key"] = newValue; */
    Value_t& operator[](const std::string& paramName) { return mParams[paramName]; }

    [[deprecated("TModelParameters::getParameterValue() is deprecated. Use TModelParameters::at()")]]
    Value_t& getParameterValue(const std::string& paramName) const { return at(paramName); }
    
    // Removes a parameter if it exists. Note that this will invalidate existing iterators to the object
    void remove(const std::string& paramName);

    // True if we contain the specified parameter name
    bool contains(const std::string& paramName) const { return mParams.count(paramName) > 0; }
    // Empties the container
    void clear() { mParams.clear(); }
    // Get number of parameters in the container
    size_t size() const { return mParams.size(); }
    // Get number of parameters in the container
    size_t getNumParams() const { return mParams.size(); }
    // Returns array containing names of all known parameters
    std::vector<std::string> getParameterNames() const;

    // Const access to the underlying map
    const std::unordered_map<std::string, Value_t>& getParameterMap() const { return mParams; }

private:
    std::unordered_map<std::string, Value_t> mParams;
};

template<typename Value_t>
inline void TModelParameters<Value_t>::add(const std::string& paramName, const Value_t& newValue)
{
    mParams[paramName] = newValue;
}

template<typename Value_t>
inline Value_t& TModelParameters<Value_t>::at(const std::string& paramName)
{
    if (!contains(paramName))
    {
        const std::string errorMsg = "Parameter '" + paramName + "' not found in ModelParameters.\n";
        throw std::out_of_range(errorMsg);
    }

    return mParams[paramName];
}

template<typename Value_t>
inline const Value_t& TModelParameters<Value_t>::at(const std::string& paramName) const
{
    auto it = mParams.find(paramName);
    if (it == mParams.end())
    {
        const std::string errorMsg = "Parameter '" + paramName + "' not found in ModelParameters.\n";
        throw std::out_of_range(errorMsg);
    }

    return it->second;
}

template<typename Value_t>
inline void TModelParameters<Value_t>::remove(const std::string& paramName)
{
    mParams.erase(paramName);
}

template<typename Value_t>
inline std::vector<std::string> TModelParameters<Value_t>::getParameterNames() const
{
    std::vector<std::string> outNames;
    if (getNumParams() > 0) outNames.reserve(getNumParams());

    for (const auto& [key, _] : mParams)
    {
        outNames.push_back(key);
    }

    return outNames;
}

using ModelParameters = TModelParameters<double>;

} // namespace
