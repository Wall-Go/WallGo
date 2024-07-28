#pragma once

#include <string>
#include <unordered_map>
#include <type_traits>
#include <cassert>

namespace wallgo
{


/* Holds physics model specific parameters that enter matrix elements.
Wraps around std::unordered_map. */
template<typename T>
struct TModelParameters
{
public:
    /* Modifies value of specified parameter. If the parameter has not yet been defined, adds it with the specified value. */
    void addOrModifyParameter(const std::string& paramName, T newValue);

    /* Returns value of the specified parameter. If the parameter is not found, returns 0 and asserts in debug builds. */
    T getParameterValue(const std::string& paramName) const;

    // True if we contain the specified parameter name
    bool contains(const std::string& paramName) const { return mParams.count(paramName) > 0; }
    void clear() { params.clear(); }
    size_t size() const { return mParams.size(); }
    uint32_t getNumParams() const { return static_cast<uint32_t>(mParams.size()); }
    std::vector<std::string> getParameterNames() const;

    // Const access to the underlying map
    const std::unordered_map<std::string, T>& getParameterMap() const { return mParams; }

private:
    std::unordered_map<std::string, T> mParams;
};

template<typename T>
inline void TModelParameters<T>::addOrModifyParameter(const std::string& paramName, T newValue)
{
    mParams[paramName] = newValue;
}

template<typename T>
inline T TModelParameters<T>::getParameterValue(const std::string& paramName) const
{
    if (!contains(paramName))
    {
        assert(false && "Parameter not found");
        return static_cast<T>(0);
    }
    return mParams.at(paramName);
}

template<typename T>
inline std::vector<std::string> TModelParameters<T>::getParameterNames() const
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
