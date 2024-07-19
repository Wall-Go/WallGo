#ifndef MODELPARAMETERS_H_
#define MODELPARAMETERS_H_

#include <string>
#include <unordered_map>
#include <type_traits>
#include <utility> // std::pair

namespace wallgo
{


// Holds physics model specific parameters that enter matrix elements
template<typename T>
struct TModelParameters
{
public:
    /* Modifies value of specified parameter. If the parameter has not yet been defined, adds it with the specified value. */
    void addOrModifyParameter(const std::string& paramName, T newValue);

    /* Returns value of the specified parameter. If the parameter is not found, returns 0 and asserts in debug builds. */
    T getParameterValue(const std::string& paramName) const;

    // True if we contain the specified parameter name
    bool contains(const std::string& paramName) const { return params.count(paramName) > 0; }
    void clear() { params.clear(); }
    uint32_t getNumParams() const { return static_cast<uint32_t>(params.size()); }
    std::unordered_map<std::string, T> getParameterMap() const { return params; }
    std::vector<std::string> getParameterNames() const;

private:
    std::unordered_map<std::string, T> params;
};

template<typename T>
inline void TModelParameters<T>::addOrModifyParameter(const std::string& paramName, T newValue)
{
    params[paramName] = newValue;
}

template<typename T>
inline T TModelParameters<T>::getParameterValue(const std::string& paramName) const
{
    if (!contains(paramName))
    {
        assert(false && "Parameter not found");
        return static_cast<T>(0);
    }
    return params.at(paramName);
}

template<typename T>
inline std::vector<std::string> TModelParameters<T>::getParameterNames() const
{
    std::vector<std::string> outNames;
    if (getNumParams() > 0) outNames.reserve(getNumParams());
    
    for (const auto& [key, _] : params)
    {
        outNames.push_back(key);
    }

    return outNames;
}

template<typename Name_t, typename Index_t>
using TParticleNameMap = std::unordered_map<Name_t, Index_t>;

// Map particle name -> particle index
using ParticleNameMap = TParticleNameMap<std::string, uint32_t>;
using ParticleNamePair = std::pair<ParticleNameMap::key_type, ParticleNameMap::key_type>;

using ModelParameters = TModelParameters<double>;

} // namespace

#endif // header guard
