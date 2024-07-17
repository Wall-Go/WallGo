#ifndef MODELPARAMETERS_H_
#define MODELPARAMETERS_H_

#include <string>
#include <unordered_map>

namespace wallgo
{


// Holds physics model specific parameters that enter matrix elements
template<typename T>
struct ModelParameters
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
inline void ModelParameters<T>::addOrModifyParameter(const std::string& paramName, T newValue)
{
    params[paramName] = newValue;
}

template<typename T>
inline T ModelParameters<T>::getParameterValue(const std::string& paramName) const
{
    if (!contains(paramName))
    {
        assert(false && "Parameter not found");
        return static_cast<T>(0);
    }
    return params.at(paramName);
}

template<typename T>
inline std::vector<std::string> ModelParameters<T>::getParameterNames() const
{
    std::vector<std::string> outNames;
    if (getNumParams() > 0) outNames.reserve(getNumParams());
    
    for (const auto& [key, _] : params)
    {
        outNames.push_back(key);
    }

    return outNames;
}

} // namespace

#endif // header guard