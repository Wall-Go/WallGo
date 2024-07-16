#include "CollisionTensor.h"

#include <cassert>

namespace wallgo
{

CollisionTensor::CollisionTensor(size_t basisSize_, bool bIncludeStatisticalErrors)
    : basisSize(basisSize_),
    elementsPerDimension(basisSize - 1)
{
    data = Array4D(elementsPerDimension, elementsPerDimension, elementsPerDimension, elementsPerDimension, 0.0);
    if (bIncludeStatisticalErrors)
    {
        errors = std::make_unique<Array4D>(elementsPerDimension, elementsPerDimension, elementsPerDimension, elementsPerDimension, 0.0);
    }
}

double CollisionTensor::valueAt(size_t m, size_t n, size_t j, size_t k) const
{
    assert(m < elementsPerDimension && n < elementsPerDimension && j < elementsPerDimension && k < elementsPerDimension);
    return data[m][n][j][k];
}

double& CollisionTensor::valueAt(size_t m, size_t n, size_t j, size_t k)
{
    assert(m < elementsPerDimension && n < elementsPerDimension && j < elementsPerDimension && k < elementsPerDimension);
    return data[m][n][j][k];
}

double CollisionTensor::errorAt(size_t m, size_t n, size_t j, size_t k) const
{
    assert(hasStatisticalErrors());
    assert(m < elementsPerDimension && n < elementsPerDimension && j < elementsPerDimension && k < elementsPerDimension);
    return (*errors.get())[m][n][j][k];
}

double& CollisionTensor::errorAt(size_t m, size_t n, size_t j, size_t k)
{
    assert(hasStatisticalErrors());
    assert(m < elementsPerDimension && n < elementsPerDimension && j < elementsPerDimension && k < elementsPerDimension);
    return (*errors.get())[m][n][j][k];
}

CollisionTensor::CollisionTensor(const CollisionTensor& other)
    : CollisionTensor(other.basisSize, false)
{
    if (other.hasStatisticalErrors())
    {
        errors = std::make_unique<Array4D>(*other.errors);
    }
}

CollisionTensor& CollisionTensor::operator=(const CollisionTensor& other)
{
    if (this == &other) return *this;

    data = other.data;
    if (other.hasStatisticalErrors())
    {
        errors = std::make_unique<Array4D>(*other.errors);
    }
    else
    {
        errors.reset();
    }
    return *this;
}

CollisionTensor::CollisionTensor(CollisionTensor&& other) noexcept
    : CollisionTensor(other.basisSize, false)
{
    data = other.data;
    errors = std::move(other.errors);
}

CollisionTensor& CollisionTensor::operator=(CollisionTensor&& other) noexcept
{
    if (this == &other) return *this;

    data = std::move(other.data);
    errors = std::move(other.errors);
    return *this;
}

} // namespace