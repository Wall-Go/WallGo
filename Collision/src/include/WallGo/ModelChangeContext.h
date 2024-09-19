#pragma once

#include "ModelParameters.h"

#include <vector>

namespace wallgo
{

struct ParticleChangeContext
{
    // Index of the changed particle
    uint32_t particleIndex;
    double newMassSq;
};

struct ModelChangeContext
{
    ModelParameters changedParams;
    std::vector<ParticleChangeContext> changedParticles;
};

}