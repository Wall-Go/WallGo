#pragma once

#include "ModelParameters.h"

namespace wallgo
{

/** Defines an "observer pattern" interface for reacting to changes in a PhysicsModel,
eg. update internal parameters when model parameters change. */

class PhysicsModel;

struct ModelChangeContext
{
    ModelParameters changedParams;
};

class IModelObserver
{
public:
    virtual void handleModelChange(const ModelChangeContext& context) = 0;
};


} // namespace wallgo
