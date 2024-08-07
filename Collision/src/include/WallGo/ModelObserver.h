#pragma once

namespace wallgo
{

/** Defines an "observer pattern" interface for reacting to changes in a PhysicsModel,
eg. update internal parameters when model parameters change. */

struct ModelChangeContext;


class IModelObserver
{
public:
    virtual void handleModelChange(const ModelChangeContext& context) = 0;
    // Used to avoid dangling pointers to a destroyed model
    virtual void handleModelDestruction() = 0;
};


} // namespace wallgo
