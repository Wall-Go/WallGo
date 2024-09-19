#pragma once

#include "Common.h"

namespace wallgo
{

struct WALLGO_API IntegrationOptions
{
    double maxIntegrationMomentum;
    // How many Monte Carlo calls between convergence checks 
    uint32_t calls;
    double relativeErrorGoal;
    double absoluteErrorGoal;
    uint32_t maxTries;
    // Enables faster computation of kinematic factors for ultrarelativistic collision elements. Should be no reason to disable this outside testing
    bool bOptimizeUltrarelativistic;

    bool bIncludeStatisticalErrors;

    // Set sensible defaults
    IntegrationOptions()
    {
        maxIntegrationMomentum = 20;
        calls = 50000;
        relativeErrorGoal = 1e-1;
        absoluteErrorGoal = 1e-8;
        maxTries = 50;
        bOptimizeUltrarelativistic = true;
        bIncludeStatisticalErrors = true;
    }
};


struct WALLGO_API CollisionTensorVerbosity
{
    /* Print progress report and time estimate to stdout when this percantage of grid integrals have been computed.
    Should be in range [0, 1]. Value of 0 means no reporting and values >= 1 mean we only report at end.
    Note that progress reporting has a small overhead particularly in multithreaded context (due to atomic operations) */
    double progressReportPercentage = 0;
    // Print total time when done?
    bool bPrintElapsedTime = false;

    /* If true, prints every element of the collision tensor to stdout.
    Very high overhead, intended for debugging only. */
    bool bPrintEveryElement = false;
};

} // namespace wallgo
