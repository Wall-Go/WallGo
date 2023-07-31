#include <gtest/gtest.h>

#include "testparams.h"
#include "CollisionIntegral.h"

// Test 5-dimensional collision __integrands__

// tt -> gg integrand
TEST(Integrand, ttgg) {
    using CollisionElement = CollElem<4>;

    CollisionElement tt_gg({ testparams::topQuark, testparams::topQuark, testparams::gluon, testparams::gluon });

    CollisionIntegral4 collisionIntegral(testparams::basisSize);

    collisionIntegral.addCollisionElement(tt_gg);


    int m, n, j, k;
    double res; 

    m = 5; n = 7; j = 3; k = 14;

    res = collisionIntegral.calculateIntegrand(testparams::p2, testparams::phi2, testparams::phi3, 
                    testparams::cosTheta2, testparams::cosTheta3, m, n, j, k);

    // Tested against fixed version of Benoit's code
    double expected = -0.00014894952394043577;
    // tolerance
    double relativeError = 1e-8; 
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));

    // test also some possibly problematic values // TODO

}