#include <gtest/gtest.h>

#include "testparams.h"
#include "CollisionIntegral.h"

// Test 5-dimensional collision __integrands__
// The expected values here are cross-checked against (fixed) version of Benoit's python code


// tt -> gg integrand 
TEST(Integrand, ttgg) {

    TestParams testParams = getTestParams();

    using CollisionElement = CollElem<4>;

    CollisionElement tt_gg({ testParams.topQuark, testParams.topQuark, testParams.gluon, testParams.gluon });

    CollisionIntegral4 collisionIntegral(testParams.basisSize);

    collisionIntegral.addCollisionElement(tt_gg);


    int m, n, j, k;
    double res; 

    m = 5; n = 7; j = 3; k = 14;

    double p2 = testParams.p2;
    double phi2 = testParams.phi2;
    double phi3 = testParams.phi3;
    double cosTheta2 = testParams.cosTheta2;
    double cosTheta3 = testParams.cosTheta3;

    res = collisionIntegral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, m, n, j, k);

    double expected = -0.00014894952394043577;
    // tolerance
    double relativeError = 1e-8; 
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));


    //***** Test some potentially problematic values 

    // p1 = 0? Not possible with Chebyshev polynomials: always either pZ != 0 or pPar != 0

    // p2 = 0
    p2 = 0;
    expected = 0.0;
    res = collisionIntegral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, m, n, j, k);
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));
    p2 = testParams.p2;

    // cosTheta3 = 1
    cosTheta3 = 1;
    expected = -0.00021465428912564882;
    res = collisionIntegral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, m, n, j, k);
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));
    cosTheta3 = testParams.cosTheta3;

}


// 'Full' top quark collision integrand as in 2204.13120. Involves QCD interactions only: tt -> gg, tg -> tg, tq -> tq
TEST(Integrand, topQuarkQCD) {

    TestParams testParams = getTestParams();

    using CollisionElement = CollElem<4>;

    CollisionElement tt_gg({ testParams.topQuark, testParams.topQuark, testParams.gluon, testParams.gluon });
    CollisionElement tg_tg({ testParams.topQuark, testParams.gluon, testParams.topQuark, testParams.gluon });
    CollisionElement tq_tq({ testParams.topQuark, testParams.lightQuark, testParams.topQuark, testParams.lightQuark });

    CollisionIntegral4 collisionIntegral(testParams.basisSize);

    collisionIntegral.addCollisionElement(tt_gg);
    collisionIntegral.addCollisionElement(tg_tg);
    collisionIntegral.addCollisionElement(tq_tq);

    std::size_t m, n, j, k;
    double res; 

    m = 5; n = 7; j = 3; k = 14;

    double p2 = testParams.p2;
    double phi2 = testParams.phi2;
    double phi3 = testParams.phi3;
    double cosTheta2 = testParams.cosTheta2;
    double cosTheta3 = testParams.cosTheta3;

    res = collisionIntegral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, m, n, j, k);

    double expected = 0.0017434429549811298;
    // tolerance
    double relativeError = 1e-8; 
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));

    //***** Test some potentially problematic values 

    // p2 = 0. In ultrarelativistic limit this also leads to p3 = 0 after applying delta functions
    p2 = 0;
    expected = 0.0;
    res = collisionIntegral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, m, n, j, k);
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));
    p2 = testParams.p2;

    // cosTheta3 = 1
    cosTheta3 = 1;
    expected = -0.0010994039140472176;
    res = collisionIntegral.calculateIntegrand(p2, phi2, phi3, cosTheta2, cosTheta3, m, n, j, k);
    EXPECT_NEAR(res, expected, std::abs(expected * relativeError));
    cosTheta3 = testParams.cosTheta3;

}