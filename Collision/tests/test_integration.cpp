#include <gtest/gtest.h>

#include "testparams.h"
#include "CollisionIntegral.h"

// Test 5-dimensional collision integration
// The expected values here are cross-checked against (fixed) version of Benoit's python code

// Test top-quark collision integrals using Vegas Monte Carlo integrator. Only QCD interactions are included here 
TEST(VegasIntegral, topQuarkQCD) {

    TestParams testParams = getTestParams();

    using CollisionElement = CollElem<4>;

    CollisionElement tt_gg({ testParams.topQuark, testParams.topQuark, testParams.gluon, testParams.gluon });
    CollisionElement tg_tg({ testParams.topQuark, testParams.gluon, testParams.topQuark, testParams.gluon });
    CollisionElement tq_tq({ testParams.topQuark, testParams.lightQuark, testParams.topQuark, testParams.lightQuark });

    CollisionIntegral4 collisionIntegral(testParams.basisSize);

    collisionIntegral.addCollisionElement(tt_gg);
    collisionIntegral.addCollisionElement(tg_tg);
    collisionIntegral.addCollisionElement(tq_tq);

    std::cout << "Starting integration tests, this may take a while...\n";

    std::size_t m, n, j, k;

    m = 2; n = 1; j = 1; k = 1;
    std::array<double, 2> res = collisionIntegral.evaluate(m, n, j, k);

    double expected = -0.00031601792353679991;
    // tolerance. This is the absolute error of integration from when the expected value was computed
    double tolerance = 8.41424e-07;
    EXPECT_NEAR(res[0], expected, tolerance);



    // Other random point
    m = 5; n = 7; j = 3; k = 14;

    res = collisionIntegral.evaluate(m, n, j, k);

    expected = 0.455390104532744;
    tolerance = 0.00226865776502422;
    EXPECT_NEAR(res[0], expected, tolerance);


    //***** Test symmetries 

    const std::size_t N = testParams.basisSize;

    // C[Tm(-x)Tn(y)] = (-1)^m C[Tm(x)Tn(y)]
    
    // Handpick some j, m values
    std::vector<std::size_t> jValues {1, 6, 10};
    std::vector<std::size_t> mValues {5, 14};

    for (auto J : jValues) {
        for (auto M : mValues) {

            std::array<double, 2> res1 = collisionIntegral.evaluate(M, n, J, k);
            std::array<double, 2> res2 = collisionIntegral.evaluate(M, n, N - J, k);
            int sign = (M % 2 == 0 ? 1 : -1);

            // Let's allow at least 5% relative error; if statistical error is larger, use that instead 
            double tolerance = std::max(0.05 * std::abs(res1[0]), std::abs(res1[1]));
            tolerance = std::max(tolerance, std::abs(res2[1])); 
            EXPECT_NEAR(res1[0], sign*res2[0], tolerance);
        }
    }

    /*  
    // All j,m combinations. Slow!
    for (j = 1; j <= N / 2; ++j) {
        for (m = 2; m < N; ++m) {

            std::array<double, 2> res1 = collisionIntegral.evaluate(m, n, j, k);
            std::array<double, 2> res2 = collisionIntegral.evaluate(m, n, N - j, k);

            int sign = (m % 2 == 0 ? 1 : -1);

            printf("%ld %ld %g %g\n", m, j, res1[0], res2[0]);

            double tolerance = std::abs(res1[1]); 
            EXPECT_NEAR(res1[0], sign*res2[0], tolerance);
        }
    }
    */

}