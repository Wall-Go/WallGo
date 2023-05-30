
/** This file defines a simple interface for calling collision 
 * integral routines from Python using the pybind11 library. **/

#include <iostream>

#include "kinematics.h"
#include "operators.h"

// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


// Data struct that can be passed on from python. Can be used for 'additional' inputs like matrix elements etc
struct InputData {
    std::map<std::string, pybind11::object> values;
};


// Test function
void pybindTestFunction(const InputData& inputData) {

    std::cout << "hello test1 from laurin\n";
}


PYBIND11_MODULE(CollisionModule, m) {

    // Bind the InputData struct
    pybind11::class_<InputData>(m, "InputData")
        .def(pybind11::init<>())
        .def_readwrite("values", &InputData::values);

    // Bind functions
    m.def("pybindTestFunction", &pybindTestFunction, 
        "Test function to call from python");
}


/* Calculate a component of the "collision tensor" C(m,j; n,k).
m,n refer to indices of basis polynomials and j,k refer to their momenta on the grid */
/* void CalcCollisionTensorComponent(int m, int j, int n, int k, const InputData& inputData) {

    int N = 20;
    CollisionIntegral coll(N);
    std::cout << "Running CalcCollisionTensorComponent()\n";

    std::vector<double> c = coll.CalcCollision(2, 1, 1, 1);
    printf("Result %g, error %g\n", c[0], c[1]);
    printf("With 2pi error: %g, error %g\n", 2*M_PI*c[0], 2*M_PI*c[1]);
} */

// Bind to python module "collision"
/* PYBIND11_MODULE(collision, m) {

    // If need to use the CollisionIntegral class directly from python, bind it here. For now I'll skip this

    // Bind the InputData struct
    pybind11::class_<InputData>(m, "InputData")
        .def(pybind11::init<>())
        .def_readwrite("values", &InputData::values);

    // Bind functions
    m.def("CalcCollisionTensorComponent", &CalcCollisionTensorComponent, 
            "Calculate a component of the 'collision tensor' C(m,j; n,k).\n m,n refer to indices of basis polynomials and j,k refer to their momenta on the grid");
}
 */