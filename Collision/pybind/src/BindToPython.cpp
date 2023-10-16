
/** This file defines a simple interface for calling collision 
 * integral routines from Python using the pybind11 library. **/

#include <iostream>
#include <string>

#include "CollisionIntegral.h"
#include "ParticleSpecies.h"
#include "Collision.h"

// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>




PYBIND11_MODULE(CollisionModule, m) {

    namespace py = pybind11;

    // Bind particle type enums
    py::enum_<EParticleType>(m, "EParticleType")
        .value("BOSON", EParticleType::BOSON)
        .value("FERMION", EParticleType::FERMION)
        // Add more enum values here if needed
        .export_values();
    

    // Bind constructor for ParticleSpecies class
    py::class_<ParticleSpecies>(m, "ParticleSpecies")
        .def(py::init<std::string, EParticleType, bool, bool, double, double>(),
        py::arg("particleName"),
        py::arg("particleType"),
        py::arg("isInEquilibrium"),
        py::arg("ultrarelativistic"),
        py::arg("msqVacuum"),
        py::arg("msqThermal"),
        "Constructor for ParticleSpecies.\n\n"
        "Args:\n"
        "    particleName (str): Name of the particle species.\n"
        "    particleType (EParticleType): Type of particle (boson or fermion).\n"
        "    isInEquilibrium (bool): Whether the species is in equilibrium.\n"
        "    ultrarelativistic (bool): Treat the particle as ultrarelativistic (m=0)?\n"
        "    msqVacuum (float): Square of the vacuum mass (in units of T^2).\n"
        "    msq_thermal (float): Square of the thermal mass (in units of T^2).\n"
    );


    //*********** Bind functions of the main control class Collision

    // READMEs for the functions
    std::string usage_Collision = 
        "Constructor for Collision class. \n\n"
        "Args:\n"
        "    polynomialBasisSize (unsigned int): Defines size of the polynomial grid\n";

    std::string usage_addParticle =
        "Add a new particle species \n\n"
        "Args:\n"
        "    particle (ParticleSpecies): Particle to add\n";

    std::string usage_addCoupling = 
        "Add a new coupling constant. This is intended for action/Lagrangian parameters."
        "Do NOT use for particle thermal/vacuum masses.\n\n"
        "Args:\n"
        "    coupling (double): Coupling to add\n";

    std::string usage_calculateCollisionIntegrals =
        "Calculates all collision integrals with the currently defined particle content and stores in .hdf5 file."
        "This is the main computation routine and will typically run for a while."
        "Call only after specifying all particles and couplings with addParticle, addCoupling\n\n";


    py::class_<Collision>(m, "Collision")
        .def(py::init<uint>(), py::arg("polynomialBasisSize"), usage_Collision.c_str())
        .def("addParticle", &Collision::addParticle, usage_addParticle.c_str())
        .def("addCoupling", &Collision::addCoupling, usage_addCoupling.c_str())
        .def("calculateCollisionIntegrals", &Collision::calculateCollisionIntegrals, usage_calculateCollisionIntegrals.c_str());





/*
    // Bind the InputData struct
    py::class_<InputData>(m, "InputData")
        .def(pybind11::init<>())
        .def_readwrite("values", &InputData::values);
*/

}


/*
// Data struct that can be passed on from python. Can be used for 'additional' inputs like matrix elements etc
struct InputData {
    std::map<std::string, pybind11::object> values;
};



// Test function
void pybindTestFunction(const InputData& inputData) {

    std::cout << "hello test1 from laurin\n";
    (void)inputData; // suppress -Wunused-variable
}
*/
