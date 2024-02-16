
/** This file defines a simple interface for calling collision 
 * integral routines from Python using the pybind11 library. **/

#include <iostream>
#include <string>

#include "WallGoCollision/CollisionIntegral.h"
#include "WallGoCollision/ParticleSpecies.h"
#include "WallGoCollision/CollisionManager.h"
#include "WallGoCollision/gslWrapper.h"

// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace pythonModule 
{
    bool bInitialized = false;

    /* Initialization function. This should be called from Python before doing anything else with the module.
    * Do things like initial allocs here.
    */
    void initModule(const std::string& configFileName) 
    {

        // Already initialized?
        if (pythonModule::bInitialized) 
        {
            std::cout << "! Module already initialized, doing nothing\n";
            return;
        }

        gslWrapper::initializeRNG();

        pythonModule::bInitialized = true;
    }
}


/* @TODO in principle we'd need some cleanup routine that eg. calls gslWrapper::clearRNG().
But seems hard to dictate when/how this should be called in Python context.
*/

/* We bind a subclass of the CollisionManager "control" class. 
This way we can override some functions with python-specific functionality.
Marking this as final prevents some non-virtual destructor warnings from clang. */
class CollisionPython final : public CollisionManager
{

public: 
    // Just call parent constructor
    CollisionPython() : CollisionManager() 
    {
        if (!pythonModule::bInitialized)
        {
            std::cerr << "Error: Collision constructor called, but the module has not been initialized. Please call initModule().\n";
            std::cerr << "This error is unrecoverable!" << std::endl;
            exit(111);
        }
    }

protected:

    // NB: if called inside OpenMP block this does lead to core dumped on CTRL-C
    // So @todo make a clean exit
    virtual inline bool shouldContinueEvaluation() final override 
    {
        // https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-properly-handle-ctrl-c-in-long-running-functions
        if (PyErr_CheckSignals() != 0)
        {
            throw pybind11::error_already_set();
            return false;
        }
        return true;
    }

};



// Module definition. This block gets executed when the module is imported.
PYBIND11_MODULE(CollisionModule, m) 
{

    namespace py = pybind11;

    // Bind variable
    m.attr("bInitialized") = pythonModule::bInitialized;

    // Bind initialization function
    m.def("initModule", &pythonModule::initModule, py::arg("configFileName") = "./config.ini",
        "Initialize the module. This needs to be called before using the module for anything. Takes path/name of the config file as argument (default: config.ini)"); 


    // Bind particle type enums
    py::enum_<EParticleType>(m, "EParticleType")
        .value("BOSON", EParticleType::BOSON)
        .value("FERMION", EParticleType::FERMION)
        // Add more enum values here if needed
        .export_values();
    

    // Bind constructor for ParticleSpecies class
    py::class_<ParticleSpecies>(m, "ParticleSpecies")
        .def(py::init<std::string, EParticleType, bool, double, double, bool>(),
        py::arg("particleName"),
        py::arg("particleType"),
        py::arg("isInEquilibrium"),
        py::arg("msqVacuum"),
        py::arg("msqThermal"),
        py::arg("ultrarelativistic"),
        "Constructor for ParticleSpecies.\n\n"
        "Args:\n"
        "    particleName (str): Name of the particle species.\n"
        "    particleType (EParticleType): Type of particle (boson or fermion).\n"
        "    isInEquilibrium (bool): Whether the species is in equilibrium.\n"
        "    msqVacuum (float): Square of the vacuum mass (in units of T^2).\n"
        "    msq_thermal (float): Square of the thermal mass (in units of T^2).\n"
        "    ultrarelativistic (bool): Treat the particle as ultrarelativistic (m=0)?\n"
    );


    //*********** Bind functions of the main control class

    // READMEs for the functions
    std::string usage_CollisionManager = 
        "Constructor for CollisionManager class. \n\n"
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


    py::class_<CollisionPython>(m, "CollisionManager")
        .def(py::init<>(), usage_CollisionManager.c_str())
        .def("addParticle", &CollisionPython::addParticle, usage_addParticle.c_str())
        .def("addCoupling", &CollisionPython::addCoupling, usage_addCoupling.c_str())
        .def("calculateCollisionIntegrals", &CollisionPython::calculateCollisionIntegrals, usage_calculateCollisionIntegrals.c_str());

}