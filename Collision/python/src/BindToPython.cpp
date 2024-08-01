
/** This file defines a simple interface for calling collision 
 * integral routines from Python using the pybind11 library. **/

#include <iostream>
#include <string>
#include <cstdlib> // std::atexit

#include "WallGo/Common.h"
#include "WallGo/ModelParameters.h"
#include "WallGo/CollisionTensor.h"
#include "WallGo/ParticleSpecies.h"
#include "WallGo/PhysicsModel.h"

// Python bindings
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>


namespace wallgo
{

// Module definition. This block gets executed when the module is imported.
PYBIND11_MODULE(_WallGoCollision, m)
{
    namespace py = pybind11;

    m.doc() = "WallGo collision module";

#if WG_DEBUG
    std::cout << "Warning: Loaded debug build of WallGo Collision module. Expect poor performance." << std::endl;
#endif

    wallgo::initializeRNG();

    // Bind GSL seed setter
    m.def("setSeed", &wallgo::setSeed, py::arg("seed"), "Set seed used by Monte Carlo integration. Default is 0.");


    py::class_<IntegrationOptions>(m, "IntegrationOptions")
        .def(py::init<>(), "Struct for configuring collision integration")
        .def_readwrite("maxIntegrationMomentum", &IntegrationOptions::maxIntegrationMomentum, "Upper limit for momentum integration in units of temperature (|p|/T).")
        .def_readwrite("calls", &IntegrationOptions::calls, "")
        .def_readwrite("relativeErrorGoal", &IntegrationOptions::relativeErrorGoal, "")
        .def_readwrite("absoluteErrorGoal", &IntegrationOptions::absoluteErrorGoal, "")
        .def_readwrite("maxTries", &IntegrationOptions::maxTries, "")
        .def_readwrite("bOptimizeUltrarelativistic", &IntegrationOptions::bOptimizeUltrarelativistic, "Allow optimized evaluation of ultrarelativistic processes")
        .def_readwrite("bIncludeStatisticalErrors", &IntegrationOptions::bIncludeStatisticalErrors, "Whether to store statistical error estimates of integration results");

    py::class_<CollisionTensorVerbosity>(m, "CollisionTensorVerbosity")
    .def(py::init<>());
    

    /*

    // Bind particle type enums
    py::enum_<EParticleType>(m, "EParticleType")
        .value("BOSON", EParticleType::BOSON)
        .value("FERMION", EParticleType::FERMION)
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

    // Bind IntegrationOptions struct
    py::class_<IntegrationOptions>(m, "IntegrationOptions")
        .def(py::init<>())
        .def_readwrite("maxIntegrationMomentum", &IntegrationOptions::maxIntegrationMomentum)
        .def_readwrite("calls", &IntegrationOptions::calls)
        .def_readwrite("relativeErrorGoal", &IntegrationOptions::relativeErrorGoal)
        .def_readwrite("absoluteErrorGoal", &IntegrationOptions::absoluteErrorGoal)
        .def_readwrite("maxTries", &IntegrationOptions::maxTries)
        .def_readwrite("bOptimizeUltrarelativistic", &IntegrationOptions::bOptimizeUltrarelativistic);

    //*********** Bind functions of the main control class

    // READMEs for the functions
    std::string usage_CollisionManager = 
        "Constructor for CollisionTensor class.\n";

    std::string usage_addParticle =
        "Add a new particle species \n\n"
        "Args:\n"
        "    particle (ParticleSpecies): Particle to add\n";

    std::string usage_setVariable = 
        "Sets value of a physics parameter used in matrix elements. The variable needs to already be defined previously with defineVariable()"
        "Args:\n"
        "    name (string): Variable name"
        "    newValue (double): New value of the variable\n";

    std::string usage_calculateAllIntegrals =
        "Calculates all collision integrals with the currently defined particle content and stores in .hdf5 file."
        "This is the main computation routine and will typically run for a while."
        "Call only after specifying all particles and couplings with addParticle, addCoupling.\n\n"
        "Args:\n"
        "   verbose = false (bool): Floods stdout with intermediate results. For debugging only.\n\n";

    std::string usage_setOutputDirectory =
        "Set output directory for collision integral results.\n"
        "Args:\n"
        "   path (string)";

    std::string usage_setMatrixElementFile =
        "Specify file path where matrix elements are read from.\n"
        "Args:\n"
        "   path (string)";

    std::string usage_configureIntegration =
        "Specify options for the integration routine.\n"
        "Args:\n"
        "   options (IntegrationOptions)";

    // For functions with default args we need to explicity define the arguments
    py::class_<CollisionPython>(m, "CollisionTensor")
        .def(py::init<>(), usage_CollisionManager.c_str())
        .def("addParticle", &CollisionPython::addParticle, usage_addParticle.c_str())
        .def("setVariable", &CollisionPython::setVariable, usage_setVariable.c_str())
        .def("calculateAllIntegrals", &CollisionPython::calculateAllIntegrals,
            usage_calculateAllIntegrals.c_str(), py::arg("bVerbose")=false)
        .def("setOutputDirectory", &CollisionPython::setOutputDirectory, usage_setOutputDirectory.c_str())
        .def("setMatrixElementFile", &CollisionPython::setMatrixElementFile, usage_setMatrixElementFile.c_str())
        .def("configureIntegration", &CollisionPython::configureIntegration, usage_configureIntegration.c_str());

    */

    //std::atexit(wallgo::cleanup);
}

} // namespace