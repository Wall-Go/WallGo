
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
#include <pybind11/functional.h>

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
    //std::atexit(wallgo::cleanup); // seems to behave badly...?

    // Bind GSL seed setter
    m.def("setSeed", &wallgo::setSeed, py::arg("seed"), "Set seed used by Monte Carlo integration. Default is 0.");

    py::class_<GridPoint>(m, "GridPoint", "")
        .def(py::init<>());

    py::class_<IntegrationOptions>(m, "IntegrationOptions", "Struct for configuring collision integration")
        .def(py::init<>())
        .def_readwrite("maxIntegrationMomentum", &IntegrationOptions::maxIntegrationMomentum, "Upper limit for momentum integration in units of temperature (|p|/T).")
        .def_readwrite("calls", &IntegrationOptions::calls, "")
        .def_readwrite("relativeErrorGoal", &IntegrationOptions::relativeErrorGoal, "")
        .def_readwrite("absoluteErrorGoal", &IntegrationOptions::absoluteErrorGoal, "")
        .def_readwrite("maxTries", &IntegrationOptions::maxTries, "")
        .def_readwrite("bOptimizeUltrarelativistic", &IntegrationOptions::bOptimizeUltrarelativistic, "Allow optimized evaluation of ultrarelativistic processes")
        .def_readwrite("bIncludeStatisticalErrors", &IntegrationOptions::bIncludeStatisticalErrors, "Whether to store statistical error estimates of integration results");

    py::class_<CollisionTensorVerbosity>(m, "CollisionTensorVerbosity", "Struct for configuring verbosity and progress tracking of collision integration")
        .def(py::init<>())
        .def_readwrite("progressReportPercentage",
            &CollisionTensorVerbosity::progressReportPercentage,
            R"(Print progress report and time estimate to stdout when this percantage of grid integrals have been computed.
            Should be in range[0, 1]. Value of 0 means no reporting and values >= 1 mean we only report at end.
            Note that progress reporting has a small overhead particularly in multithreaded context (due to atomic operations)")
        .def_readwrite("bPrintElapsedTime", &CollisionTensorVerbosity::bPrintElapsedTime, "Print elapsed time when done?")
        .def_readwrite("bPrintEveryElement",
            &CollisionTensorVerbosity::bPrintEveryElement,
            "If true, prints every element of the collision tensor to stdout."
            "Very high overhead, intended for debugging only."
        );

    py::class_<CollisionResultsGrid>(m, "CollisionResultsGrid", "Rank 4 tensor that holds collision integration results on the grid for (particle1, particle2) pair")
        .def("hasStatisticalErrors", &CollisionResultsGrid::hasStatisticalErrors, "Returns True if statistical errors are included")
        .def("getBasisSize", &CollisionResultsGrid::getBasisSize, "Basis size of the momentum grid")
        ;


    py::class_<ModelParameters>(m, "ModelParameters", "Container for physics model-dependent parameters (couplings etc)")
        .def(py::init<>())
        .def("addOrModifyParameter", &ModelParameters::addOrModifyParameter, "Define a new named parameter or modify value of an existing one.", py::arg("name"), py::arg("value"))
        .def("getParameterValue", &ModelParameters::getParameterValue, "Get current value of specified parameter. Returns 0 if the parameter is not found (prefer the contains() method if unsure)", py::arg("name"))
        .def("contains", &ModelParameters::contains, "Returns True if the specified parameter has been defined, otherwise returns False", py::arg("name"))
        .def("clear", &ModelParameters::clear, "Empties the parameter container")
        .def("getNumParams", &ModelParameters::getNumParams, "Returns number of contained parameters")
        .def("getParameterNames", &ModelParameters::getParameterNames, "Returns list containing names of parameters that have been defined")
        // Operator[] on Python side is __getitem__. We bind a helper lambda to achieve this
        .def("__getitem__",
            [](const ModelParameters& self, const std::string& paramName)
            {
                return self[paramName];
            },
            "Get current value of specified parameter. Returns 0 if the parameter is not found (prefer the contains() method if unsure)",
            py::arg("name")
        );
        

    py::enum_<EParticleType>(m, "EParticleType")
        .value("eNone", EParticleType::eNone)
        .value("eBoson", EParticleType::eBoson)
        .value("eFermion", EParticleType::eFermion);

    py::class_<ParticleDescription>(m, "ParticleDescription", "Data-only container for describing a particle species to WallGoCollision")
        .def(py::init<>())
        .def_readwrite("name", &ParticleDescription::name, "Particle species name, must be unique.")
        .def_readwrite("index", &ParticleDescription::index, "Integer identifier for the species. Must be unique and match intended index used in matrix elements.")
        .def_readwrite("type", &ParticleDescription::type, "Particle statistics (boson or fermion).")
        .def_readwrite("bInEquilibrium", &ParticleDescription::bInEquilibrium, "Set to true if the particle species is assumed to remain in thermal equilibrium.")
        .def_readwrite("bUltrarelativistic", &ParticleDescription::bUltrarelativistic, "Whether particles should be treated as ultrarelativistic in collision processes (ie. neglect mass in dispersion relations).")
        .def_readwrite(
            "massSqFunction",
            &ParticleDescription::massSqFunction,
            R"(Function with signature WallGoCollision.ModelParameters -> float that calculates
               particle mass-squared. You must specify this function for all particle types that are NOT marked as ultrarelativistic.
               The output should be in units of the temperature, ie. return value is (m/T)^2.)"
        );

    py::class_<ModelDefinition>(m,
            "ModelDefinition",
            R"(Helper class for defining a WallGoCollision.PhysicsModel. Fill in your model parameters and particle content here.)"
        )
        .def(py::init<>())
        .def("defineParticleSpecies", &ModelDefinition::defineParticleSpecies, "Registers a new ParticleDescription with the model")
        // Bind the right overload, see https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
        .def("defineParameter", static_cast<void(ModelDefinition::*)(const std::string&, double)>(&ModelDefinition::defineParameter), "Defines a new symbolic parameter and its initial value")
        .def("defineParameters", &ModelDefinition::defineParameters, "Defines new symbolic parameters and their initial values");

    py::class_<PhysicsModel>(m, "PhysicsModel", "Model class used by WallGoCollisions")
        .def(py::init<const ModelDefinition&>())
        .def("updateParameter", static_cast<void(PhysicsModel::*)(const std::string&, double)>(&PhysicsModel::updateParameter), "Updates a symbolic parameter value.The symbol must have been defined at model creation time")
        .def("updateParameters", &PhysicsModel::updateParameters, "Updates model parameter values. The parameters must have been defined at model creation time")
        // Bind lambda that takes std::string instead of std::filesystem::path
        .def("readMatrixElements",
            [](PhysicsModel& self, const std::string& filePath, bool bPrintMatrixElements)
            {
                return(
                    self.readMatrixElements(std::filesystem::path(filePath), bPrintMatrixElements)
                );
            },
            R"(Read matrix elements from a file and stores them internally.
            This will only consider expressions where at least one currently registered out-of-equilibrium particle appears as an external particle.
            Note that this function clears any previously stored matrix elements for the model.
            Returns false if something goes wrong.)",
            py::arg("filePath"), py::arg("bPrintMatrixElements") = false
        );
        //.def("createCollisionTensor")

    /*
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
}

} // namespace