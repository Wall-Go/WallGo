#include <iostream>
#include <string>
#include <filesystem> // don't bind std::filesystem stuff directly, will wrap them in lambdas

#include "WallGo/Common.h"
#include "WallGo/Utils.h"
#include "WallGo/ModelParameters.h"
#include "WallGo/CollisionTensor.h"
#include "WallGo/ParticleSpecies.h"
#include "WallGo/PhysicsModel.h"
#include "WallGo/ResultContainers.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

namespace wallgo
{

/** NOTE. When working with Python bindings we must pay special care to Python's Global Interpreter Lock (GIL)
because the collision integration routines are multithreaded. Reference: https://pybind11.readthedocs.io/en/stable/advanced/misc.html#global-interpreter-lock-gil.

In our case we have a GIL problem in CollisionIntegral4::evaluateOnGrid, because:
    - Evaluation of a MatrixElement is not thread safe because it needs to set internal parameters in order to evaluate the parsed matrix element
    - evaluateOnGrid() has to take a thread-local copy of the CollisionIntegral4 object (itself) to ensure that each thread has its own MatrixElements to operate with
    - The copy process must hold GIL if CollisionIntegral4 contains any Python state, and pybind11 ensures this by default (it never implicitly releases GIL).
    This is because copying a Python object increases its internal reference count.
    - If evaluateOnGrid() holds GIL while trying to take thread-local copies in an OMP parallel region, only the main thread can do work while others hang.

In our case CollisionIntegral4 DOES contain Python state because it contains ParticleSpecies, which in turn contains std::function (the mass function) that we've bound to Python.
So taking thread-local copies while holding GIL is not possible.
Currently it's just std::functions that are problematic, all other Python-bound data that we use is immutable (ints, floats etc)
and is converted to pure C++ instead of having to keep Python state present.

The present workaround is to do a scoped release of GIL in any Python-bound function that calls CollisionIntegral4::evaluateOnGrid, see eg. the binding of computeIntegralsAll.
An earlier solution was to make CollisionElements store raw pointers to ParticleSpecies instead of copies; This also avoids the copy problem since copying a pointer can be done without
increasing reference count on Python side.
*/


/* Module definition.This block gets executed when the module is imported.
Module name is passed from cmake. */
PYBIND11_MODULE(WG_PYTHON_MODULE_NAME, m)
{
    namespace py = pybind11;

    m.doc() = "WallGo collision module";

#if WG_DEBUG
    std::cout << "Warning: Loaded debug build of WallGo Collision module. Expect poor performance." << std::endl;
#endif

#if !WITH_OMP
    std::cout << "Warning: WallGo Collision module running in single-threaded mode (no OpenMP installation found during compilation). Expect poor performance." << std::endl;
#endif

    wallgo::initializeRNG();

    // Let pybind11 handle cleanup timing, works better than std::atexit
    m.add_object("_cleanup", py::capsule(wallgo::cleanup));

    /* Check if exit signal was received from Python side.
    Here we avoid throwing and just check the flag and return bool based on it.
    See https://pybind11.readthedocs.io/en/stable/faq.html#how-can-i-properly-handle-ctrl-c-in-long-running-functions. */
    utils::gExitSignalChecker = []() -> bool
        {
            // Hacky, ensure we have GIL before accessing Python state. Necessary because we do signal checks from long-running parallelized functions during which GIL is released
            py::gil_scoped_acquire acquire;
            bool bShouldExit = static_cast<bool>(PyErr_CheckSignals());
            return bShouldExit;
        };

    // Bind GSL seed setter
    m.def("setSeed", &wallgo::setSeed, py::arg("seed"), "Set seed used by Monte Carlo integration. Default is 0.");

    py::class_<GridPoint>(m, "GridPoint", "")
        .def(py::init<>())
        .def_readwrite("m", &GridPoint::m, "First Polynomial index")
        .def_readwrite("n", &GridPoint::n, "Second polynomial index")
        .def_readwrite("j", &GridPoint::j, "First momentum index")
        .def_readwrite("k", &GridPoint::k, "Second momentum index");

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
        // Bind the constant accessors only
        .def("valueAt", static_cast<double(CollisionResultsGrid::*)(const GridPoint&) const>(&CollisionResultsGrid::valueAt), "Get value at specified grid point")
        .def("errorAt", static_cast<double(CollisionResultsGrid::*)(const GridPoint&) const>(&CollisionResultsGrid::errorAt), "Get statistical error at specified grid point")
        .def("updateValue", &CollisionResultsGrid::updateValue, "Updates value (and error, if they are included) at specified grid point")
        .def(
            "writeToHDF5",
            [](const CollisionResultsGrid& self, const std::string& filePath, bool bWriteErrors = true)
            {
                return self.writeToHDF5(std::filesystem::path(filePath), bWriteErrors);
            },
            R"(Write array contents to a HDF5 file. This always overrides the file if it exists.
            Dataset name for the integration results will be of form "particle1, particle2".
            Dataset name for integration errors will be "particle1, particle2 errors"
            Return value is False if something goes wrong.)",
            py::arg("filePath"), py::arg("bWriteErrors") = true
        );

    py::class_<CollisionTensorResult>(m, "CollisionTensorResult", "Rank 6 tensor that holds integration results on a grid for all out-of-equilibrium particle pairs")
        .def(
            "writeToIndividualHDF5",
            [](const CollisionTensorResult& self, const std::string& outDirectory, bool bWriteErrors = true)
            {
                return self.writeToIndividualHDF5(outDirectory, bWriteErrors);
            },
            R"(Writes the contents to disk, so that each particle pair is written into a separate HDF5 file.
            In practice this just calls CollisionResultsGrid::writeToHDF5 for each particle pair.
            Return value is false if something went wrong.)",
            py::arg("outDirectory"), py::arg("bWriteErrors") = true)
        .def(
            "getResultsForParticlePair",
            static_cast<CollisionResultsGrid * (CollisionTensorResult::*)(const std::string&, const std::string&)>(&CollisionTensorResult::getResultsForParticlePair),
            py::return_value_policy::reference,
            R"(Returns reference to collision integration results of the specified particle pair.
            Can be None if the pair is not found)",
            py::arg("particle1"), py::arg("particle2")
        );

    py::class_<CollisionTensor>(m,
        "CollisionTensor",
        R"(The CollisionTensor class acts as a main interface into collision integral computations.
	    It will be linked to the PhysicsModel that creates it, so that changes to model parameters and particles
	    will directly propagate to CollisionTensor objects created from it.
	    This also means that you MUST keep the model alive for as long as you use CollisionTensors linked to it.)"
    )
        // For simplicity let's require that integration always uses cached IntegrationOptions and CollisionTensorVerbosity, so remove "default" from descriptions
        .def("setIntegrationOptions",
            &CollisionTensor::setDefaultIntegrationOptions,
            R"(Configures integration options that are used by integration routines)")
        .def("setIntegrationVerbosity",
            &CollisionTensor::setDefaultIntegrationVerbosity,
            R"(Configures integration verbosity settings)")
        .def("changePolynomialBasisSize", &CollisionTensor::changePolynomialBasisSize, "Change basis size used by the polynomial grid")
        .def("countIndependentIntegrals", &CollisionTensor::countIndependentIntegrals, "Count how many independent collision integrals we have")
        .def("computeIntegralsAll",
            static_cast<CollisionTensorResult(CollisionTensor::*)()>(&CollisionTensor::computeIntegralsAll),
            /* GIL released for the duration of this function to avoid it mess with multithreading.
            FIXME How safe is this actually? Currently we do not do any thread-unsafe changes to Python state, and at least it seems to work.
            */
            py::call_guard<py::gil_scoped_release>(),
            R"(Calculates all collision integrals associated with this tensor.)"
        );

    py::class_<ModelParameters>(m, "ModelParameters", "Container for physics model-dependent parameters (couplings etc)")
        .def(py::init<>())
        .def("addOrModifyParameter", &ModelParameters::addOrModifyParameter, "Define a new named parameter or modify value of an existing one.", py::arg("name"), py::arg("value"))
        .def("getParameterValue", &ModelParameters::getParameterValue, "Get current value of specified parameter. Returns 0 if the parameter is not found (prefer the contains() method if unsure)", py::arg("name"))
        .def("contains", &ModelParameters::contains, "Returns True if the specified parameter has been defined, otherwise returns False", py::arg("name"))
        .def("clear", &ModelParameters::clear, "Empties the parameter container")
        .def("getNumParams", &ModelParameters::getNumParams, "Returns number of contained parameters")
        .def("getParameterNames", &ModelParameters::getParameterNames, "Returns list containing names of parameters that have been defined")
        // Operator[] on Python side is __getitem__. Bind a helper lambda to achieve this
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
        // Bind the right overload by explicitly casting to specific signature, see https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods
        .def("defineParameter", static_cast<void(ModelDefinition::*)(const std::string&, double)>(&ModelDefinition::defineParameter), "Defines a new symbolic parameter and its initial value")
        .def("defineParameters", &ModelDefinition::defineParameters, "Defines new symbolic parameters and their initial values");

    py::class_<PhysicsModel>(m, "PhysicsModel", "Model class used by WallGoCollisions")
        .def(py::init<const ModelDefinition&>())
        .def("updateParameter", static_cast<void(PhysicsModel::*)(const std::string&, double)>(&PhysicsModel::updateParameter), "Updates a symbolic parameter value.The symbol must have been defined at model creation time")
        .def("updateParameters", &PhysicsModel::updateParameters, "Updates model parameter values. The parameters must have been defined at model creation time")
        // Bind lambda wrapper that takes std::string instead of std::filesystem::path
        .def("readMatrixElements",
            [](PhysicsModel& self, const std::string& filePath, bool bPrintMatrixElements)
            {
                return(self.readMatrixElements(std::filesystem::path(filePath), bPrintMatrixElements));
            },
            R"(Read matrix elements from a file and stores them internally.
            This will only consider expressions where at least one currently registered out-of-equilibrium particle appears as an external particle.
            Note that this function clears any previously stored matrix elements for the model.
            Returns false if something goes wrong.)",
            py::arg("filePath"), py::arg("bPrintMatrixElements") = false
        )
        .def("createCollisionTensor",
            static_cast<CollisionTensor(PhysicsModel::*)(size_t)>(&PhysicsModel::createCollisionTensor),
            py::return_value_policy::take_ownership, // Python takes ownership, important for keeping observer registrations alive
            R"(Creates a CollisionTensor object that contains and manages collision integrals relevant for this model)",
            py::arg("basisSize")
        )
        .def("getNumObservers", &PhysicsModel::getNumObservers, "Returns number of objects that are registered as observers for this model");
}

} // namespace
