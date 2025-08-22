===========================================
FAQs
===========================================

.. contents::
    :local:
    :depth: 2


General
=======

- **How should I cite WallGo?**

    WallGo is free and open source, but if you use WallGo in your work, we ask that you
    support us by please citing the WallGo paper, `JHEP 04 (2025) 101 <https://doi.org/10.1007/JHEP04(2025)101>`_. The complete BibTex citation from `Inspire <https://inspirehep.net/literature/2846423>`_ is::

        @article{Ekstedt:2024fyq,
            author = "Ekstedt, Andreas and Gould, Oliver and Hirvonen, Joonas and Laurent, Benoit and Niemi, Lauri and Schicho, Philipp and van de Vis, Jorinde",
            title = "{How fast does the WallGo? A package for computing wall velocities in first-order phase transitions}",
            eprint = "2411.04970",
            archivePrefix = "arXiv",
            primaryClass = "hep-ph",
            reportNumber = "CERN-TH-2024-174, DESY-24-162, HIP-2024-21/TH",
            doi = "10.1007/JHEP04(2025)101",
            journal = "JHEP",
            volume = "04",
            pages = "101",
            year = "2025"
        }


Installation and running
============

- **I can not install WallGo.**

    Please take a look at our :doc:`installation instructions <installation>`. If it doesn't
    work for you, feel free to :doc:`send us an email <contact>`.

- **How do I run a first example?**

    After having installed WallGo, you can run one of the examples in the Models folder, e.g.
    
    .. code-block:: bash

        python Models/SingletStandardModel_Z2/singletStandardModelZ2.py
    
    A full run of the example file requires that the collisions files
    are located in the model folder as well.
    You can also generate the collision files yourself, e.g. (for a basis size of 5)
    
    .. code-block:: bash

        python Models/SingletStandardModel_Z2/singletStandardModelZ2.py --recalculateCollisions --momentumGridSize 5 

    This will create a folder called CollisionOutput_N5_UserGenerated containing the collision files.
    A grid size of 5 is typically too small for a full computation of the wall velocity, 
    but it does allow you to confirm that everything is installed correctly, 
    within a small amount of computation time.

Matrix elements
===============

- **I want to use a different set of matrix elements, is this possible?**

    Definitely! You can load your own matrix elements file. The default format is
    a JSON file with a specific structure, described in detail in the WallGo paper. 

- **Can I compute the matrix elements for my model using FeynRules, FeynArts and FeynCalc?**
    Yes, this works as an alternative to the WallGo MatrixElements pacakge, and in fact
    we used this to cross check our results. We have included an example in the repository
    for the `WallGoMatrix package <https://github.com/Wall-Go/WallGoMatrix>`_. Take
    a look at the directory `tests/FeynCalc`.

Collision integrals
===================

- **Can I parallelize the computation of the collision terms?**

    Yes! If you have `OpenMP <https://www.openmp.org/>`_ installed, by default the collision
    code is compiled with parallelisation. This allows processors with shared memory to work
    in parallel, so can make use of the multiple processors on your computer, or use up to
    one node on a supercomputer. To choose the number of threads, you need to set an
    environment variable, as follows

    .. code-block:: bash

        export OMP_NUM_THREADS=4

    Once done, if you then run the computation of the collision integrals, they will run with
    4 threads, which should speed up the computation by a factor of almost 4, if you have at
    least 4 physical cores.

- **Why doesn't the parallelisation work on my Mac?**

    Note that for Mac users, OpenMP can be a little more tricky to set up. We recommend using
    the Homebrew version, which requires an export statement to properly link,

    .. code-block:: bash

        brew install libomp
        export OpenMP_ROOT=$(brew --prefix)/opt/libomp

    The second line can be added to your `~/.zprofile` or `~/.zshrc` file so that it is called
    every time you open a terminal.

- **Can I reuse the same collision integrals for different models/parameter choices?**

    Yes, as long as your new model/parameter choice has the same interaction strength, 
    thermal masses (for the out-of-equilibrium particles) and momentum grid size as the model
    with which you obtained the collision integrals.

- **WallGo tells me that it can not read the collision files.**

    This might happen when you download the collision files from the git repository. 
    As the git repository uses Git Large File Storage (LFS) to manage the large collision files, the downloaded
    files will be pointers, and not the full collision files. To obtain the collision files requires Git LFS.

    Complete installation instructions for Git LFS can be found at `git-lfs.com <https://git-lfs.com/>`_. This depends on your operating system, but should be straightforward. For example, on Ubuntu you can use

    .. code-block:: bash

        sudo apt-get install git-lfs

    or on a Mac, you can use

    .. code-block:: bash

        brew install git-lfs
    
    Then, within the WallGo repository run
    
    .. code-block:: bash

        git lfs install
        git lfs fetch --all
    
    Alternatively, you can generate the collision files yourself.

Creating a model in Python
==========================

Model requirements
------------------

- **What is the parameter fieldCount?**

    This is the number of scalar background fields that your effective potential depends on and must be specified when
    subclassing EffectivePotential. It is used internally to reshape various helper arrays.

- **What is the msqVacuum in the Particle definition?**

    This is the field-dependent, vacuum (zero temperature) mass squared. The size of this quantity affects the strength of the 
    friction effect in the equation of motion of the scalar field, and the force that the particle feels from the wall. 
    Note that this parameter needs to be of the type Fields. If the particle is in equilibrium the type does not matter, and it
    msqVacuum can simply be set to zero.

- **What is the msqDerivative in the Particle definition?**

    This is the field-derivative of msqVacuum.
    Note that this parameter needs to be of the type Fields. If the particle is in equilibrium the type does not matter, and it
    msqVacuum can simply be set to zero.

- **How do I cound the totalDOFs in the Particle definition?**

    totalDOFs counts the total number of degrees of freedom for a particle species. This includes summing over e.g. spins and colors. 
    E.g. totalDOFs for the SM gluon would be 16. For a top quark with only SU(3) interactions totalDOFs would be 12,
    but if we distinguish left-handed and right-handed top quarks both would have totalDOFs = 6.

Effective potentials
--------------------

- **How can I check if implemented my potential correctly?**

    Assuming that you know what the critical temperature of your model is, you could cross-check if
    WallGo gives you the same. The critical temperature is not computed by default, but can be obtained
    from WallGoManager.thermodynamics.findCriticalTemperature( dT, rTol, paranoid), where dT is the 
    temperature step size, rTol the relative tolerance, and bool a setting for the phase tracing. The 
    latter two arguments are optional.

    Another cross-check is the position of the minimum at the provided nucleation temperature. 
    This can be checked with WallGoManager.model.effectivePotential.findLocalMinimum(phaseInput.phaseLocation, Tn),
    where phaseLocation is the approximate postion of the phase.

- **I want to describe the one-loop effective potential without high-temperature expansion. How do I include the thermal integrals in WallGo?**

    WallGo has predefined methods to compute the fermionic and bosonic one-loop
    sum-integrals. It also has a default table of precomputed values. These are
    located in the sub-package called PotentialTools, and can be imported as

    .. code-block:: python

        from WallGo import PotentialTools

    For a model using PotentialTools see the singlet scalar extension example.

- **My effective potential is complex, what should I do?**

    In a self-consistent calculation, the equations of motion for the scalar field/s
    should be real, and hence so the relevant effective potential should be real too. 
    Yet, computations of the effective potential can yield complex values. The same issue arises in the context of the bubble nucleation rate, and can resolved using
    effective field theory. :footcite:p:`Gould:2021ccf`

    By default WallGo requires a real effective potential, so it is up to the user to ensure this. For the one-loop PotentialTools sub-package of WallGo gives four
    different options for how to remove unwanted imaginary parts, listed in the
    enum :py:class:`WallGo.PotentialTools.EImaginaryOption`. See the docs for more
    details.

Free energy
-----------
- **I already know the value of the field and the effective potential as a function of temperature, can I provide these to WallGo to circumvent the phase tracing?**

    If the phase tracing does not work properly for your model, or if you want to speed up the
    initialization phase, you can provide arrays with the values of the field(s) in the minimum of the
    potential and the corresponding effective potential for the appropriate temperature range. 
    These are passed as a :py:class:`WallGo.FreeEnergyArrays` object, to the function
    :py:meth:`WallGo.WallGoManager.setupThermodynamicsHydrodynamics()`. These arrays are optional arguments;
    if they are not provided, WallGo will execute its default phase tracing algorithm.



Settings
========

- **Can I choose any value for the grid size?**

    No! The momentum-grid size has to be an ODD number. It should also be a large
    enough. We have found that 11, 13, ..., 21 are often sufficient, but larger
    grid sizes are needed when the model has a hierarchy of scales to resolve.


Running the Python code
=======================

Understanding the output
------------------------

- **Why does WallGo return a wall velocity of None?**

    You found a runaway wall. The included hydrodynamic backreaction and out-of-equilibrium friction effects are not sufficient
    to stop the wall from accelerating. Additional out-of-equilibrium particles might provide additional friction to obtain a
    static solution. Also note that a too small grid size could falsely suggest that the wall runs away. If the runaway behavior
    persists, your phase transition might be very strong. A proper computation of the wall velocity would require next-to-leading
    order contributions to the friction. These will be added to WallGo in the future.

- **Why does the hydrodynamic local thermal equilibrium velocity differ from the solution to the equation of motion?**

    The hydrodynamic solution in local thermal equilibrium and the solution to the equation of motion are not supposed to be
    exactly identical. The solution in the equation of motion relies on a Tanh-Ansatz. As a result, the equation of motion is
    not exactly satisfied, whereas the hydrodynamic solution is obtained under the assumption that this is the case. 

- **Why does the template model give me a terminal wall velocity, but the full hydrodynamics and the equation of motion do not?**

    The template model is an approximation of the full equation of state: it assumes that the sound speed is everywhere constant,
    and equal to the value at the nucleation temperature. Moreover: the plasma does not have a maximum or minimum temperature
    in the template model. In the full equation of state, there could be a maximum/minimum temperature due to the finite range of
    existence of the phases. This could limit the hydrodynamic backreaction effect, and as a result no terminal velocity can be found.

Warnings and errors
-------------------

- **Why does WallGo throw the error "Failed to solve Jouguet velocity at input temperature!"**

    WallGo can not solve the hydrodynamic matching condition to obtain the Jouguet velocity. 
    Please check your effective potential, and confirm that the thermodynamic quantities are reasonable 
    (alpha positive, the speeds of sound real and positive and the ratio of enthalpies smaller than 1). 
    Make sure that the field-independent contributions are also included in the effective potential 
    (e.g. the T^4 contribution from light fermions).
    Also make sure that you provided the WallGoManager with a temperature variation scale
    that was not too large, as this might prevent finding a correct tracing of (one of) the phases.

- **Why do I get the warning "Truncation error large, increase N or M"?**
    
    The accuracy of the solution to the Boltzmann equation and equations of motion increases with the grid size.
    WallGo will throw the warning "Truncation error large, increase N or M" when the estimated error on the solution of
    the out-of-equilibirum is large. This happens when the truncation error (obtained with John Boyd's Rule-of-thumb-2) is larger 
    than the finite-difference error *and* the truncation error is larger than the chosen error tolerance.

Parallelisation
---------------

- **I am running a scan. Can I parallelise the computation of the wall velocity with Python?**

    For a single parameter point, the Python part of WallGo does not parallelise
    simply. But, when running a scan, WallGo can be trivially parallelised, by sharing
    out the parameter points between processors.

Bugs
====

- **I think I found a bug in WallGo, what can I do?**

    Please create an issue on our `GitHub Issues page <https://github.com/Wall-Go/WallGo/issues>`_
    including sufficient detail that we can follow it up, ideally with a minimal
    example demonstrating the bug. Alternatively, :doc:`send us an email <contact>`
    and we will take a look at it. Please do check the FAQs and GitHub issues first,
    in case your bug has already been described.
