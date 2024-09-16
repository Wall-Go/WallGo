===========================================
FAQs
===========================================

- **How come WallGo is so amazing?**

    Great question!

- **Why does WallGo return a wall velocity of 1?**

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

- **Why does WallGo throw the error "Failed to solve Jouguet velocity at input temperature!"**

    WallGo can not solve the hydrodynamic matching condition to obtain the Jouguet velocity. 
    Please check your effective potential, and confirm that the thermodynamic quantities are reasonable 
    (alpha positive, the speeds of sound real and positive and the ratio of enthalpies smaller than 1). 
    Make sure that the field-independent contributions are also included in the effective potential 
    (e.g. the T^4 contribution from light fermions).
    Also make sure that you provided the WallGoManager with a temperature scale
    that was not too large, as this might prevent finding a correct tracing of (one of) the phases.

- **How can I check if implemented my potential correctly?**

    Assuming that you know what the critical temperature of your model is, you could cross-check if
    WallGo gives you the same. The critical temperature is not computed by default, but can be obtained
    from WallGoManager.thermodynamics.findCriticalTemperature( dT, rTol, paranoid), where dT is the 
    temperature step size, rTol the relative tolerance, and bool a setting for the phase tracing. The 
    latter two arguments are optional.

    Another cross-check is the position of the minimum at the provided nucleation temperature. 
    This can be checked with WallGoManager.model.effectivePotential.findLocalMinimum(phaseInput.phaseLocation, Tn),
    where phaseLocation is the approximate postion of the phase.

- **Can I choose any value for the grid size?**

    No! The momentum-grid size has to be an ODD number. 

- **Why do I get the warning "Truncation error large, increase N or M"?**
    
    The accuracy of the solution to the Boltzmann equation and equations of motion increases with the grid size.
    WallGo will throw the warning "Truncation error large, increase N or M" when the estimated error on the solution of
    the out-of-equilibirum is large. This happens when the truncation error (obtained with John Boyd's Rule-of-thumb-2) is larger 
    than the finite-difference error *and* the truncation error is larger than the chosen error tolerance.

- **Can I reuse the same collision integrals for different models/parameter choices?**

    Yes, as long as your new model/parameter choice has the same interaction strength, 
    thermal masses (for the out-of-equilibrium particles) and momentum grid size as the model
    with which you obtained the collision integrals.

- **My effective potential is complex, what should I do?**

    To do

- **I want to compare to use a different set of matrix elements, is this possible?**

    Definitely! You can load your own matrix elements file. [Here we need we write what the requirements are].

- **I do not have a Mathematica license, can I still generate matrix elements?**

    [To do]

- **Can I parallelize the computation of the collision terms?**

    [To do]

- **I am running a scan. Can I parallelize the computation of the wall velocity with Python?**

    [To do]

- **I can not install WallGo.**

    [To do]

- **I think I found a bug in WallGo, what can I do?**

    [To do]
