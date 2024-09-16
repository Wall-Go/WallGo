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



