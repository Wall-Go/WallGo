#include "kinematics.h"
#include "FourVector.h"


//*****************Basic routines that work for generic four-vectors***********************



//*****************Specialized routines for our collision integrals*********************



// Creates s,t,y Mandelstam invariants
void CreateInvariants(double mandelstam[3],FourVector p,FourVector k, FourVector p2, FourVector k2){
    //help four-vector
    //The s invariant
    mandelstam[0] = 2.0*p*k;

    //The t invariant
    mandelstam[1] = -2.0*p*p2;

    //The u invariant
    mandelstam[2]=-mandelstam[0]-mandelstam[1];
}





