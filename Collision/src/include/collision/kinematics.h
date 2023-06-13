#include "FourVector.h"


//Basic routines that work for generic four vectors

double SP3(FourVector a, FourVector b);	// Vector product of the spatial components

//Specialized routines for our collision integrals


// Calculates s,t,y Mandelstam invariants
void CreateInvariants(double mandelstam[3],FourVector p,FourVector k, FourVector p2, FourVector k2);
