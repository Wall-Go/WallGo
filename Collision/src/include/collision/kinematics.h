#include "FourVector.cpp"


//Basic routines that work for generic four vectors

double SP4(FourVector a, FourVector b);	// Scalar product of two four-vector
double SP3(FourVector a, FourVector b);	// Vector product of the spatial components
double setLength(double Length); //Sets the length of the spatial vector
FourVector addFV(FourVector a, FourVector b); //Adds two four-vectors
FourVector subtractFV(FourVector a, FourVector b); // Subtracts the four-vector b from a. So vec=a-b


//Specialized routines for our collision integrals

// Calculates the norm for the spatial p2 vector using k2^2=(p+k-p2)^2, where all k2,p,k,p2 are four-vectors
double P2Norm(FourVector k,FourVector p, FourVector p2);

// Calculates s,t,y Mandelstam invariants
void CreateInvariants(double mandelstam[3],FourVector p,FourVector k, FourVector p2, FourVector k2);

