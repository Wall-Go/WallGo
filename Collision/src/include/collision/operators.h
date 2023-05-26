#include <math.h>
#include "CollElem.h"

//Functions for setting global variables
void specifyChebyshev(int n, int m); //Specifies which chebyshev polynomials should be used
void setNumberElements(int numberInput); //Specifies how many different matrix elements the user want to use

void specifyCollTerm(CollElem* collElemInput);//Specification for the matrix element, spin, and delta_f terms that should be used



//Integrating the collision operator
double integrateCollision(double pVec[2], double preFac); //pVec[0]=p and pVec[1]=theta_p,preFac contain coupling constants and factors of Pi