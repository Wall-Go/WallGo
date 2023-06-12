#include "operators.h"
#include "kinematics.h"
#include <stdio.h>
#include <stdlib.h>
#include <cfloat>
#include <cmath>
#include <gsl/gsl_math.h>
#include <gsl/gsl_monte.h>
#include <gsl/gsl_monte_plain.h>
#include <gsl/gsl_monte_miser.h>
#include <gsl/gsl_monte_vegas.h>
#include <iostream>

#include <FourVector.h>

using namespace std;

//For now I just hard code the thermal masses
#define PI 3.14159265358979323846

//*********************Global variables************************(Yes i know... please don't send a raptor after me!)
unsigned int nZ=0; //The index for the p_z chebyshev polynomial
unsigned int mPerp=0; //The index for the p_perp chebyshev polynomial


//**************Global options****************
void specifyChebyshev(int n, int m){
	//Specifies which chebyshev polynomials should be used
			nZ=n;
			mPerp=m;
}

/***********Specification for the matrix element, spin, and delta_f terms***********/

int numberElements; //Specifies how many matrix elements the user wants to include
CollElem* collTerm; //Each element in collTerm contains one matrix element, spin, and specifies which delta_f are included


void setNumberElements(int numberInput){
	numberElements=numberInput;
}


void specifyCollTerm(CollElem* collElemInput){
	collTerm=collElemInput;
}



/********Distribution functions and variable transformations**********/


//Variable transformation for the z-component that is used as inputs for chebyshev polynomials
static inline double toTanh(double kz) {
  return tanh(kz/2);
}

//Variable transformation for the perp-component that is used as inputs for chebyshev polynomials
static inline double toExp(double kperp) {
  return 1-2*exp(-kperp);
}


//**************Chebyshev functions****************
double chebFunc(double vecZ, double vecPerp){
//For the input it is assumed that tanh(#/2) and 1-2*exp(-#) have already been applied to vecZ and vecPerp
		if (nZ>0&&mPerp>0)
		{
			if (nZ%2==0) //We use a different polynomial depending on if nZ is even or odd
			{
				return (cos(nZ*acos(vecZ))-1.0)*(cos(mPerp*acos(vecPerp))-1.0);
			}else{

				return (cos(nZ*acos(vecZ))-vecZ)*(cos(mPerp*acos(vecPerp))-1.0);
			}
			
		}else{

				return 1.0; //By default we return 1. For example if nZ or mPerp is negative the chebyshev term is not included
		}

}


/**********************products of Boltzmann factors and chebyshev polynomials*************/

double boltzDeltaFTot(double *boltzFac, double *chebFac){
//Adds up all requested terms for a specific matrix element
//boltzFac is a 4-d array that contains the contribution from distribution functions
//chebFac is a 4-d array that contains the contribution from chebyshev polynomails

	double res=0.0;
	for (int i = 0; i < 4; ++i)
	{
		res+=(*(boltzFac+i))*(*(chebFac+i));
	}
	
	return res;
}


double* chebFactors(int* deltaInd,FourVector *FV){
//deltaInd specifies which deltaF terms should be calculated.
//Each term involves the same chebyshev polynomial, but the arguement differs
//FV contains all four vectors: FV=(FVp,FVk,FVp2,FVk2)

	static double chebRes[4]={0.0,0.0,0.0,0.0};//We return zero unless otherwise specified
	for (int i = 0; i < 4; ++i)
	{
		if (*(deltaInd+i)>0)
		{
			chebRes[i]=chebFunc(toTanh(FV[i].zComp()),toExp(FV[i].perpComp()));
		}
	}

 return chebRes;
}


//Note that we do not include the overall factor depending on pi and coupling constants here

double f(double *xx, size_t dim, void *fdata)
{
	  (void)(dim); /* avoid unused parameter warnings */

		FourVector FVp=*(FourVector*)fdata; //Four vector for the incoming momenta, p


		//Note to self: Change everything so that we integrate over Cos(angles)
		double k=xx[0];				//xx[0] - k
		double thetaK=xx[1];	//xx[1] - theta_k
		double thetaP2=xx[2];	//xx[2] - theta_p2
		double phiK=xx[3];		//xx[3] - phi_k
		double phiP2=xx[4];		//xx[4] - phi_p2


		FourVector FVk, FVp2, FVk2; //Four vectors that are used for the computation
		double mandelstam[3]={0,0,0}; //Mandelstam variables in the order s,t,u


		// TODO check that these are correct
		FVk = FourVector(k, k*sin(thetaK)*sin(phiK), k*sin(thetaK)*cos(phiK), k*cos(thetaK));
		FVp2 = FourVector(1.0, sin(thetaP2)*sin(phiP2), sin(thetaP2)*cos(phiP2), cos(thetaP2)); // we will fix the magnitude of the vector by a delta function


		// Now we fix the norm of p2
		double norm = FVp*FVk / (FVp*FVp2 + FVp2*FVk);
		FVp2 *= norm;

		// Now we use momentum conservation,k2=p+k-p2, and create the k2 four-vector
		FVk2 = FVp + FVk - FVp2;

		//Jacobian factor from the k2^2=0 delta function
		double beta=FVp2.energy() / (FVp*FVk);

		CreateInvariants(mandelstam,FVp,FVk,FVp2, FVk2); //Calculates the mandelstam variables in the order s,t,u




		//The contribution from chebyshev polynomials
		double vecZ, vecPerp; //stores the z and perp component of the relevant vector
		double integralRes=0.0; //Stores the total value of the integral sumed over all user-defined terms


		//The full function
		double chebBoltzmannRes,matrixElem,boltzmannFac,kinFac;//The full integrand is the product of these factors


		
		int *deltaFID;//A help variables that specifies, for each matric element, which deltaF terms should be computed
		CollElem collHelp; //A help variable that is used to sum over all user-defined matrix elements

		FourVector FVTot[4]={FVp,FVk,FVp2,FVk2};//This can be made neater


		double momenta[4]={FVp.energy(),FVk.energy(), FVp2.energy(), FVk2.energy()}; //All the energies of the particles
		double boltzFac[4]={0.0,0.0,0.0,0.0}; //Contains all contributions from Boltzmann factors, the statistics of these are specified by a
																					// a collFunc variable
		

		double *chebFac; //Contain the contributions from the chebyshev polynomials

		
		kinFac=k*FVp2.energy()*beta; //Kinematical factors from the integration measure


		for (int j = 0; j < numberElements; ++j) // numberElements specifies how many matrixElements the user wanted
			{

				collHelp=*(collTerm+j); //We now loop over all the matrix elements
																//For each element we specify the spin, matrix element, and what delta_f terms
																//should be used.

				collHelp.specifyMatrixElement();  //We now specify the matrix element
				matrixElem=collHelp.calculateMatrixElement(mandelstam[0],mandelstam[1],mandelstam[2]); //The value of the matrix element


				//We now sum over all delta_f terms that the user wants
				deltaFID=collHelp.printDeltaF();
				chebFac=chebFactors(deltaFID,FVTot);


				boltzFac[0]=collHelp.calculateDeltaP(momenta[0],momenta[1],momenta[2],momenta[3]);
				boltzFac[1]=collHelp.calculateDeltaK(momenta[0],momenta[1],momenta[2],momenta[3]);
				boltzFac[2]=collHelp.calculateDeltaP2(momenta[0],momenta[1],momenta[2],momenta[3]);
				boltzFac[3]=collHelp.calculateDeltaK2(momenta[0],momenta[1],momenta[2],momenta[3]);


			 chebBoltzmannRes=boltzDeltaFTot(boltzFac,chebFac);//We now sum up the total contribution from all delta_f terms

			 integralRes+=kinFac*chebBoltzmannRes*matrixElem; //The total contribution for a given matrix element
			}

    return integralRes;

}

/*************************The actual integration****************/

/*********************Definitions used by the integration library*******************************/

#define MaxMomentum 20.0 //The momentum that we integrate up to



double integrateCollision(double pVec[2], double preFac){

	// pVec[0] = p1_Z, pVec[1] = p1_parallel

	double res, err;

	double xl[5] = { 0, -1.0, -1.0 ,0,0};
	double xu[5] = { MaxMomentum,1.0,1.0, 2*PI,2*PI };	//We integrate the Cos(polar angle) from -1 to 1


	//double pV[3]={pVec[0],cos(pVec[1]),0.0};

	double energy1 = std::sqrt(pVec[0]*pVec[0] + pVec[1]*pVec[1]);
	FourVector FVp= FourVector(energy1, 0.0, pVec[1], pVec[0]);


	const gsl_rng_type *T;
	gsl_rng *r;
	gsl_monte_function G = { &f, 5, &FVp };			//f is the integrand that takes pData=pVec as an input


	size_t calls = 10000;	//The number of monte-carlo points

	gsl_rng_env_setup ();

	T = gsl_rng_default;
	r = gsl_rng_alloc (T);


	gsl_monte_vegas_state *s = gsl_monte_vegas_alloc (5);

	gsl_monte_vegas_integrate (&G, xl, xu, 5, calls, r, s,
							&res, &err);

	gsl_monte_vegas_free (s);
	gsl_rng_free (r);


	//  printf("%f %f\n",res, 100*err/res ); //prints the result and relative error(in%)

	return res*preFac;
}








