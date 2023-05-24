#include <math.h>


// The collElem class specifies which matrix element, the spin of the particles, and what
//delta_f terms the user wants to calculate
//The idea is that the user can create #collElem classes at the start which are all included in a given
// integration call

//For now I just hard code the thermal masses
#define  MQ2 0.251327 //Top quark mass squared
#define  MG2 3.01593 //Gluon mass squared
#define PI 3.14159265358979323846


#ifndef _FUNCS_H
#define _FUNCS_H
typedef double (*funcptr)(double, unsigned int n);
double (*boltzmannFunc)(double, double, double, double);


#define FDIM 1
#define NMAX 50000
#define MaxMomentum 20.0 //The momentum that we integrate up to

class collOperator {
	
	private:
	// Variables for the chebyshev polynomials
	unsigned int nZ=0;
	unsigned int mPerp=0;

	//Variable to choose which delta_f term should be computed
	// p:deltaF=0, k:deltaF=1, p2:deltaF=2, k2=deltaF=3
	int deltaF=0;

	//This function specifies the matrix element that the user wants to use
	double (*matrixFunc)(double, double, double); 



	typedef double (*distriFunc)(double);

	//delta_f factors
	distriFunc distriP;
	distriFunc distriK;
	distriFunc distriP2;
	distriFunc distriK2;



	FourVector FVp,FVk, FVp2, FVk2; //Four vectors that are used for the computation
	double mandelstam[3]={0,0,0}; //Mandelstam variables in the ordet s,t,u



	int numberElements; //Specifies how many matrix elements the user wants to include
	collElem* collTerm;


	public:

	double integrateCollision(double pVec[2], double preFac){
	//n specifies the p_z polynomial; m specifies the p_perp polynomial

	//pVec[0]=p and pVec[1]=theta_p
	//preFac contain coupling constants and factors of Pi

	//USERDATA is the input to the integration
	//Even though I declare it as double  USERDATA[1] and  USERDATA[2] are later cast as integers
	double USERDATA[2]={pVec[0],pVec[1]};//This contains all the information about p



	double xmin[5] = {0,0,0,0,0}, xmax[5] = {MaxMomentum,PI,PI,2*PI,2*PI}, val, err;


	hcubature(1, f, &USERDATA, 5, xmin, xmax, NMAX, 1e-6, 1e-6, ERROR_INDIVIDUAL, &val, &err);


	return preFac*val;
	}

	int f(unsigned ndim, const double *xx, void *fdata, unsigned fdim, double *fval){


	//pVec[0]=p and pVec[1]=theta_p
	double *pVec = (double *)fdata;
	double p = *(pVec);
	double thetaP = *(pVec+1);


	//I here use the conventions
	//xx[0] - k
	//xx[1] - theta_k
	//xx[2] - theta_p2
	//xx[3] - phi_k
	//xx[4] - phi_p2
	#define k xx[0]
	#define thetaK xx[1]
	#define thetaP2 xx[2]
	#define phiK xx[3]
	#define phiP2 xx[4]



	//First we create the four-vectors
	double kV[3]={k,thetaK,phiK};
	double p2V[3]={1.0,thetaP2,phiP2}; //e will fix the magnitude of the vector by a delta function
	double pV[3]={p,thetaP,0.0};


	FVk=FourVector(kV);
	FVp=FourVector(pV);
	FVp2=FourVector(p2V);


	//Fix the norm of p2 and create the k2 four-vector
	double norm=P2Norm(FVp,FVk,FVp2);
	FVp2.setLength(norm);

	//k2=p+k-p2
	FVk2=addFV(FVp,FVk);
	FVk2=subtractFV(FVk2,FVp2);



	//Jacobian factor from the k2^2=0 delta function
	double beta=abs(FVp2.energy()/SP4(FVp,FVk));
	CreateInvariants(mandelstam,FVp,FVk,FVp2, FVk2); //Calculates the mandelstam variables in the order s,t,u



	//The contribution from chebyshev polynomials
	double vecZ, vecPerp; //stores the z and perp component of the relevant vector



	double integralRes=0.0; //Stores the total value of the integral sumed over all user-defined terms
	//The full function
	double chebFac,matrixElem,boltzmannFac,kinFac;


	//We now include all the terms specified by the collTerm variable

	int *spin, matrixElemID, *deltaFID;
	collElem collHelp; //A help variable that is looped over


	//We now loop over all the different matrix elements
	// For each element we then add up all delta_f terms that the user specified
	for (int j = 0; j < numberElements; ++j) // numberElements specifies how many matrixElements the user wanted
	{

		collHelp=*(collTerm+j);

		//Set the spin of the particles
		spin=collHelp.printSpin();
		specifySpin(*(spin),*(spin+1),*(spin+2),*(spin+3));


		//Set the matrix element
		specifyMatrixElement(collHelp.printMatrixElem());


	// The matrixElem and kinFac factors do not depend on the delta_F flag
		matrixElem=matrixFunc(mandelstam[0],mandelstam[1],mandelstam[2]); //The matrix element
		kinFac=k*FVp2.energy()*sin(thetaK)*sin(thetaP2)*beta; //Kinematical factors from the integration measure


		//The full function
		//We now sum over all delta_f terms that the user wants
		deltaFID=collHelp.printDeltaF();
		for (int i = 0; i < 4; ++i)
		{
			if (*(deltaFID+i)<0)
			{
				break;
			}
			else{
				deltaF=*(deltaFID+i);
				specifyDeltaF(deltaF);
			}

			double debug;
			// p:deltaF=0, k:deltaF=1, p2:deltaF=2, k2=deltaF=3
				switch(deltaF) {
				  case 0:
				    vecZ=toTanh(FVp.zComp());
				    vecPerp=toExp(FVp.perpComp());
				    break;
				  case 1:
				    vecZ=toTanh(FVk.zComp());
				    vecPerp=toExp(FVk.perpComp());
				    break;
				  case 2:
				    vecZ=toTanh(FVp2.zComp());
				    vecPerp=toExp(FVp2.perpComp());
				    break;
				  case 3:
				    vecZ=toTanh(FVk2.zComp());
				    vecPerp=toExp(FVk2.perpComp());
				    break;
				}

				//The full function
			 chebFac=ChebFunc1(vecZ,nZ)*ChebFunc2(vecPerp,mPerp); //The Chebyshev factor
			 boltzmannFac=boltzmannFunc(FVp.energy(),FVk.energy(), FVp2.energy(), FVk2.energy()); //Distribution functions


			 integralRes+=chebFac*matrixElem*boltzmannFac*kinFac;

			 


			}
	}


	    fval[0] = integralRes;

	    return 0; // success*

	}


	void setNumberElements(int numberInput){
		numberElements=numberInput;
	}


	void specifyCollTerm(collElem* collElemInput){
		collTerm=collElemInput;
	}


	void specifyDeltaF(int n){
	//Function to choose which delta_f term should be computed
	// p:deltaF=0, k:deltaF=1, p2:deltaF=2, k2=deltaF=3

	switch(n) {
	  case 0:
	    boltzmannFunc=prefacP;
	    break;
	  case 1:
	    boltzmannFunc=prefacK;
	    break;
	  case 2:
	    boltzmannFunc=prefacP2;
	    break;
	  case 3:
	    boltzmannFunc=prefacK2;
	    break;
	}

		deltaF=n;
	}


	//The product of distribution functions that multiply delta f_p
	double prefacP(double p,double k, double p2, double k2){
		return distriK(k)*distriP2(p2)*distriK2(k2)*exp(k)/distriP(p);
	}

	//The product of distribution functions that multiply delta f_k
	double prefacK(double p,double k, double p2, double k2){
		return distriP2(p2)*distriK2(k2)*distriP(p)/distriK(k)*exp(p);
	}

	//The product of distribution functions that multiply delta f_p2
	double prefacP2(double p,double k, double p2, double k2){
		return -distriK2(k2)*distriP(p)*distriK(k)/distriP2(p2)*exp(k2);
	}


	//The product of distribution functions that multiply delta f_k2
	double prefacK2(double p,double k, double p2, double k2){
		return -distriP2(p2)*distriP(p)*distriK(k)/distriK2(k2)*exp(p2);
	}

	//Function to determine the spin-statistics of all particles
	void specifySpin(int nP,int nK,int nP2,int nK2){

		if (nP%2==0)
		{
			distriP=nB; //Bose-Einstein
		}
		else{
			distriP=nF; //Fermi-Dirac
		}


		if (nK%2==0)
		{
			distriK=nB; //Bose-Einstein
		}
		else{
			distriK=nF; //Fermi-Dirac
		}


		if (nP2%2==0)
		{
			distriP2=nB; //Bose-Einstein
		}
		else{
			distriP2=nF; //Fermi-Dirac
		}


		if (nK2%2==0)
		{
			distriK2=nB; //Bose-Einstein
		}
		else{
			distriK2=nF; //Fermi-Dirac
		}


	}


	void specifyMatrixElement(int n){
	//Function to choose which matrix element should be used
	// QQVV:0, QVQV:1, TQTQ: 2

	switch(n) {
	  case 0:
	    matrixFunc=matrixElementQQVV;
	    break;
	  case 1:
	    matrixFunc=matrixElementTQTQ;
	    break;
	 case 2:
	    matrixFunc=matrixElementTQTQ;
	    break;
	   
	}

		deltaF=n;
	}

	void specifyChebyshev(int n, int m){
	//l just specifies whether the function should be made trivial
	if(n>0&& m>0 ){
		ChebFunc2=Tn1;
			if (n%2==0)
			{
				ChebFunc1=Tn1;
			}else{
				ChebFunc1=Tn2;
			}
	}else{
		ChebFunc2=Tn3;
		ChebFunc1=Tn3;
	}		

	nZ=n;
	mPerp=m;


	}

};



#endif