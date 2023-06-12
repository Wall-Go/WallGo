/*******/

#include <iostream>
#include <chrono>
using namespace std::chrono;
using namespace std;
#include <math.h>
#include <sys/time.h>
#include <time.h>
#include "kinematics.h"
#include "operators.h"
#include "CollElem.h"

#include "hdf5Interface.h"




#define PI 3.14159265358979323846
#define GS 1.2279920495357861




void printTime(long duration){
     //3600000000 microseconds in an hour
     long hr = duration / 3600000000;
     duration = duration - 3600000000 * hr;
     //60000000 microseconds in a minute
     long min = duration / 60000000;
     duration = duration - 60000000 * min;

     //1000000 microseconds in a second
     long sec = duration / 1000000;

     cout << hr << " hours and " << min << " minutes and " << sec << " seconds" << endl;

}




//***************

void calculateAllCollisions() {

     /****************************************************/

     //The numerical prefactor that depends on coupling constants and factors of pi
     double prefac=-64*GS*GS*GS*GS/9/8/(2*PI)/(2*PI)/(2*PI)/(2*PI)/(2*PI);   //All factors of Pi and couplings that multiply the integral. Should be changed
     double intVal=0.0;                                                      //The value of the integral

     double pVec[2]={1.0,PI*0.5}; //pVec[0] is the magnitude of p, and pVec[1] is the polar angle relative to the z-axis


     int nElements=1;                  //This specifies that we want one matrix elements
     CollElem collTermTot[nElements];  //For each matrix element the spin and needed delta_F terms are stored in the class collElem
     setNumberElements(nElements);     //Sets the global variable that keeps tracks of how many elements we have

     //For all particles we use the convention that the process is p+k->p2+k2, where p is the external momenta
     //that should be evaluated on the grid
     //All specification of spin and matrix elements assumes the order p k p2 k2, so {1,1,2,2} etc 


     //To specify the spin we use the conventions (should probably be changed) that even numbers correspond to fermions
     //and odd number to bosons


     CollElem collTerm;            //Creates a matrix element
     int spin[4]={1,1,2,2};        //Specifies that p and k are fermions, and that p2 and k2 are bosons
     int deltaF[4]={1,1,0,0};      //Specifies that we only want the delta_p and delta_k terms

     collTerm.setSpin(spin);       //Sets the spin
     collTerm.setMatrixElem(0);    //Specifies that we want the tt->gg matrix element. Hard coded for now. See CollElem for all available matrix elements
     collTerm.setDeltaF(deltaF);   
     collTerm.setDistributions();  //Initializes Bose-Einstein and Fermi-Dirac distributions


     collTermTot[0]=collTerm;      //Loads all requested matrix elements. In this case only one
     specifyCollTerm(collTermTot);


     //For the Chebyshev polynomials we use the convention: T_nZ(rho_z)T_mPerp(rho_perp) (with rho_z=tanh(pz/2) and rho_perp=1-exp(-p_perp))
     specifyChebyshev(2,2);        //Specifies that we are interested in nZ=2 and mPerp=2.


     intVal=integrateCollision(pVec,prefac);
     cout<< intVal<<endl; //You should get 0.433736 with the inputs as above


     // /****************Performance tests******************/

     /*Note in practice we can do way faster than the estimate below.
     Both from increasing the number of cores, and by using symmetry,
     and by using that deltaF_p terms only need to be evaluated once*/

     srand (static_cast <unsigned> (time(0)));

     //     // Get starting timepoint
     auto start = high_resolution_clock::now();
     auto stop = high_resolution_clock::now();
     auto duration = duration_cast<microseconds>(stop - start);

     double r;

     start = high_resolution_clock::now();
     for (int i = 0; i < 100; ++i)
     {
          r = (double)rand() / RAND_MAX;
     pVec[0]= 0.0 + r * 20.0; //Just some random values for the p vector
     pVec[1]= 0.0 + r * PI;   //Just some random values for the p vector
     intVal=integrateCollision(pVec,prefac);
     }

     stop = high_resolution_clock::now();

     duration = duration_cast<microseconds>(stop - start);

     cout<<"Estimated time to generate the entire grid for 1 thread:"<<endl;
     printTime(pow(20,4)*duration.count()/100);



     /*******************If you want to generate the entire grid you can do something like this***************************/

     int gridSizeN = 20;
     int Ncheb = gridSizeN - 1; //(N-1)^4 terms in total

     //The tensor is allocated as collGrid[nZ,mPerp,iZ,jPerp]
     Array4D collGrid(gridSizeN-1, gridSizeN-1, gridSizeN-1, gridSizeN-1, 0.0);

     double rhoZ[Ncheb], rhoPerp[Ncheb]; //Contains the Chebyshev interpolation points for pZ and pPerp


     //All the interpolation points for pZ
     for (int i = 1; i < Ncheb; ++i)
     {
          rhoZ[i-1]=2.0*atanh(-cos(PI*i/(Ncheb+0.0)));
     }


     //All the interpolation points for pPerp
     for (int j = 0; j < Ncheb-1; ++j)
     {
          rhoPerp[j]=-log((1.0+cos(PI*j/(Ncheb-1.0)))/2.0);
     }

     // We now do the for loop over all the grid points

     std::cout << "Now computing all collision integrals. See you in a bit...\n"; 
     //std::cout << "TEST VERSION: Generating dummy collision integrals\n";

     for (int m = 0; m < Ncheb; ++m) {

          for (int n = 0; n < Ncheb; ++n) {
               specifyChebyshev(m+2,n+1);    //specifies the relevant chebyshev point
                                             //note that n is between 2 and nZ
                                             //, and m is between 1 and mPerp-1
               for (int j = 0; j < Ncheb; ++j) {
                    pVec[0]=rhoZ[j];

                    for (int k = 0; k < Ncheb; ++k) {
                         pVec[1]=rhoPerp[k]; 
                         
                         collGrid[m][n][j][k]=integrateCollision(pVec,prefac);

                         printf("%d %d %d %d %g\n", n, m, j, k, collGrid[m][n][j][k]);
                         
                         // Dummy 
                         collGrid[m][n][j][k]= 0.0;
                    }
               }
          }
     }

     // Write these to file
     std::string filename = "collisions_Chebyshev_" + std::to_string(gridSizeN) + ".hdf5";
     WriteToHDF5(collGrid, filename, "top");
}



int main(int argc, char const *argv[]) {

     calculateAllCollisions();

     return 0;
}
