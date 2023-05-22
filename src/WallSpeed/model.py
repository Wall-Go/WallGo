import numpy as np
import math
import warnings
#import DRwizard
import FiniteT


class Model:
    '''
    Class that generates the model given external model file
    '''

    def __init__(self):
        '''
        Initialise class
        '''

    def J3(self,m):
        '''
        Log integral for 1loop 3d Veff
        '''
        return -m**3/(12*np.pi)



    def Vtree(self,X,show_V=False):
        X = np.asanyarray(X)
        h,s = X[...,0],X[...,1]
        vtree = (
            +self.muh2*h**2/2
            +self.mus2*s**2/2
            +self.lambdaH*h**4/4
            +self.lambdaS*s**4/4
            +self.lambdaHS*(h*s)**2/4
            +self.CTs[0]
            +self.CTs[1]*h**2
            +self.CTs[2]*s**2
            +self.CTs[3]*h**4)
        if show_V:
            print(vtree)
        return vtree


    def Veff4d(self):
        '''
        Thermal effective potential at 1 loop in 4d
        '''

    def Veff3d(self):
        '''
        Thermal effective potential at 1 loop at the supersoft scale
        '''

    def dVdT(self,X,T,include_radiation=True):
        '''
        1st T-derivative of the effective potential
        '''

    def d2VdT2(self,X,T,include_radiation=True):
        '''
        2nd T-derivative of the effective potential
        '''

    def d3VdT3(self,X,T,include_radiation=True):
        '''
        3rd T-derivative of the effective potential
        '''

    def Vefftot(self):
        '''
        Function that generates the effective potential given a pre-defined model
        '''


def main():
    pot = Veff
    print(pot.Vtree());
