# -*- coding: utf-8 -*-
"""
Model: Standard Model + singlet scalar

Ref:
"""


"""
Model definition
"""
import numpy as np # arrays, maths and stuff
import math
import WallSpeed
from WallSpeed import Particle 

class xSM(WallSpeed.GenericModel):
    def __init__(self,
        mu4D,
        mus,lams,lamm,
        use_EFT=False,
        ):
        r"""Initialisation

        Parameters
        ----------
        mu4D : float 
            4d renormalisaton scale.
        use_EFT : bool
            True if 3d EFT is used.

        Returns
        -------
        cls : Model 
            An object of the Model class.
        """
        self.use_EFT=use_EFT

        self.mus = mus
        self.lams = lams
        self.lamm = lamm

        self.v0 = 246.
        self.muh = 125.
        self.lamh = self.muh**2/(2*self.v0**2)
        self.muhsq = -self.lamh*self.v0**2
        self.mussq = +self.mus**2-self.lamm*self.v0**2/2

        self.musq = [self.muhsq, self.mussq]

        """
        Number of bosonic and fermionic dofs
        """
        self.num_boson_dof = 29
        self.num_fermion_dof = 90
        """
        Number of fermion generations and colors
        """
        self.nf = 3
        self.Nc = 3
        """
        4D RG scale of EFT as fraction of temperature
        """
        self.mu4D = mu4D
        self.mu4Dsq = mu4D*mu4D
        """
        Z,W,t mass, strong gauge coupling and fermion generations
        """
        self.MW = 80.379
        self.MZ = 91.1876
        self.Mt = 173.

        self.g0 = 2*self.MW/self.v0
        self.g1 = self.g0*math.sqrt((self.MZ/self.MW)**2-1)
        self.g2 = self.g0
        self.g3 = 1.2279920495357861
        self.yt = math.sqrt(1/2)*self.g0*self.Mt/self.MW

        self.musT = (
                +1./6*lamm
                +1./4*lams)
        self.muhT = (
                (
                +1*self.g1**2
                +3*self.g2**2
                +4./3*self.Nc*self.yt**2
                +8*self.lamh)/16
                +self.lamm/24
                )

        self.musqT = [self.muhT, self.musT]


    def modelParameters(self) -> dict[str, float]:
        params = {
            'muhsq': self.muhsq,
            'mussq': self.mussq,
            'lamh': self.lamh,
            'lams': self.lams,
            'lamm': self.lamm,
            'muhT': self.muhT,
            'musT': self.musT,
            'g1': self.g1,
            'g2': self.g2,
            'g3': self.g3,
            'yt': self.yt
        }
        return params

    def particles(self) -> np.ndarray[Particle]:
        mh2 = lambda fields: self.muhsq+3*self.lamh*fields[0,...]**2+self.lamm*fields[1,...]**2/2
        ms2 = lambda fields: self.mussq+3*self.lams*fields[1,...]**2+self.lamm*fields[0,...]**2/2
        mhs2 = lambda fields: self.lamm*fields[0,...]*fields[1,...]
        sqrt = lambda fields: np.sqrt((mh2(fields)-ms2(fields))**2+4*mhs2(fields)**2)
        m1 = lambda fields: (mh2(fields)+ms2(fields))/2+sqrt(fields)/2
        m2 = lambda fields: (mh2(fields)+ms2(fields))/2-sqrt(fields)/2

        higgs = WallSpeed.Particle(
            "higgs",
            msqVacuum=m1,
            msqThermal=lambda T: self.muhT * T**2,
            statistics="Boson",
            degreesOfFreedom=1,
            coefficientCW=3/2,
            inEquilibrium=True,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        singlet = WallSpeed.Particle(
            "singlet",
            msqVacuum=m2,
            msqThermal=lambda T: self.musT * T**2,
            statistics="Boson",
            degreesOfFreedom=1,
            coefficientCW=3/2,
            inEquilibrium=True,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        chi = WallSpeed.Particle(
            "chi",
            msqVacuum=lambda fields: self.muhsq + self.lamh*fields[0,...]**2+self.lamm*fields[1,...]**2/2,
            msqThermal=lambda T: self.musT * T**2,
            statistics="Boson",
            degreesOfFreedom=3,
            coefficientCW=3/2,
            inEquilibrium=True,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        w = WallSpeed.Particle(
            "w",
            msqVacuum=lambda fields: (self.g2**2)*fields[0,...]**2/4,
            msqThermal=lambda T: self.musT * T**2,
            statistics="Boson",
            degreesOfFreedom=6,
            coefficientCW=5/6,
            inEquilibrium=True,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        z = WallSpeed.Particle(
            "z",
            msqVacuum=lambda fields: (self.g1**2+self.g2**2)*fields[0,...]**2/4,
            msqThermal=lambda T: self.musT * T**2,
            statistics="Boson",
            degreesOfFreedom=3,
            coefficientCW=5/6,
            inEquilibrium=True,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        top = WallSpeed.Particle(
            "top",
            msqVacuum=lambda fields: self.yt**2 * np.asanyarray(fields)[0,...]**2/2,
            msqThermal=lambda T: 0.251327 * T**2,
            statistics="Fermion",
            degreesOfFreedom=12,
            coefficientCW=3/2,
            inEquilibrium=False,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        return [higgs,singlet,chi,w,z,top]

    def outOfEquilibriumParticles(self) -> np.ndarray[Particle]:
        top = WallSpeed.Particle(
            "top",
            msqVacuum=lambda fields: self.yt**2 * np.asanyarray(fields)[0,...]**2/2,
            msqThermal=lambda T: 0.251327 * T**2,
            statistics="Fermion",
            degreesOfFreedom=12,
            coefficientCW=3/2,
            inEquilibrium=False,
            ultrarelativistic=False,
            collisionPrefactors=[self.g3**4, self.g3**4, self.g3**4],
        )
        return [top]

    def treeLevelVeff(self, fields: np.ndarray[float], T, show_V=False):
        """
        Tree-level effective potential

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        T : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `fields` and `T`
            should be such that ``fields.shape[:-1]`` and ``T.shape`` are
            broadcastable (that is, ``fields[...,0]*T`` is a valid operation).

        Returns
        -------
        V: tree-level effective potential 
        """
        fields = np.asanyarray(fields)

        if self.use_EFT:
            for i in range(len(self.musq)):
                self.musq[0] += T**2*self.musqT[0]
            lamh = self.lamh*T
            lams = self.lams*T
            lamm = self.lamm*T
        else:
            musq = self.musq
            lamh = self.lamh
            lams = self.lams
            lamm = self.lamm

        h1,s1 = fields[0,...],fields[1,...]
        V = (
            +1/2*musq[0]*h1**2
            +1/2*musq[1]*s1**2
            +1/4*lamh*h1**4
            +1/4*lams*s1**4
            +1/4*lamm*(h1*s1)**2)
        if show_V:
            print(V)
        return V