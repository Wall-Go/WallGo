"""
Class for the one-loop effective potential without high-temperature expansion
"""

from abc import ABC, abstractmethod
import numpy as np
import numpy.typing as npt

from WallGo import EffectivePotential
from WallGo import Integrals


class EffectivePotentialNoResum(EffectivePotential, ABC):
    """Class EffectivePotential_NoResum -- Specialization of the abstract
    EffectivePotential class that implements common functions for computing
    the 1-loop potential at finite temperature, without any
    assumptions regarding the temperature (no high- or low-T approximations).
    In some literature this would be the ``4D effective potential''.

    """

    integrals: Integrals

    def __init__(
        self,
        modelParameters: dict[str, float],
        fieldCount: int,
        integrals: Integrals = None,
    ):
        ##
        super().__init__(modelParameters, fieldCount)

        ## Use the passed Integrals object if provided,
        ## otherwise create a new one with default settings
        if integrals:
            self.integrals = integrals
        else:
            self.integrals = Integrals()

    @abstractmethod
    def bosonStuff(self, fields, temperature):
        """
        Calculate the boson particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        temperature : float or array_like
            The temperature at which to calculate the boson masses. Can be used
            for including thermal mass corrrections. The shapes of `fields` and
            `temperature` should be such that ``fields.shape[:-1]`` and 
            ``temperature.shape`` are broadcastable 
            (that is, ``fields[0,...]*T`` is a valid operation).

        Returns
        -------
        massSq : array_like
            A list of the boson particle masses at each input point `X`. The
            shape should be such that
            ``massSq.shape == (X[...,0]*T).shape + (Nbosons,)``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degreesOfFreedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c : float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        rgScale : float or array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Typically, one
            takes the same rgScale for all particles, but different scales
            for each particle are possible.
        """

    @abstractmethod
    def fermionStuff(self, fields, temperature):
        """
        Calculate the fermion particle spectrum. Should be overridden by
        subclasses.

        Parameters
        ----------
        fields : array_like
            Field value(s).
            Either a single point (with length `Ndim`), or an array of points.
        temperature : float or array_like

        Returns
        -------
        massSq : array_like
            A list of the fermion particle masses at each input point `field`. The
            shape should be such that  ``massSq.shape == (field[...,0]).shape``.
            That is, the particle index is the *last* index in the output array
            if the input array(s) are multidimensional.
        degreesOfFreedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            len
        c : float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 3/2` for all fermions.
        rgScale : float or array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Typically, one
            takes the same rgScale for all particles, but different scales
            for each particle are possible.
        """

    @staticmethod
    def jCW(massSq: float, degreesOfFreedom: int, c: float, rgScale: float):
        """
        Coleman-Weinberg potential

        Parameters
        ----------
        msq : array_like
            A list of the boson particle masses at each input point `X`.
        degreesOfFreedom : float or array_like
            The number of degrees of freedom for each particle. If an array
            (i.e., different particles have different d.o.f.), it should have
            length `Ndim`.
        c: float or array_like
            A constant used in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Generally
            `c = 1/2` for gauge boson transverse modes, and `c = 3/2` for all
            other bosons.
        rgScale : float or array_like
            Renormalization scale in the one-loop zero-temperature effective
            potential. If an array, it should have length `Ndim`. Typically, one
            takes the same rgScale for all particles, but different scales
            for each particle are possible.

        Returns
        -------
        jCW : float or array_like
            One-loop Coleman-Weinberg potential for given particle spectrum.
        """
        # do we want to take abs of the mass??
        return (
            degreesOfFreedom
            * massSq
            * massSq
            * (np.log(np.abs(massSq / rgScale**2) + 1e-100) - c)
        )

    def potentialOneLoop(self, bosons, fermions, checkForImaginary: bool = False):
        """
        One-loop corrections to the zero-temperature effective potential
        in dimensional regularization.

        Parameters
        ----------
        bosons : array of floats
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : array of floats
            fermionic particle spectrum (here: masses, number of dofs)
        RGscale: float
            RG scale of the effective potential

        Returns
        -------
        potential : float
        """

        ## LN: should the return value actually be complex in general?

        massSq, nb, c, rgScale = bosons
        potential = np.sum(self.jCW(massSq, nb, c, rgScale), axis=-1)

        massSq, nf, c, rgScale = fermions
        potential -= np.sum(self.jCW(massSq, nf, c, rgScale), axis=-1)

        if checkForImaginary and np.any(massSq < 0):
            try:
                potentialImag = potential.imag / (64 * np.pi * np.pi)[np.any(massSq < 0,
                                                                             axis=0)]
            except IndexError:
                potentialImag = potential.imag / (64 * np.pi * np.pi)
            print(f"Im(potentialOneLoop)={potentialImag}")

        return potential / (64 * np.pi * np.pi)

    def potentialOneLoopThermal(
        self,
        bosons,
        fermions,
        temperature: npt.ArrayLike,
        checkForImaginary: bool = False,
    ):
        """
        One-loop thermal correction to the effective potential without any 
        temperature expansions.

        Parameters
        ----------
        bosons : ArrayLike
            bosonic particle spectrum (here: masses, number of dofs, ci)
        fermions : ArrayLike
            fermionic particle spectrum (here: masses, number of dofs)
        temperature: ArrayLike

        Returns
        -------
        potential : 4d 1loop thermal potential
        """

        ## m2 is shape (len(T), 5), so to divide by T we need to transpose T,
        ## or add new axis in this case.
        ## But make sure we don't modify the input temperature array here.
        temperature = np.asanyarray(temperature)

        temperatureSq = temperature * temperature + 1e-100

        ## Need reshaping mess for numpy broadcasting to work
        if temperatureSq.ndim > 0:
            temperatureSq = temperatureSq[:, np.newaxis]

        ## Jb, Jf take (mass/T)^2 as input, np.array is OK.
        ## Do note that for negative m^2 the integrals become wild and convergence
        ## is both slow and bad, so you may want to consider taking the absolute
        ## value of m^2. We will not enforce this however

        ## Careful with the sum, it needs to be column-wise.
        ## Otherwise things go horribly wrong with array T input.
        ## TODO really not a fan of hardcoded axis index

        massSq, nb, _, _ = bosons
        potential = np.sum(nb * self.integrals.Jb(massSq / temperatureSq), axis=-1)

        massSq, nf, _, _ = fermions
        potential += np.sum(nf * self.integrals.Jf(massSq / temperatureSq), axis=-1)

        if checkForImaginary and np.any(massSq < 0):
            try:
                potentialImag = potential.imag * temperature**4 / (2 * np.pi * np.pi)[
                    np.any(massSq < 0, axis=-1)]
            except IndexError:
                potentialImag = potential.imag * temperature**4 / (2 * np.pi * np.pi)
            print(f"Im(V1T)={potentialImag}")

        return potential * temperature**4 / (2 * np.pi * np.pi)
