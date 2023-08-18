# helper functions for BubbleDet
import findiff


def derivative(f, x, dx=1.0, n=1, order=4, scheme="center", args=None):
    r"""Computes numerical derivatives of a callable function.

    To replace scipy.misc.derivative which is to be deprecated.

    Based on the findiff package.

    Parameters
    ----------
    f : function
        The function to take derivatives of. It should take a float as its
        argument and return a float, potentially with other fixed arguments.
    x : float
        The position at which to evaluate the derivative.
    dx : float, optional
        The magnitude of finite differences.
    n : int, optional
        The number of derivatives to take, i.e. :math:`{\rm d}^nf/{\rm d}x^n`.
    order: int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}+1})`.
    scheme: {\"center\", \"forward\", \"backward\"}, optional
        Type of finite difference scheme.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the derivative of :py:data:`f` evaluated at :py:data:`x`.

    Examples
    --------
    >>> from BubbleDet.helpers import derivative
    >>> def f(x):
    >>>     return x ** 4
    >>> derivative(f, 1, dx=0.01)
    4.000000000000011

    """
    if args is None:
        fA = f
    else:
        fA = lambda xx: f(xx, *args)
    coeffs = findiff.coefficients(deriv=n, acc=order)[scheme]
    n_coeffs = len(coeffs["coefficients"])
    res = 0
    for i in range(n_coeffs):
        coeff = coeffs["coefficients"][i]
        offset = coeffs["offsets"][i]
        c_i = coeff / dx ** n
        x_i = x + offset * dx
        res += c_i * fA(x_i)
    return res
