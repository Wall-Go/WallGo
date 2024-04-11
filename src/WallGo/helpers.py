import numpy as np

FIRST_DERIV_COEFF = {'2': np.array([[-0.5,0.5],
                                    [-1,1],
                                    [-1,1]],dtype=float),
                     '4': np.array([[1,-8,8,-1],
                                    [-4,-6,12,-2],
                                    [-22,36,-18,4],
                                    [-4,18,-36,22],
                                    [2,-12,6,4]],dtype=float)/12}
SECOND_DERIV_COEFF = {'2': np.array([[1,-2,1],
                                     [1,-2,1],
                                     [1,-2,1]],dtype=float),
                      '4': np.array([[-1,16,-30,16,-1],
                                     [11,-20,6,4,-1],
                                     [35,-104,114,-56,11],
                                     [11,-56,114,-104,35],
                                     [-1,4,6,-20,11]],dtype=float)/12}

FIRST_DERIV_POS = {'2': np.array([[-1,1],
                                  [0,1],
                                  [-1,0]],dtype=float),
                   '4': np.array([[-2,-1,1,2],
                                  [-1,0,1,2],
                                  [0,1,2,3],
                                  [-3,-2,-1,0],
                                  [-2,-1,0,1]],dtype=float)}
SECOND_DERIV_POS = {'2': np.array([[-1,0,1],
                                  [0,1,2],
                                  [-2,-1,0]],dtype=float),
                   '4': np.array([[-2,-1,0,1,2],
                                  [-1,0,1,2,3],
                                  [0,1,2,3,4],
                                  [-4,-3,-2,-1,0],
                                  [-3,-2,-1,0,1]],dtype=float)}

HESSIAN_POS = {'2': np.array([[1,1,-1,-1],
                              [1,-1,1,-1]],dtype=float),
               '4': np.array([[2,2,1,1,-1,-1,-2,-2],
                              [2,-2,1,-1,1,-1,2,-2]],dtype=float)}
HESSIAN_COEFF = {'2': np.array([1,-1,-1,1],dtype=float)/4,
                 '4': np.array([-1,1,16,-16,-16,16,1,-1],dtype=float)/48}

def derivative(f, x, n=1, order=4, bounds=None, epsilon=1e-16, scale=1.0, dx=None, args=None):
    r"""Computes numerical derivatives of a callable function. Use the epsilon
    and scale parameters to estimate the optimal value of dx, if the latter is
    not provided. 
    

    Parameters
    ----------
    f : function
        Function to differentiate. Should take a float or an array as argument 
        and return a float or array (the returned array can have a different 
        shape as the input, but the first axis must match).
    x : float or array-like
        The position at which to evaluate the derivative.
    n : int, optional
        The number of derivatives to take. Can be 0, 1, 2. The default is 1.
    order : int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}/(\text{order+n})})`. Can be 2 or 4. Note that 
        the order at the endpoints is reduced by 1 as it would require 
        more function evaluations to keep the same order. The default is 4.
    bounds : tuple or None, optional
        Interval in which f can be called. If None, can be evaluated anywhere.
        The default is None.
    epsilon : float, optional
        Fractional accuracy at which f can be evaluated. If f is a simple 
        function, should be close to the machine precision. Default is 1e-16.
    scale : float, optional 
        Typical scale at which f(x) change by order 1. Default is 1.
    dx : float or None, optional
        The magnitude of finite differences. If None, use epsilon and scale to
        estimate the optimal dx. Default is None.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the derivative of :py:data:`f` evaluated at :py:data:`x`.

    """
    x = np.asarray(x)
    
    if bounds is None:
        bounds = (-np.inf,np.inf)
    if args is None:
        args = []
    
    assert isinstance(bounds, tuple) and len(bounds) == 2 and bounds[1] > bounds[0], "Derivative error: bounds must be a tuple of 2 elements or None."
    assert n == 0 or n == 1 or n == 2, "Derivative error: n must be 0, 1 or 2."
    assert order == 2 or order == 4, "Derivative error: order must be 2 or 4."
    assert np.all(x <= bounds[1]) and np.all(x >= bounds[0]), "Derivative error: x must be inside bounds."
    
    if n == 0:
        return f(x, *args)
    
    # If dx is not provided, we estimate it from scale and epsilon by minimizing 
    # the total error ~ epsilon/dx**n + dx**order.
    if dx is None:
        assert isinstance(epsilon, float), "Derivative error: epsilon must be a float."
        assert isinstance(scale, float), "Derivative error: scale must be a float."
        dx = scale * epsilon**(1/(n+order))
    
    # This step increases greatly the accuracy because it makes sure (x + dx) - x
    # is exactly equal to dx (no precision error).
    temp = x + dx
    dx = temp - x
    
    offset = np.zeros_like(x,dtype=int)
    offset -= x + dx > bounds[1]
    offset += x - dx < bounds[0]
    if order == 4:
        offset -= x + 2*dx > bounds[1]
        offset += x - 2*dx < bounds[0]
        
    if n == 1:
        pos = x[None,...] + FIRST_DERIV_POS[str(order)].T[:,offset.tolist()]*dx
        coeff = FIRST_DERIV_COEFF[str(order)].T[:,offset.tolist()]/dx
    elif n == 2:
        pos = x[None,...] + SECOND_DERIV_POS[str(order)].T[:,offset.tolist()]*dx
        coeff = SECOND_DERIV_COEFF[str(order)].T[:,offset.tolist()]/dx**2
    
    fx = f(pos, *args)
    fxShapeLength = len(fx.shape)
    coeffShapeLength = len(coeff.shape)
    return np.sum(coeff.reshape(coeff.shape+(fxShapeLength-coeffShapeLength)*(1,)) * f(pos, *args), axis=0)
        
    
def gradient(f, x, order=4, epsilon=1e-16, scale=1.0, dx=None, axis=None, args=None):
    r"""Computes the gradient of a callable function. Use the epsilon
    and scale parameters to estimate the optimal value of dx, if the latter is
    not provided. 
    

    Parameters
    ----------
    f : function
        Function to differentiate. Should take an array as argument 
        and return an array.
    x : array-like
        The position at which to evaluate the derivative. The size of the last
        axis must correspond to the number of variables on which f depends.
    order : int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}/(\text{order}+1)})`. 
        Can be 2 or 4. The default is 4.
    epsilon : float, optional
        Fractional accuracy at which f can be evaluated. If f is a simple 
        function, should be close to the machine precision. Default is 1e-16.
    scale : float or array-like, optional 
        Typical scale at which f(x) change by order 1. Can be an array, in which 
        case each element corresponds to the scale of a different variable.
        Default is 1.
    dx : float or None, optional
        The magnitude of finite differences. Can be an array, in which case 
        each element corresponds to the dx of a different variable.If None, use 
        epsilon and scale to estimate the optimal dx. Default is None.
    axis : list, int or None, optional
        Element of the gradient to return. If None, returns the whole gradient.
        Default is None.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the gradient of :py:data:`f` evaluated at :py:data:`x`.

    """
    x = np.asarray(x)
    nbrVariables = x.shape[-1]
    
    if args is None:
        args = []
        
    if isinstance(axis, int):
        axis = [axis]
    elif axis is None:
        axis = np.arange(nbrVariables).tolist()
    for i in axis:
        assert -nbrVariables <= i < nbrVariables, "Gradient error: axis must be between -nbrVariables and nbrVariables-1 or None."
    
    assert order == 2 or order == 4, "Gradient error: order must be 2 or 4."
    
    # If dx is not provided, we estimate it from scale and epsilon by minimizing 
    # the total error ~ epsilon/dx**n + dx**order.
    if dx is None:
        assert isinstance(epsilon, float), "Gradient error: epsilon must be a float."
        
        if isinstance(scale, float):
            scale = scale*np.ones(nbrVariables)
        else:
            scale = np.asanyarray(scale)
            assert scale.size == nbrVariables, "Gradient error: scale must be a float or an array of size nbrVariables."
        dx = scale * epsilon**(1/(1+order))
    elif isinstance(dx, float):
        dx = dx*np.ones(nbrVariables)
    else:
        dx = np.asanyarray(dx)
        assert dx.size == nbrVariables, "Gradient error: dx must be None, a float or an array of size nbrVariables."
    
    # This step increases greatly the accuracy because it makes sure (x + dx) - x
    # is exactly equal to dx (no precision error).
    temp = x + dx
    dx = temp - x
    
    pos = np.expand_dims(x, (-3,-2)) + FIRST_DERIV_POS[str(order)][0,:,None,None]*np.identity(nbrVariables)[axis,:]*np.expand_dims(dx, (-3,-2)) 
    shape = pos.shape[:-1]
    pos = pos.reshape((int(pos.size/nbrVariables), nbrVariables))
    coeff = FIRST_DERIV_COEFF[str(order)][0,:,None]/np.expand_dims(dx[...,axis], -2) 
        
    fEvaluation = f(pos, *args).reshape(shape)
    return np.sum(coeff*fEvaluation, axis=-2)

def hessian(f, x, order=4, epsilon=1e-16, scale=1.0, dx=None, xAxis=None, yAxis=None, args=None):
    r"""Computes the hessian of a callable function. Use the epsilon
    and scale parameters to estimate the optimal value of dx, if the latter is
    not provided. 
    

    Parameters
    ----------
    f : function
        Function to differentiate. Should take an array as argument 
        and return an array.
    x : array-like
        The position at which to evaluate the derivative. The size of the last
        axis must correspond to the number of variables on which f depends.
    order : int, optional
        The accuracy order of the scheme. Errors are of order
        :math:`\mathcal{O}({\rm d}x^{\text{order}/(\text{order}+2)})`. 
        Can be 2 or 4. The default is 4.
    epsilon : float, optional
        Fractional accuracy at which f can be evaluated. If f is a simple 
        function, should be close to the machine precision. Default is 1e-16.
    scale : float, optional 
        Typical scale at which f(x) change by order 1. Default is 1.
    dx : float or None, optional
        The magnitude of finite differences. If None, use epsilon and scale to
        estimate the optimal dx. Default is None.
    xAxis : list, int or None, optional
        Lines of the hessian matrix to return. If None, returns all the lines.
        Default is None.
    yAxis : list, int or None, optional
        Columns of the hessian matrix to return. If None, returns all the columns.
        Default is None.
    args: list, optional
        List of other fixed arguments passed to the function :math:`f`.

    Returns
    -------
    res : float
        The value of the hessian of :py:data:`f` evaluated at :py:data:`x`.

    """
    x = np.asarray(x)
    nbrVariables = x.shape[-1]
    
    if args is None:
        args = []
        
    if isinstance(xAxis, int):
        xAxis = [xAxis]
    elif xAxis is None:
        xAxis = np.arange(nbrVariables).tolist()
    for i in xAxis:
        assert -nbrVariables <= i < nbrVariables, "Hessian error: axis must be between -nbrVariables and nbrVariables-1 or None."
    if isinstance(yAxis, int):
        yAxis = [yAxis]
    elif yAxis is None:
        yAxis = np.arange(nbrVariables).tolist()
    for i in yAxis:
        assert -nbrVariables <= i < nbrVariables, "Hessian error: axis must be between -nbrVariables and nbrVariables-1 or None."
    
    assert order == 2 or order == 4, "Hessian error: order must be 2 or 4."
    
    # If dx is not provided, we estimate it from scale and epsilon by minimizing 
    # the total error ~ epsilon/dx**n + dx**order.
    if dx is None:
        assert isinstance(epsilon, float), "Hessian error: epsilon must be a float."
        
        if isinstance(scale, float):
            scale = scale*np.ones(nbrVariables)
        else:
            scale = np.asanyarray(scale)
            assert scale.size == nbrVariables, "Hessian error: scale must be a float or an array of size nbrVariables."
        dx = scale * epsilon**(1/(2+order))
    elif isinstance(dx, float):
        dx = dx*np.ones(nbrVariables)
    else:
        dx = np.asanyarray(dx)
        assert dx.size == nbrVariables, "Hessian error: dx must be None, a float or an array of size nbrVariables."
    
    # This step increases greatly the accuracy because it makes sure (x + dx) - x
    # is exactly equal to dx (no precision error).
    temp = x + dx
    dx = temp - x
    
    pos = (np.expand_dims(x, (-4,-3,-2)) 
           + HESSIAN_POS[str(order)][0,:,None,None,None]*np.identity(nbrVariables)[xAxis,None,:]*np.expand_dims(dx, (-4,-3,-2))  
           + HESSIAN_POS[str(order)][1,:,None,None,None]*np.identity(nbrVariables)[None,yAxis,:]*np.expand_dims(dx, (-4,-3,-2)) )
    shape = pos.shape[:-1]
    pos = pos.reshape((int(pos.size/nbrVariables), nbrVariables))
    coeff = HESSIAN_COEFF[str(order)][:,None,None]/(np.expand_dims(dx[...,yAxis], (-3,-2))*np.expand_dims(dx[...,xAxis], (-3,-1)))
    fEvaluation = f(pos, *args).reshape(shape)
    return np.sum(coeff*fEvaluation, axis=-3)
        
    


def gammaSq(v):
    r"""
    Lorentz factor :math:`\gamma^2` corresponding to velocity :math:`v`
    """
    return 1./(1. - v*v)

def boostVelocity(xi, v):
    """
    Lorentz-transformed velocity
    """
    return (xi - v)/(1. - xi*v)