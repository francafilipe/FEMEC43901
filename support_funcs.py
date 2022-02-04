from math import sqrt
from numpy import array, zeros, sum
from sympy import Float, fu
from test_functions import *

def fibonacci(index):
    # Returns de value of the fibonacci series in the predefined index
    value = (sqrt(5)/5)*(((1+sqrt(5))/2)**(index+1))-(sqrt(5)/5)*(((1-sqrt(5))/2)**(index+1))
    return value

def finiteDiff(function,x,h=1e-4):
    # Returns the value of the function f 1st and 2nd derivatives based on the finite differences method
    
    fxy      = function(x)

    # Compute the 1st-Order & 2nd-Order Partial Derivatives
    fx_plus  = zeros((len(x),))
    fx_minus = zeros((len(x),))
    dfdx     = zeros((len(x),))
    d2fdx2   = zeros((len(x),))

    for k in range(len(x)):
        xk_plus     = array([i for i in x])
        xk_minus    = array([i for i in x])

        xk_plus[k]  = xk_plus[k] + h
        xk_minus[k] = xk_minus[k] - h

        fx_plus[k]  = function(xk_plus)
        fx_minus[k] = function(xk_minus)

        dfdx[k]   = (fx_plus[k]-fx_minus[k])/(2*h)
        d2fdx2[k] = (fx_plus[k]+fx_minus[k]-2*fxy)/(h**2)

    gradiente = dfdx

    # Compute the 2nd-Order Cross Derivative (Hessian Matrix)
    f_plus  = function(x+h)
    f_minus = function(x-h)
    d2fdxy  = (f_plus+f_minus-sum(fx_plus)-sum(fx_minus)+2*fxy)

    hessian   = zeros((len(x),len(x)), dtype=Float)
    hessian[0,0] = d2fdx2[0]
    hessian[1,1] = d2fdx2[1]
    hessian[0,1] = d2fdxy
    hessian[1,0] = d2fdxy

    return gradiente, hessian
