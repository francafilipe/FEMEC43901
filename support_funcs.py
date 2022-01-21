from math import sqrt
from test_functions import *

def fibonacci(index):
    # Returns de value of the fibonacci series in the predefined index
    value = (sqrt(5)/5)*(((1+sqrt(5))/2)**(index+1))-(sqrt(5)/5)*(((1-sqrt(5))/2)**(index+1))
    return value

def finiteDiff(function,x,h):
    # Returns the value of the function f 1st and 2nd derivatives based on the finite differences method
    fx       = function(x)
    fx_plus  = function(x+h)
    fx_minus = function(x-h)

    f_dev1 = (fx_plus-fx_minus)/(2*h)
    f_dev2 = (fx_plus-2*fx+fx_minus)/(h**2)

    return f_dev1, f_dev2

