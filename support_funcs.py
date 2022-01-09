from math import sqrt
from zdt import *

def fibonacci(index):
    # Returns de value of the fibonacci series in the predefined index
    value = (sqrt(5)/5)*(((1+sqrt(5))/2)**(index+1))-(sqrt(5)/5)*(((1-sqrt(5))/2)**(index+1))
    return value

def finiteDiff(f,x,h):
    # Returns the value of the function f 1st and 2nd derivatives based on the finite differences method
    fx       = ZDT(x, func=f)
    fx_plus  = ZDT(x+h, func=f)
    fx_minus = ZDT(x-h, func=f)

    f_dev1 = (fx_plus-fx_minus)/(2*h)
    f_dev2 = (fx_plus-2*fx+fx_minus)/(h**2)

    return f_dev1, f_dev2

