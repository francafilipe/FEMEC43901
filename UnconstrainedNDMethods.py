from numpy import array, linspace, polyfit, polyder, roots, zeros, size, argmin
from numpy.random import rand
from numpy.lib.polynomial import polyder
from support_funcs import *
from zdt import *


def randomSearch(function,xlim,ylim,N,order=2):
    # Define N random points to be tested 
    r = rand(N)
    x = xlim[0] + r*(xlim[1] - xlim[0])
    y = ylim[0] + r*(ylim[1] - ylim[0])

    # Evaluate function
    f = ZDT(x=x, y=y, func=function)

    evaluated = array([x, y, f])

    # Find minimum value from the evaluated one
    min_index = argmin(f)
    f_optimal = f[min_index]
    x_optimal = x[min_index]
    y_optimal = y[min_index]

    optimal = array([x_optimal, y_optimal, f_optimal])

    return optimal, evaluated

