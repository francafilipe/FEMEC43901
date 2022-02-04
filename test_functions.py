from math import pi, e
from numpy import array, cos, sin, exp, sqrt

# Default functions used as examples on the optimization course notes
# by prof. Aldemir Cavalini Jr.
def default1d(x=0):

    E = 5000            # [kN/cm²]  Elasticity module
    I = 3000            # [cm^4]    Moment of inertia
    L = 600             # [cm]      Lenght
    w = 2.5             # [kN/cm]   Maximum Load distribution value

    f = w/(120*E*I*L)*(-x**5+2*L**2*x**3-L**4*x)
    return f



# Test functions used for optimization scripts evaluation, known as ZDTs functions
# which specifications can be found at: https://en.wikipedia.org/wiki/Test_functions_for_optimization

# All functions were implemented so that it would be able to use it using N-D input variables or 1-D

def sphere(x):
    # x is an array with N dimensions
    f = sum(x**2)
    return f


def beale(x):
    # x can be either a scalar or a 2-d array
    if (len(x) == 1):
        y = 0
    else:
        y = x[1]
        x = x[0]

    f = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    return f


def threeHumpCamel(x):
    # x can be either a scalar or a 2-d array
    if (len(x) == 1):
        y = 0
    else:
        y = x[1]
        x = x[0]

    f = 2*x**2 - 1.05*x**4 + (1/6)*x**6 + x*y + y**2
    return f


def rastrigin(x):
    # x is a N-d array
    n = len(x)

    f = 10*n + sum(x**2 - 10*cos(2*pi*x))
    return f


def ackley(x):
    # x can be either a scalar or a 2-d array
    if (len(x) == 1):
        y = 0
    else:
        y = x[1]
        x = x[0]

    f = -20*exp(-0.2*sqrt(0.5*(x**2 + y**2))) - exp(0.5*(cos(2*pi*x) + cos(2*pi*y))) + e + 20
    return f


def leviNo13(x=0):
    # x can be either a scalar or a 2-d array
    if (len(x) == 1):
        y = 0
    else:
        y = x[1]
        x = x[0]

    f = (sin(3*pi*x))**2 + ((x-1)**2)*(1 + (sin(3*pi*y)**2)) + ((y-1)**2)*(1 + (sin(2*pi*y))**2) 
    return f


def easom(x):
    # x can be either a scalar or a 2-d array
    if (len(x) == 1):
        y = 0
    else:
        y = x[1]
        x = x[0]

    f = -cos(x)*cos(y)*exp(-((x-pi)**2 + (y-pi)**2))
    return f


def schafferNo2(x=0,y=0):
    # x can be either a scalar or a 2-d array
    if (len(x) == 1):
        y = 0
    else:
        y = x[1]
        x = x[0]

    f = 0.5 + ((sin(x**2 - y**2))**2 - 0.5)/((1 + 0.001*(x**2 + y**2))**2)
    return f
