import math
from numpy import cos, sin, exp, sqrt

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
def sphere(x=0, y=0):

    f = x**2 + y**2
    return f


def beale(x=0, y=0.5):

    f = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2
    return f

def threeHumpCamel(x=0, y=0):

    f = 2*x**2 - 1.05*x**4 + (1/6)*x**6 + x*y + y**2
    return f

def rastrigin(x=0, y=0):

    f = 10 + x**2 - 10*cos(2*math.pi*x)
    return f


def ackley(x=0, y=0):

    f = -20*exp(-0.2*sqrt(0.5*(x**2 + y**2))) - exp(0.5*(cos(2*math.pi*x) + cos(2*math.pi*y))) + math.e + 20
    return f


def leviNo13(x=0, y=0):

    f = (sin(3*math.pi*x))**2 + ((x-1)**2)*(1 + (sin(3*math.pi*y)**2)) + ((y-1)**2)*(1 + (sin(2*math.pi*y))**2) 

def easom(x=0,y=0):

    f = -cos(x)*cos(y)*exp(-((x-math.pi)**2 + (y - math.pi)**2))
    return f

def schafferNo2(x=0,y=0):

    f = 0.5 + ((sin(x**2 - y**2))**2 - 0.5)/((1 + 0.001*(x**2 + y**2))**2)
    return f
