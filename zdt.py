import math
from numpy import cos, sin, exp, sqrt

def ZDT(x=0,y=0,func='Default'):

    if func=='Default':
        E = 5000            # [kN/cm²]  Elasticity module
        I = 3000            # [cm^4]    Moment of inertia
        L = 600             # [cm]      Lenght
        w = 2.5             # [kN/cm]   Maximum Load distribution value

        f = w/(120*E*I*L)*(-x**5+2*L**2*x**3-L**4*x)

    elif func=='Sphere':
        f = x**2 + y**2

    elif func=='Beale':
        y = 0.5
        f = (1.5 - x + x*y)**2 + (2.25 - x + x*y**2)**2 + (2.625 - x + x*y**3)**2

    elif func=='Three-Hump Camel':
        f = 2*x**2 - 1.05*x**4 + (1/6)*x**6 + x*y + y**2
    
    elif func=='Rastrigin':
        f = 10 + x**2 - 10*cos(2*math.pi*x)

    elif func=='Ackley':
        f = -20*exp(-0.2*sqrt(0.5*(x**2 + y**2))) - exp(0.5*(cos(2*math.pi*x) + cos(2*math.pi*y))) + math.e + 20

    elif func=='LeviNo13':
        f = (sin(3*math.pi*x))**2 + ((x-1)**2)*(1 + (sin(3*math.pi*y)**2)) + ((y-1)**2)*(1 + (sin(2*math.pi*y))**2) 

    elif func=='Easom':
        f = -cos(x)*cos(y)*exp(-((x-math.pi)**2 + (y - math.pi)**2))

    elif func=='SchafferNo2':
        f = 0.5 + ((sin(x**2 - y**2))**2 - 0.5)/((1 + 0.001*(x**2 + y**2))**2)

    else:
        raise NameError('Função a ser avaliada não definida!')

    return f
