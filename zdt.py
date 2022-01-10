import math
from numpy import cos 

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

    else:
        raise NameError('Função a ser avaliada não definida!')

    return f
