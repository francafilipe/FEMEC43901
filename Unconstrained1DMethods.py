from numpy import array, linspace, polyfit, polyder, roots, zeros, size
from numpy.lib.polynomial import polyder
from support_funcs import *
from test_functions import *

# Methods based on the reduction of the Search Space

def bisseccao(function,interval=[-1e3, 1e3],delta=1e-3,tol=1e-3,N=100):
    # Input Values
    i = 1
    a = interval[0]
    b = interval[1]

    # Loop for Reduction of Search Space
    while (i <= N):
        c = (a+b)/2
        f_plus  = function(c+delta)
        f_minus = function(c-delta)

        if (f_plus >= f_minus):
            b = c
        elif (f_plus < f_minus):
            a = c
        else:
            raise NameError('Enter a new interval [a,b] or check the delta value')
        
        # Stopping Criteria   
        if ((b-a)/2 <= tol):
            iterations = i
            x_optimal = (a+b)/2
            f_optimal = function(c)
            #print('Resultado p/ função', function,'usando o método Bisseção:\n x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break
        i = i+1

    return x_optimal, f_optimal, iterations


def golden_ratio(function,interval=[-1e3, 1e3],tol=1e-3,N=100):
    # Input Values
    tal = 0.618034
    i = 1
    a = interval[0]
    b = interval[1]
    alpha = a + (1-tal)*(b-a)
    beta  = a + tal*(b-a)

    # Loop for Reduction of Search Space
    while (i <= N):
        f_alpha = function(alpha)
        f_beta  = function(beta)
        
        if (f_alpha >= f_beta):
            a     = alpha
            alpha = beta
            beta  = a + tal*(b-a)
        elif (f_alpha < f_beta):
            b     = beta
            beta  = alpha
            alpha = a + (1-tal)*(b-a)
        else:
            raise NameError('Enter a new interval [a,b] or check the delta value')

        # Stopping Criteria   
        if ((beta-alpha)/2 <= tol):
            iterations = i
            x_optimal = (alpha+beta)/2
            f_optimal = function(x_optimal)
            #print('Resultado p/ função', function,'usando o método Seção Aurea:
            #print('x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break 
        i = i+1

    return x_optimal, f_optimal, iterations


def fibonacci_method(function='Default',interval=[-1e3, 1e3],tol=1e-3,N=100):
    # Input Values
    tal = fibonacci(N+1)/fibonacci(N+2)
    i = 1
    a = interval[0]
    b = interval[1]
    alpha = a + (1-tal)*(b-a)
    beta  = b - (1-tal)*(b-a)

    # Loop for Reduction of Search Space
    while (i <= N):
        f_alpha = function(alpha)
        f_beta  = function(beta)
        
        if (f_alpha >= f_beta):
            a     = alpha
            alpha = beta
            beta  = b - (1-tal)*(b-a)
        elif (f_alpha < f_beta):
            b     = beta
            beta  = alpha
            alpha = a + (1-tal)*(b-a)
        else:
            print('Enter a new interval [a,b] or check the delta value')
            return None

        # Stopping Criteria   
        if ((beta-alpha)/2 <= tol):
            iterations = i
            x_optimal = (alpha+beta)/2
            f_optimal = function(x_optimal)
            #print('Resultado p/ função', function,'usando o método Seção Aurea:
            #print('x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break 
        i   = i+1
        tal = fibonacci(N+i+1)/fibonacci(N+i+2)

    return x_optimal, f_optimal, iterations



# Methods not based on the reduction of the Search Space

def polinomialApproximation(function,interval=[-1e3, 1e3],N=2):
    x = linspace(interval[0],interval[1],N+1)
    f = function(x)
    p = polyfit(x, f, N)        # Polynomial approximation coefficients
    d = polyder(p, 1)           # Derivative of the polynomial (coefficients)

    x_optimal = roots(d)
    x_optimal = x_optimal[x_optimal>interval[0]]
    x_optimal = x_optimal[x_optimal<interval[1]-1e-3]

    return x_optimal, p


def Newton(function,initial=0,tol=1e-3,N=100,h=1e-4):
    f = zeros(N)
    x = zeros(N)
    x[0] = initial
    f[0] = function(x[0])
    i = 0

    while (i <= N-2):
        f_dev1, f_dev2 = finiteDiff(function,x[i],h)
        x[i+1] = x[i] - f_dev1/f_dev2
        f[i+1] = function(x[i+1])
        if (abs(x[i+1]-x[i]) <= tol):
            iterations = i+1
            x_optimal = x[i+1]
            f_optimal = function(x[i+1])
            optimal = array([x_optimal, f_optimal])
            break
        i = i+1

    return optimal, iterations, x, f


def LevenbergMarquardt(function,x0=0,lamb0=1,tol=1e-3,N=100,h=1e-4):
    f = zeros(N)
    x = zeros(N)
    x[0] = x0
    f[0] = function(x[0])
    lamb = zeros(N)
    lamb[0] = lamb0
    i = 0

    while (i <= N-2):
        f_dev1, f_dev2 = finiteDiff(function,x[i],h)
        x[i+1] = x[i] - f_dev1/(f_dev2+lamb[i])
        f[i+1] = function(x[i+1])
        lamb[i+1] = abs(f[i+1] - f[i])/abs(f[0])
        if (abs(x[i+1]-x[i]) <= tol):
            iterations = i+1
            x_optimal = x[i+1]
            f_optimal = f[i+1]
            optimal = array([x_optimal, f_optimal])
            break
        i = i+1

    return optimal, iterations, x, f


def Quasi_Newton(function,x0=0,xp=0,tol=1e-3,N=100,h=1e-4):
    f = zeros(N)
    x = zeros(N)
    x[0] = x0
    f[0] = function(x[0])
    fp   = function(xp)
    fp_dev1, fp_dev2 = finiteDiff(function,xp,h)
    i = 0

    while (i <= N-2):
        f_dev1, f_dev2 = finiteDiff(function,x[i],h)
        x[i+1] = x[i] - f_dev1/((f_dev1-fp_dev1)/(x[i]-xp))
        f[i+1] = function(x[i+1])
        if (abs(x[i+1]-x[i]) <= tol):
            iterations = i+1
            x_optimal = x[i+1]
            f_optimal = f[i+1]
            optimal = array([x_optimal, f_optimal])
            break
        i = i+1

    return optimal, iterations, x, f
