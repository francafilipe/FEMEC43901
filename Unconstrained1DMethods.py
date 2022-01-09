from numpy import linspace, polyfit, polyder, roots, zeros
from numpy.lib.polynomial import polyder
from support_funcs import *
from zdt import *

# Methods based on the reduction of the Search Space

def bisseccao(function='Default',interval=[-1e3, 1e3],delta=1e-3,tol=1e-3,N=100):
    # Input Values
    i = 1
    a = interval[0]
    b = interval[1]

    # Loop for Reduction of Search Space
    while (i <= N):
        c = (a+b)/2
        f_plus  = ZDT((c+delta), func=function)
        f_minus = ZDT((c-delta), func=function)
        
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
            f_optimal = ZDT(c, func=function)
            #print('Resultado p/ função', function,'usando o método Bisseção:\n x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break
        i = i+1

    return x_optimal, f_optimal, iterations


def golden_ratio(function='Default',interval=[-1e3, 1e3],tol=1e-3,N=100):
    # Input Values
    tal = 0.618034
    i = 1
    a = interval[0]
    b = interval[1]
    alpha = a + (1-tal)*(b-a)
    beta  = a + tal*(b-a)

    # Loop for Reduction of Search Space
    while (i <= N):
        f_alpha = ZDT(alpha, func=function)
        f_beta  = ZDT(beta, func=function)
        
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
            f_optimal = ZDT(x_optimal, func=function)
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
        f_alpha = ZDT(alpha, func=function)
        f_beta  = ZDT(beta, func=function)
        
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
            f_optimal = ZDT(x_optimal, func=function)
            #print('Resultado p/ função', function,'usando o método Seção Aurea:
            #print('x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break 
        i   = i+1
        tal = fibonacci(N+i+1)/fibonacci(N+i+2)

    return x_optimal, f_optimal, iterations



# Methods not based on the reduction of the Search Space

def polinomialApproximation(function='Default',interval=[-1e3, 1e3],N=2):
    x = linspace(interval[0],interval[1],N+1)
    f = ZDT(x, func=function)
    p = polyfit(x, f, N)        # Polynomial approximation coefficients
    d = polyder(p, 1)           # Derivative of the polynomial (coefficients)
    
    x_optimal = roots(d)
    x_optimal = x_optimal[x_optimal>interval[0]]
    x_optimal = x_optimal[x_optimal<interval[1]-1e-3]

    return x_optimal


def Newton(function='Default',initial=0,tol=1e-3,N=100,h=1e-4):
    x = zeros(N)
    x[0] = initial
    i = 0

    while (i <= N):
        f_dev1, f_dev2 = finiteDiff(function,x[i],h)
        x[i+1] = x[i] - f_dev1/f_dev2
        if (abs(x[i+1]-x[i]) <= tol):
            iterations = i+1
            x_optimal = x[i+1]
            f_optimal = ZDT(x[i+1], func=function)
            break
        i = i+1

    return x_optimal, f_optimal, iterations


def LevenbergMarquardt(function='Default',x0=0,lamb0=1,tol=1e-3,N=100,h=1e-4):
    f = zeros(N)
    x = zeros(N)
    x[0] = x0
    f[0] = ZDT(x[0], func=function)
    lamb = zeros(N)
    lamb[0] = lamb0
    i = 0

    while (i <= N):
        f_dev1, f_dev2 = finiteDiff(function,x[i],h)
        x[i+1] = x[i] - f_dev1/(f_dev2+lamb[i])
        f[i+1] = ZDT(x[i+1], func=function)
        lamb[i+1] = abs(f[i+1] - f[i])/abs(f[0])
        if (abs(x[i+1]-x[i]) <= tol):
            iterations = i+1
            x_optimal = x[i+1]
            f_optimal = f[i+1]
            break
        i = i+1

    return x_optimal, f_optimal, iterations

x_optimal, f_optimal, iterations = LevenbergMarquardt(function='Default',x0=300,tol=1e-8,N=100,h=1e-4)
print(x_optimal, '\n', f_optimal, '\n', iterations)