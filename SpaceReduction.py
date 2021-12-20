from math import sqrt
from zdt import *

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
            x_optimal = (a+b)/2
            f_optimal = ZDT(c, func=function)
            #print('Resultado p/ função', function,'usando o método Bisseção:\n x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break
        i = i+1

    return x_optimal, f_optimal


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
            x_optimal = (alpha+beta)/2
            f_optimal = ZDT(x_optimal, func=function)
            #print('Resultado p/ função', function,'usando o método Seção Aurea:
            #print('x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break 
        i = i+1

    return x_optimal, f_optimal


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
            x_optimal = (alpha+beta)/2
            f_optimal = ZDT(x_optimal, func=function)
            #print('Resultado p/ função', function,'usando o método Seção Aurea:
            #print('x* = ',x_optimal,'& f(x*) = ',f_optimal)
            break 
        i   = i+1
        tal = fibonacci(N+i+1)/fibonacci(N+i+2)

    return x_optimal, f_optimal


def fibonacci(index):
    # Returns de value of the fibonacci series in the predefined index
    value = (sqrt(5)/5)*(((1+sqrt(5))/2)**(index+1))-(sqrt(5)/5)*(((1-sqrt(5))/2)**(index+1))
    return value
