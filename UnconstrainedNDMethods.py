from sympy import Symbol, lambdify
from numpy import array, gradient, linspace, polyfit, polyder, roots, zeros, size, argmin, eye, inner, transpose
from numpy.random import rand
from numpy.lib.polynomial import polyder
from Unconstrained1DMethods import *
from support_funcs import *
from test_functions import *
from time import sleep

# Metodos de Ordem Zero (Busca Aleatoria e Direções Conjugadas de Powell)

def randomSearch(function,xlim,ylim,N,order=2):
    # Define N random points to be tested 
    r = rand(N)
    x = xlim[0] + r*(xlim[1] - xlim[0])
    y = ylim[0] + r*(ylim[1] - ylim[0])

    # Evaluate function
    f = function(x=x, y=y)

    evaluated = array([x, y, f])

    # Find minimum value from the evaluated one
    min_index = argmin(f)
    f_optimal = f[min_index]
    x_optimal = x[min_index]
    y_optimal = y[min_index]

    optimal = array([x_optimal, y_optimal, f_optimal])

    return optimal, evaluated


# Metodos de Primeira Ordem (Maxima Descida & Direções Conjugadas)

def maximaDescida(function,x0,tol=1e-3,itermax=100,runitermax=False):
    from sympy.abc import x, y, a
    f  = function(x=x,y=y)
    df = [f.diff(x), f.diff(y)]
    f  = lambdify([x,y],f,"numpy")
    df = [lambdify([x,y],df[0],"numpy"), lambdify([x,y],df[1],"numpy")] 

    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Create parametric function f = f(a) = f( x0 - a*df/dx|x0, y0 - a*df/dy|y0 )
        fn_symb = f( xsol[k,0] - a*df[0](xsol[k,0],xsol[k,1]), xsol[k,1] - a*df[1](xsol[k,1],xsol[k,1]) )
        fn = lambdify(a,fn_symb,"numpy")

        # Optimize the function fn for \alhpa (a)
        a_optimal, __, __, __ = Newton(function=fn,x0=0,tol=tol,N=itermax,h=1e-4)

        # Update the initial guess value (x0 & y0)
        xsol[k+1,0] = xsol[k,0] - a_optimal[0]*df[0](xsol[k,0],xsol[k,1])
        xsol[k+1,1] = xsol[k,1] - a_optimal[0]*df[1](xsol[k,0],xsol[k,1])
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break

    else:
        solution = xsol

    return solution


def direcoesConjugadas(function,x0,tol=1e-3,itermax=100,runitermax=False):
    from sympy.abc import x, y, a
    f  = function(x=x,y=y)
    df = [f.diff(x), f.diff(y)]
    f  = lambdify([x,y],f,"numpy")
    df = [lambdify([x,y],df[0],"numpy"), lambdify([x,y],df[1],"numpy")] 

    S = zeros((itermax+1,2))
    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Calculate function gradient and update parameter S_k
        if ( k == 0 ):
            S[k,0]  = -df[0](xsol[k,0],xsol[k,1])
            S[k,1]  = -df[1](xsol[k,0],xsol[k,1])
        else:
            gradk   = [ df[0](xsol[k,0],xsol[k,1]) , df[1](xsol[k,0],xsol[k,1]) ]
            gradk_1 = [ df[0](xsol[k-1,0],xsol[k-1,1]) , df[1](xsol[k-1,0],xsol[k-1,1]) ]

            S[k,0]  = -gradk[0] + S[k-1,0]*(transpose(gradk[0])*gradk[0])/(transpose(gradk_1[0])*gradk_1[0])
            S[k,1]  = -gradk[1] + S[k-1,1]*(transpose(gradk[1])*gradk[1])/(transpose(gradk_1[1])*gradk_1[1])

        # Create parametric function f = f(a) = f( x0 - a*df/dx|x0, y0 - a*df/dy|y0 )
        fn_symb = f( xsol[k,0] + a*S[k,0] , xsol[k,1] + a*S[k,1] )
        fn = lambdify(a,fn_symb,"numpy")

        # Optimize the function fn for \alhpa (a)
        a_optimal, __, __, __ = Newton(function=fn,x0=0,tol=tol,N=itermax,h=1e-4)

        # Update the initial guess value (x0 & y0)
        xsol[k+1,0] = xsol[k,0] + a_optimal[0]*S[k,0]
        xsol[k+1,1] = xsol[k,1] + a_optimal[0]*S[k,0]
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break

    else:
        solution = xsol

    return solution



"""
sol = maximaDescida(rastrigin,x0=[5, 5],tol=1e-3,itermax=100)
print('\n',sol,'\n')
"""

"""
sol = direcoesConjugadas(beale,x0=[5, 5],tol=1e-3,itermax=50,runitermax=True)
print('\n',sol,'\n')
"""
