from cProfile import run
from sympy import Symbol, lambdify
from numpy import array, concatenate, gradient, identity, linspace, newaxis, polyfit, polyder, roots, zeros, size, argmin, eye, inner, transpose, delete, dot
from numpy.random import rand
from numpy.lib.polynomial import polyder
from numpy.linalg import inv
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


def powell(function,x0,tol=1e-3,itermax=100,runitermax=False):
    from sympy.abc import x, y, a
    f  = function(x=x,y=y)
    f  = lambdify([x,y],f,"numpy")

    u  = identity(2)
    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        
        xitr  = zeros((3,2))
        for i in range(len(x0)):
            # Create parametric function f = f(a) = f( x0 - a*u(:,i) )
            fn_symb = f( xitr[i,0] + a*u[0,i] , xitr[i,1] + a*u[1,i] )
            fn = lambdify(a,fn_symb,"numpy")
            
            # Optimize the function fn for \alhpa (a) for the inner problem
            a_optimal, __, __, __ = Newton(function=fn,x0=0,tol=tol,N=itermax,h=1e-4)
            alpha = a_optimal[0]

            # Update the inner problem solution vetor value
            xitr[i+1,:] = xitr[i,:] + alpha*u[:,i]

        # Update matrix u & evaluate (u(:,n) = xn - x0)
        un = (xitr[2,:] - xsol[0,:])[:,newaxis]
        u = delete(u,0,1)
        u = concatenate( (u,un) , axis=1 )

        # Create parametric function f = f(a) = f( x0 - a*u(:,N) )
        fn_symb = f( xsol[k,0] + a*u[0,1] , xsol[k,1] + a*u[1,1] )
        fn = lambdify(a,fn_symb,"numpy")

        # Optimize the function fn for \alhpa (a)
        a_optimal, __, __, __ = Newton(function=fn,x0=0,tol=tol,N=itermax,h=1e-4)
        alpha = a_optimal[0]

        # Update the initial guess value (x0 & y0)
        xsol[k+1,:] = xsol[k,:] + alpha*u[:,1]
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break

    else:
        solution = xsol

    return solution



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
        print(fn_symb)
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



# Metodos de Segunda Ordem (Newton, Levenberg-Marquardt, Variável Métrica)

def newton2D(function,x0,tol=1e-3,itermax=100,runitermax=False):
    from sympy.abc import x, y, a
    f   = function(x=x,y=y)
    df  = [f.diff(x), f.diff(y)]
    d2f = [ [f.diff(x,2), f.diff(x,y)] , [f.diff(y,x), f.diff(y,2)] ]
    f   = lambdify([x,y],f,"numpy")
    df  = [lambdify([x,y],df[0],"numpy"), lambdify([x,y],df[1],"numpy")]
    d2f = [ [lambdify([x,y],d2f[0][0],"numpy"), lambdify([x,y],d2f[0][1],"numpy")], [lambdify([x,y],d2f[1][0],"numpy"), lambdify([x,y],d2f[1][1],"numpy")] ] 

    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Calculate the gradient & hassian of the function for x[k]
        df_k  = array([ df[0](xsol[k,0],xsol[k,1]),  df[1](xsol[k,1],xsol[k,1]) ])
        d2f_k = array([ [d2f[0][0](xsol[k,0],xsol[k,1]), d2f[0][1](xsol[k,0],xsol[k,1])], [d2f[1][0](xsol[k,0],xsol[k,1]), d2f[1][1](xsol[k,0],xsol[k,1])] ])

        # Create parametric function f = f(a) = f( x0 - a*df/dx|x0, y0 - a*df/dy|y0 )
        x_k = xsol[k,0] - a*dot(inv(d2f_k),df_k)
        fn_symb = f( x_k[0], x_k[1] )
        fn = lambdify(a,fn_symb,"numpy")

        # Optimize the function fn for \alhpa (a)
        a_optimal, __, __, __ = Newton(function=fn,x0=0,tol=tol,N=itermax,h=1e-4)
        alpha = a_optimal[0]

        # Update the initial guess value (x0 & y0)
        xsol[k+1,:] = xsol[k,:] - alpha*dot(inv(d2f_k),df_k)
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break

    else:
        solution = xsol

    return solution


def levenbergMarquardt2D(function,x0,lamb0=0,tol=1e-3,itermax=100,runitermax=False):
    from sympy.abc import x, y, a
    f   = function(x=x,y=y)
    df  = [f.diff(x), f.diff(y)]
    d2f = [ [f.diff(x,2), f.diff(x,y)] , [f.diff(y,x), f.diff(y,2)] ]
    f   = lambdify([x,y],f,"numpy")
    df  = [lambdify([x,y],df[0],"numpy"), lambdify([x,y],df[1],"numpy")]
    d2f = [ [lambdify([x,y],d2f[0][0],"numpy"), lambdify([x,y],d2f[0][1],"numpy")], [lambdify([x,y],d2f[1][0],"numpy"), lambdify([x,y],d2f[1][1],"numpy")] ] 

    lamb = zeros((itermax+1,))
    lamb[0] = lamb0
    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Calculate the gradient & hassian of the function for x[k]
        df_k  = array([ df[0](xsol[k,0],xsol[k,1]),  df[1](xsol[k,1],xsol[k,1]) ])
        d2f_k = array([ [d2f[0][0](xsol[k,0],xsol[k,1]), d2f[0][1](xsol[k,0],xsol[k,1])], [d2f[1][0](xsol[k,0],xsol[k,1]), d2f[1][1](xsol[k,0],xsol[k,1])] ])

        # Create parametric function f = f(a) = f( x0 - a*df/dx|x0, y0 - a*df/dy|y0 )
        x_k = xsol[k,0] - a*dot(( inv(d2f_k) + lamb[k]*identity(2) ), df_k)
        fn_symb = f( x_k[0], x_k[1] )
        fn = lambdify(a,fn_symb,"numpy")

        # Optimize the function fn for \alhpa (a)
        a_optimal, __, __, __ = Newton(function=fn,x0=0,tol=tol,N=itermax,h=1e-4)
        alpha = a_optimal[0]

        # Update the initial guess value (x0 & y0)
        xsol[k+1,:] = xsol[k,:] - alpha*dot(( inv(d2f_k) + lamb[k]*identity(2) ), df_k)
        lamb[k+1]   = abs( f(xsol[k+1,0],xsol[k+1,1]) -  f(xsol[k,0],xsol[k,1]) )/abs( f(xsol[k,0],xsol[k,1]) )
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break

    else:
        solution = xsol

    return solution



"""
sol = powell(sphere,x0=[-5, 5],tol=1e-3,itermax=100,runitermax=True)
print('\n',sol,'\n')
"""


"""
sol = maximaDescida(sphere,x0=[5, 5],tol=1e-3,itermax=100)
print('\n',sol,'\n')
"""

"""
sol = direcoesConjugadas(beale,x0=[5, 5],tol=1e-3,itermax=50,runitermax=True)
print('\n',sol,'\n')
"""


"""
sol = newton2D(sphere,x0=[5, 5],tol=1e-3,itermax=10,runitermax=True)
print('\n',sol,'\n')
"""

"""
sol = levenbergMarquardt2D(sphere,x0=[5, 5],lamb0=0,tol=1e-3,itermax=100,runitermax=True)
print('\n',sol,'\n')
"""
