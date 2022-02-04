from cProfile import run
from sympy import Symbol, lambdify, cos
from numpy import array, concatenate, gradient, identity, linspace, newaxis, polyfit, polyder, roots, zeros, size, argmin, eye, inner, transpose, delete, dot
from numpy.random import rand
from numpy.lib.polynomial import polyder
from numpy.linalg import inv
from Unconstrained1DMethods import *
from support_funcs import *
from test_functions import *
from time import sleep
from math import pi

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
    nVar = len(x0)
    u    = identity(nVar)
    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        
        xitr  = zeros((3,2))
        for i in range(len(x0)):

            # Find the value of a (\alpha) that optimize the function f( x[i-1] + a*u(:,i) ) for each directions u(:,i)
            a = zeros(itermax+1)
            i = 0
            while (i < itermax):
                x_i = xitr[i,0] + a[i]*u[0,i]

                # Evaluate the 1st & 2nd order derivatives for the unidimensional function f(a)
                fa        = function(x_i)
                f_h_plus  = function(xsol[k,:] - (a[i]+1e-4)*u[:,i])
                f_h_minus = function(xsol[k,:] - (a[i]-1e-4)*u[:,i])
                firstDerivative = (f_h_plus-f_h_minus)/(2*1e-4)
                secondDerivative = (f_h_plus+f_h_minus-2*fa)/(1e-4**2)

                a[i+1] = a[i] - firstDerivative/(secondDerivative+1e-4)
                i = i+1
                if (abs(a[i]-a[i-1]) <= tol):
                    break
            alpha = a[i]

            # Update the inner problem solution vetor value
            xitr[i+1,:] = xitr[i,:] + alpha*u[:,i]

        # Update matrix u & evaluate (u(:,n) = xn - x0)
        un = (xitr[2,:] - xsol[0,:])[:,newaxis]
        u = delete(u,0,1)
        u = concatenate( (u,un) , axis=1 )


        # Change multidimensional problem to unidimensional one using: x_{k+1} =  x_{k} - a*gradiente(x_{k})
        # Find the value of a (\alpha) that optimize the function (now unidimensional)
        a = zeros(itermax+1)
        i = 0
        while (i < itermax):
            x_k   = xsol[k,:] - a[i]*u[:,1]
            # Evaluate the 1st & 2nd order derivatives for the unidimensional function f(a)
            fa        = function(x_k)
            f_h_plus  = function(xsol[k,:] - (a[i]+1e-4)*u[:,1])
            f_h_minus = function(xsol[k,:] - (a[i]-1e-4)*u[:,1])
            firstDerivative = (f_h_plus-f_h_minus)/(2*1e-4)
            secondDerivative = (f_h_plus+f_h_minus-2*fa)/(1e-4**2)

            a[i+1] = a[i] - firstDerivative/(secondDerivative+1e-4)
            i = i+1
            if (abs(a[i]-a[i-1]) <= tol):
                break
        alpha = a[i]


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

def maximaDescida(function,x0,tol=1e-3,itermax=100,runitermax=False,diffRes=1e-4):
    grad = zeros((itermax+1,len(x0)))
    xsol = zeros((itermax+1,len(x0)))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Calculate the gradient of the function in the point x[k]
        grad[k,:], _ = finiteDiff(function,xsol[k,:],h=1e-4)


        # Change multidimensional problem to unidimensional one using: x_{k+1} =  x_{k} - a*gradiente(x_{k})
        # Find the value of a (\alpha) that optimize the function (now unidimensional)
        a = zeros(itermax+1)
        i = 0
        while (i < itermax):
            x_k   = xsol[k,:] - a[i]*grad[k,:]
            # Evaluate the 1st & 2nd order derivatives for the unidimensional function f(a)
            fa        = function(x_k)
            f_h_plus  = function(xsol[k,:] - (a[i]+diffRes)*grad[k,:])
            f_h_minus = function(xsol[k,:] - (a[i]-diffRes)*grad[k,:])
            firstDerivative = (f_h_plus-f_h_minus)/(2*diffRes)
            secondDerivative = (f_h_plus+f_h_minus-2*fa)/(diffRes**2)

            a[i+1] = a[i] - firstDerivative/(secondDerivative+1e-4)
            i = i+1
            if (abs(a[i]-a[i-1]) <= tol):
                break
        alpha = a[i]


        # Update the initial guess value (x0 & y0)
        xsol[k+1,0] = xsol[k,0] - alpha*grad[k,0]
        xsol[k+1,1] = xsol[k,1] - alpha*grad[k,1]
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break
    else:
        solution = xsol

    return solution


def direcoesConjugadas(function,x0,tol=1e-3,itermax=100,runitermax=False,diffRes=1e-4):
    S    = zeros((itermax+1,len(x0)))
    grad = zeros((itermax+1,len(x0)))
    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')


        # Calculate function gradient (for x[k]) and update parameter S_k
        grad[k,:], _ = finiteDiff(function,xsol[k,:],h=1e-4)

        if ( k == 0 ):
            S[k,:]  = -grad[k,:]
        else:
            S[k,:]  = -grad[k,:] + S[k-1,:]*( dot(grad[k,:].T, grad[k,:]) )/( dot(grad[k,:].T, grad[k,:]) )


        # Change multidimensional problem to unidimensional one using: x_{k+1} =  x_{k} - a*grad(x_{k})
        # Find the value of a (\alpha) that optimize the function (now unidimensional)
        a = zeros(itermax+1)
        i = 0
        while (i < itermax):
            x_k   = xsol[k,:] - a[i]*S[k,:]
            # Evaluate the 1st & 2nd order derivatives for the unidimensional function f(a)
            fa        = function(x_k)
            f_h_plus  = function(xsol[k,:] - (a[i]+diffRes)*S[k,:])
            f_h_minus = function(xsol[k,:] - (a[i]-diffRes)*S[k,:])
            firstDerivative = (f_h_plus-f_h_minus)/(2*diffRes)
            secondDerivative = (f_h_plus+f_h_minus-2*fa)/(diffRes**2)

            a[i+1] = a[i] - firstDerivative/(secondDerivative+1e-4)
            i = i+1
            if (abs(a[i]-a[i-1]) <= tol):
                break
        alpha = a[i]


        # Update the initial guess value (x0 & y0)
        xsol[k+1,0] = xsol[k,0] + alpha*S[k,0]
        xsol[k+1,1] = xsol[k,1] + alpha*S[k,1]
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break
    else:
        solution = xsol

    return solution



# Metodos de Segunda Ordem (Newton, Levenberg-Marquardt, Variável Métrica)

def newton2D(function,x0,tol=1e-3,itermax=100,runitermax=False,diffRes=1e-4):
    hess = zeros((itermax+1,len(x0),len(x0)))
    grad = zeros((itermax+1,len(x0)))
    xsol = zeros((itermax+1,len(x0)))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Calculate the gradient & hassian of the function for x[k]
        grad[k,:], hess[k,:,:] = finiteDiff(function,xsol[k,:],h=1e-4)
        S = dot( inv(hess[k,:,:]) , grad[k,:] )


        # Change multidimensional problem to unidimensional one using: x_{k+1} =  x_{k} - a*grad(x_{k})
        # Find the value of a (\alpha) that optimize the function (now unidimensional)
        a = zeros(itermax+1)
        i = 0
        while (i < itermax):
            x_k   = xsol[k,:] - a[i]*S
            # Evaluate the 1st & 2nd order derivatives for the unidimensional function f(a)
            fa        = function(x_k)
            f_h_plus  = function(xsol[k,:] - (a[i]+diffRes)*S)
            f_h_minus = function(xsol[k,:] - (a[i]-diffRes)*S)
            firstDerivative = (f_h_plus-f_h_minus)/(2*diffRes)
            secondDerivative = (f_h_plus+f_h_minus-2*fa)/(diffRes**2)

            a[i+1] = a[i] - firstDerivative/(secondDerivative+1e-4)
            i = i+1
            if (abs(a[i]-a[i-1]) <= tol):
                break
        alpha = a[i]


        # Update the initial guess value (x0 & y0)
        xsol[k+1,:] = xsol[k,:] - alpha*S
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break
    else:
        solution = xsol

    return solution


def levenbergMarquardt2D(function,x0,lamb0=0,tol=1e-3,itermax=100,runitermax=False,diffRes=1e-4):
    nVar = len(x0)
    hess = zeros((itermax+1,nVar,nVar))
    grad = zeros((itermax+1,nVar))
    xsol = zeros((itermax+1,nVar))
    xsol[0,:] = x0

    lamb = zeros((itermax+1,))
    lamb[0] = lamb0
    xsol = zeros((itermax+1,2))
    xsol[0,:] = x0

    k = 0
    while (k < itermax):
        print('\n Starting ' + str(k+1) + '° iteration: ')

        # Calculate the gradient & hassian of the function for x[k]
        grad[k,:], hess[k,:,:] = finiteDiff(function,xsol[k,:],h=1e-4)
        S = dot( (inv(hess[k,:,:]) + lamb[k]*eye(nVar)) , grad[k,:] )


        # Change multidimensional problem to unidimensional one using: x_{k+1} =  x_{k} - a*grad(x_{k})
        # Find the value of a (\alpha) that optimize the function (now unidimensional)
        a = zeros(itermax+1)
        i = 0
        while (i < itermax):
            x_k   = xsol[k,:] - a[i]*S
            # Evaluate the 1st & 2nd order derivatives for the unidimensional function f(a)
            fa        = function(x_k)
            f_h_plus  = function(xsol[k,:] - (a[i]+diffRes)*S)
            f_h_minus = function(xsol[k,:] - (a[i]-diffRes)*S)
            firstDerivative = (f_h_plus-f_h_minus)/(2*diffRes)
            secondDerivative = (f_h_plus+f_h_minus-2*fa)/(diffRes**2)

            a[i+1] = a[i] - firstDerivative/(secondDerivative+1e-4)
            i = i+1
            if (abs(a[i]-a[i-1]) <= tol):
                break
        alpha = a[i]


        # Update the initial guess value (x0 & y0)
        xsol[k+1,:] = xsol[k,:] - alpha*S
        lamb[k+1]   = abs( function(xsol[k+1,:]) -  function(xsol[k,:]) ) / abs( function(xsol[k,:]) )
        k = k+1

        if ( (sum(abs(xsol[k,:] - xsol[k-1,:])) <= tol) and (not runitermax) ):
            solution = xsol[k,:]
            break

    else:
        solution = xsol

    return solution



sol = powell(sphere,[5., 5.],tol=1e-3,itermax=10,runitermax=True)
print(sol)