# File used to test the implemented algorithms for each optimization method

from numpy import poly1d, argmin, array
from Unconstrained1DMethods import *
import matplotlib.pyplot as plt

"""
# Evaluate Multi-modal functions

func = 'SchafferNo2'
dom = array([-5, 5])
div = 10
N = 1000
tol = 1e-3
h = 1e-4

subdoms = linspace(dom[0],dom[1],div+1) # [-5.12 -4.12 ...]
bests   = zeros((div,2))
for i in range(len(subdoms)-1):
    x0 = (subdoms[i+1]+subdoms[i])/2
    optimal, iterations, x, f = LevenbergMarquardt(function=func,x0=x0,tol=tol,N=N,h=h)
    bests[i,:] = optimal

min_index = argmin(bests[:,1])
optimal = bests[min_index,:]

y_real = ZDT(x=linspace(dom[0],dom[1],500),func=func)
plt.plot(linspace(dom[0],dom[1],500),y_real,'k-')
for i in range(len(subdoms)):
    plt.axvline(x=subdoms[i],color='k',ls='--',linewidth=0.5)
for i in range(len(bests)):
    plt.axvline(x=bests[i,0],ymin=0,ymax=bests[i,1],color='r',ls='-.',linewidth=0.75)
plt.axvline(x=optimal[0],color='g',ls='--',linewidth=1.5)

plt.suptitle('Otimização Multi-Modal p/ função ' + func + '\n Intervalo = ' + str(dom) + ', ' + str(div) + ' divisões de domínio', fontweight='bold')
plt.ylabel('Valor da função f(x)'); plt.xlabel('Valor da variável de projeto x')
plt.xlim(dom)
plt.show()
"""

"""
# Evaluate the Polynomial Approximation Method
# Simulation&Optimization Parameters
func = 'Rastrigin'
intv = [-5.12, 5.12]
max_N = 6
plot_styles = ['go','r^','bv','ms','ch']
y_real = ZDT(x=linspace(intv[0],intv[1],500),func=func)
plt.plot(linspace(intv[0],intv[1],500),y_real,'k-')

for N in range(2,max_N+1):
    sol, p = polinomialApproximation(function=func,interval=intv,N=N)
    poly = poly1d(p)
    x = linspace(intv[0],intv[1],50)
    y = poly(x)
    plt.plot(x,y,plot_styles[N-2])

plt.suptitle('Aproximação Polinomial p/ função ' + func + '\n Intervalo = ' + str(intv), fontweight='bold')
plt.ylabel('Valor da função f(x)'); plt.xlabel('Valor da variável de projeto x');
plt.legend(('Função f','n = 2','n = 3', 'n = 4', 'n = 5', 'n = 6'))
plt.show()
"""

"""
# Evaluate test functions for Newton, Levenberg-Marquardt & Quasi-Newton Methods


func = 'Rastrigin'
x0 = -0.1
N = 15
tol = 1e-8
h = 1e-4

optimal1, iterations1, x1, f1 = Newton(function=func,initial=x0,tol=tol,N=N,h=h)
optimal2, iterations2, x2, f2 = LevenbergMarquardt(function=func,x0=x0,tol=tol,N=N,h=h)
optimal3, iterations3, x3, f3 = Quasi_Newton(function=func,x0=x0,xp=0.1,tol=tol,N=N,h=h)

x1[iterations1:-1] = optimal1[0]; x1[-1] = optimal1[0]
x2[iterations2:-1] = optimal2[0]; x2[-1] = optimal2[0]
x3[iterations3:-1] = optimal3[0]; x3[-1] = optimal3[0]

f1[iterations1:-1] = optimal1[1]; f1[-1] = optimal1[1]
f2[iterations2:-1] = optimal2[1]; f2[-1] = optimal2[1]
f3[iterations3:-1] = optimal3[1]; f3[-1] = optimal3[1]


# Plot the Results
iterations = range(1,N+1)

figure1, plots = plt.subplots(2,1)

plots[0].plot(iterations, f1, 'b--',iterations, f2, 'b-.', iterations, f3, 'bo')
plots[1].plot(iterations, x1, 'b--', iterations, x2, 'b-.', iterations, x3, 'bo')
plt.suptitle('Convergência dos Métodos p/ função ' + func + '\n Tolerância = 10^-8 & Chute inicial = ' + str(x0), fontweight='bold')
plots[0].set_ylabel('Valor da função f(x)'); plots[1].set_ylabel('Valor da variável de projeto x');
plots[0].set_xlabel('N° de iterações'); plots[1].set_xlabel('N° de iterações') 
plt.setp(plots, xticks=range(0,N+1))
plots[0].legend(('Newton','Levenberg-Marquardt','Quasi-Newton'))

plt.show()
"""


"""
# Evaluate test functions for Bissection, Golden Ratio & Fibonacci Methods
print('Função Default - Exemplo Apostila')
print('Bisseção: ',     bisseccao('Default',[0, 600],0.2,1e-8,100))
print('Seção Aurea: ',  golden_ratio('Default',[0, 600],1e-8,100))
print('Fibonacci: ',    fibonacci_method('Default',[0, 600],1e-8,100))
print('-----------------------')

print('Função Sphere (y=0)')
print('Bisseção: ',     bisseccao('Sphere',[-1e3, 1e3],0.5,1e-8,100))
print('Seção Aurea: ',  golden_ratio('Sphere',[-1e3, 1e3],1e-8,100))
print('Fibonacci: ',    fibonacci_method('Sphere',[-1e3, 1e3],1e-8,100))
print('-----------------------')

print('Função Beale (y=0.5)')
print('Bisseção: ',     bisseccao('Beale',[-4.5, 4.5],0.2,1e-8,100))
print('Seção Aurea: ',  golden_ratio('Beale',[-4.5, 4.5],1e-8,100))
print('Fibonacci: ',    fibonacci_method('Beale',[-4.5, 4.5],1e-8,100))
print('-----------------------')

print('Função Three-Hump Camel (y=0)')
print('Bisseção: ',     bisseccao('Three-Hump Camel',[-5, 5],0.5,1e-8,100))
print('Seção Aurea: ',  golden_ratio('Three-Hump Camel',[-5, 5],1e-8,100))
print('Fibonacci: ',    fibonacci_method('Three-Hump Camel',[-5, 5],1e-8,100))
print('-----------------------')

print('Função Rastrigin (y=0)')
print('Bisseção: ',     bisseccao('Rastrigin',[-5.12, 5.12],0.5,1e-3,100))
print('Seção Aurea: ',  golden_ratio('Rastrigin',[-5.12, 5.12],1e-3,100))
print('Fibonacci: ',    fibonacci_method('Rastrigin',[-5.12, 5.12],1e-3,100))
print('-----------------------')
"""