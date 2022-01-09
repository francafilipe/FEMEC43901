# File used to test the implemented algorithms for each optimization method

from Unconstrained1DMethods import *

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
