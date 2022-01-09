from math import sqrt

def fibonacci(index):
    # Returns de value of the fibonacci series in the predefined index
    value = (sqrt(5)/5)*(((1+sqrt(5))/2)**(index+1))-(sqrt(5)/5)*(((1-sqrt(5))/2)**(index+1))
    return value

def finiteDiff(f,x,h):
    # Returns the value of the function f 1st and 2nd derivatives based on the finite differences method
    f_dev1 = (f(x+h)-f(x-h))/(2*h)
    f_dev2 = (f(x+h)-2*f(x)+f(x-h))/(h**2)
    return f_dev1, f_dev2

