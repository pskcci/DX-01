import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 6

NumberOfPoints = 101
x = np.linspace(-5.,5,NumberOfPoints)
fx = f(x)
plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()
