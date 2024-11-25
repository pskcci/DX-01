import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf

def f(x):
    return x**2 -4*x +6
def grad_fx(x):
    return 2*x- 4
def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, verbose=True):
    paths = []
    for i in range(Maxlter):
        x1 = x0 - learning_rate * grad_func(x0)
        if verbose:
            print('{0:03d} : {1:4.3f}'.format(i,x1,func(x1)))
        x0 = x1
        paths.append(x0)
    return(x0, func(x0), paths)
    

NumberofPoints = 101
x = np.linspace(-5,5,NumberofPoints)
fx = f(x)

plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')
plt.show()

xid = np.argmin(fx)
xopt = x[xid]
print(xopt, f(xopt))

plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('plot of f(x)2')

plt.plot(xopt, f(xopt),'xr')
plt.show()

xopt, fopt, paths = steepest_descent(f,grad_fx,0.0,learning_rate=1.2)

x = np.linspace(0.5,2.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('plot of f(x)3')

plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths),'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()


xopt, fopt, paths = steepest_descent(f ,grad_fx, 1.0, learning_rate=1)

x = np.linspace(0.5,3.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)4')
plt.plot(paths, f(paths), 'o-')
plt.show()

plt.plot(f(paths),'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost2')
plt.show()

xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=0.001)
x = np.linspace(0.5,3.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths,f(paths),'o-')
plt.show()

plt.plot(f(paths))
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()

xopt, fopt, paths = steepest_descent(f, grad_fx, 3.0, learning_rate=0.9)

x = np.linspace(0.5,3.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('plot of f(x)')

plt.plot(paths,f(paths),'o-')
plt.show()

