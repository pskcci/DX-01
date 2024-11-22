import numpy as np
from numpy import linalg as LA
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D

###익명함수

f = lambda x,y : x+y
a = 1
b = 2
d = f(a,b)
print(d)

##딕셔너리

car = {"brand": "Ford","model": "Mustang","year":1964}

x = car.values()

print(x)

car["year"] = 2020

print(x)


car = {"brand": "Ford","model": "Mustang","year":1964}

x = car.items()
print(x)
car.update({"color": "red"})
x = car.items()
print(x)


#numpy
c = np.array([[1,2,3],[-1,1,4]])
print(LA.norm(c,axis = 0))
print(LA.norm(c,axis = 1))
print(LA.norm(c,ord = 1,axis = 1))
print(LA.norm(c,ord = 2,axis = 1))

a = np.array([1,2,3,4,5,6])
print(a.reshape(3,2))
print(a.shape)
b = a.reshape(3,-1)
print(b)
print(b.shape)
c = a.reshape(-1,2)
print(c)
print(c.shape)

##전치 연산

a = np.array([[1],[2],[3],[4]])
print(a)
print(a.T)
print(a.T.reshape(-1,4))
print(a.shape)
print(a.T.reshape(-1,4).T.shape)

a = np.array([1,2,3,4])
b = a. reshape(4,-1)
print(a)
print(a.reshape(2,-1))
print(a.shape,",",b.shape,",",np.array([[1,2,3,4]]).shape)

###line plot, surface

x = np.linspace(-2,2,11)
f = lambda x: x**2
fx = f(x)
print(x)
print(fx)

plt.plot(x,fx,'-o')
plt.grid()
plt.xlabel('x')
plt.ylabel('y')
plt.title('This is an example for 1d graph')
plt.show()

x = np.linspace(-2,2,11)
y = np.linspace(-2,2,11)

print(x)
print(y)

x,y = np.meshgrid(x,y)
print(x)
print(y)

f = lambda x,y : (x-1)**2 + (y-1 )**2
z = f(x,y)
print(z)

ax = plt.axes(projection = '3d', elev = 50, azim = -50)
ax.plot_surface(x,y,z,cmap=plt.cm.jet)
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
ax.set_zlabel('$z$')

plt.show()