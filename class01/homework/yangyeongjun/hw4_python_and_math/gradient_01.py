import numpy as np
import matplotlib.pylab as plt
from mpl_toolkits.mplot3d import Axes3D
x = np.linspace(-2,2, 11)
y = np.linspace(-2,2, 11)

print(x)
print(y)

x,y = np.meshgrid(x,y)
print(x)
print(y)

f = lambda x,y : (x-1)**2 + (y-1)**2
z = f(x,y)

print(z)

grad_f_x = lambda x, y : 2 * (x-1)
grad_f_y = lambda x, y : 2 * (y-1)

dz_dx = grad_f_x(x,y)
dz_dy = grad_f_y(x,y)

ax = plt.axes()
ax.contour(x,y,z, levels = np.linspace(0,10,20), cmap=plt.cm.jet)
ax.quiver(x,y, -dz_dx, -dz_dy)
ax.grid()
ax.axis('equal')
ax.set_xlabel('$x$')
ax.set_ylabel('$y$')
plt.show()