import matplotlib.pylab as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf

#1
def f(x):
    return x**2 -4*x + 6

points = 101
x = np.linspace(-5.,5,points)
fx = f(x)
plt.plot(x,fx)
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')
plt.title('check')
plt.show()

#2
def f(x):
    return x**2 -4*x +6

def grad_fx(x):
    return 2*x -4

def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, Verbose=True):
    paths =[]
    for i in range(Maxlter):
        #
        x1 = x0 - learning_rate * grad_func(x0)
        if Verbose:
            print('{0:03d} : {1:4.3f},{2:4.2E}'.format(i,x1,func(x1)))
        x0 = x1
        paths.append(x0)
    return(x0, func(x0), paths)

xopt, fopt, paths = steepest_descent(f,grad_fx, 0.0, learning_rate=1.2)


x = np.linspace(0.5,2.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(paths, f(paths), 'o-')
plt.show()

#3
def f(x):
    return x**2 -4*x +6

def grad_fx(x):
    return 2*x -4

def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxlter=10, Verbose=True):
    paths =[]
    for i in range(Maxlter):
        x1 = x0 - learning_rate * grad_func(x0)
        if Verbose:
            print('{0:03d} : {1:4.3f},{2:4.2E}'.format(i,x1,func(x1)))
        x0 = x1
        paths.append(x0)
    return(x0, func(x0), paths)

xopt, fopt, paths = steepest_descent(f,grad_fx, 1.0, learning_rate=0.89)

x = np.linspace(0.5,3.5,1000)
paths = np.array(paths)
plt.plot(x,f(x))
plt.grid()
plt.xlabel('x')
plt.ylabel('f(x)')

plt.plot(paths, f(paths), 'o-')
plt.show()

#4
xmin, xmax, xstep = -4.0, 4.0, .25
ymin, ymax, ystep = -4.0, 4.0, .25

x,y = np.meshgrid(np.arange(xmin,xmax + xstep, xstep),np.arange(ymin,ymax + ystep, ystep))

f = lambda x,y : (x-2)**2 + (y-2)**2
z = f(x,y)
minima = np.array([2.,2.])

f(*minima)

minima_ = minima.reshape(-1,1)
print(minima,minima_)
surf(f,x,y,minima=minima_)

grad_f_x = lambda x,y: 2 * (x-2)
grad_f_y = lambda x,y: 2 * (y-2)

contour_with_quiver(f,x,y,grad_f_x,grad_f_y,minima=minima_)

def steepest_descent(func, gradx, grady, x0, Maxlter=10, learning_rate=0.25,  Verbose=True):
    paths =[x0]
    fval_paths = [f(x0[0],x0[1])]
    for i in range(Maxlter):
        grad = np.array([grad_f_x(*x0),grad_f_y(*x0)])
        x1 = x0 - learning_rate * grad
        fval = f(*x1)
        if Verbose:
            print(i,x1,fval)
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)
    paths = np.array(paths)
    paths = np.array(np.matrix(paths).T)
    fval_paths = np.array(fval_paths)
    return(x0, fval, paths, fval_paths)

x0 = np.array([-2.,-2.])
xopt,fopt,paths,fval_paths = steepest_descent(f,grad_f_x,grad_f_y,x0)
contour_with_path(f,x,y,paths, minima=np.array([[2],[2]]))

#5
np.random.seed(320)
x_train = np.linspace(-1,1,51)
f = lambda x: 0.5 * x + 1.0
y_train = f(x_train) +0.4 *np.random.rand(len(x_train))
plt.plot(x_train,y_train,'o')
plt.grid()
plt.show()

"""
np.random.seed(303)
shuffled_id = np.arange(0,len(x_train))
np.random.shuffle(shuffled_id)
x_train = x_train[shuffled_id]
y_train = y_train[shuffled_id]
"""
#6
#SGD
batch_size =10
lr = 0.01
MaxEpochs = 51

#momentum
alpha = .9

w0 = np.array([4.0,-1.0])
path_sgd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch,w0,loss(w0,x_train,y_train))
    for x_batch, y_batch in generate_bathes(batch_size, x_train, y_train):
        path_sgd.append(w0)
        grad = loss_grad(w0,x_batch,y_batch)
        w1 = w0 -lr *grad
        w0 = w1

#Momentum
w0 = np.array([4.0, -1.0])
path_mm =[]
velocity = np.zeros_like(w0)
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch,20,loss(w0,x_train,y_train))
