import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf




def steepest_descent_twod(func, gradx, gtady,x0, Maxlter=10, learning_rate=0.25, verbose=True):

    paths = [x0]
    fval_paths = [f(x0[0],x0[1])]
    for i in range(Maxlter):
        grad = np.array([grad_f_x(*x0), grad_f_y(*x0)])
        x1 = x0 - learning_rate * grad
        fval = f(*x1)
        if verbose:
            print(i, x1, fval)
        x0 = x1
        paths.append(x0)
        fval_paths.append(fval)
    paths = np.array(paths)
    paths = np.array(np.matrix(paths).T)
    fval_paths = np.array(fval_paths)
    return(x0,fval, paths, fval_paths)
def loss(w, x_set, y_set) :
    N = len(x_set)
    val = 0.0
    for i in range(len(x_set)):
        val += 0.5 * (w[0]* x_set[i]+w[1]-y_set[i])**2
        return val/N
def loss_grad(w,x_set,y_set):
    N= len(x_set)
    val = np.zeros(len(w))
    for i in range(len(x_set)):
        er = w[0] * x_set[i] + w [1] - y_set[i]
        val += er * np.array([x_set[i],1.0])
    return val / N
def generate_batches(batch_size, features, labels):
    assert len(features) == len(labels)
    outout_batches = []
    sample_size = len(features)
    for start_i in range(0,sample_size,batch_size):
        end_i = start_i + batch_size
        batch = [features[start_i:end_i],labels[start_i:end_i]]
        outout_batches.append(batch)
    return outout_batches

xmin, xmax, xstep = -4.0,4.0, .25
ymin, ymax, ystep = -4.0,4.0, .25

x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),np.arange(ymin, ymax + ystep, ystep))

f = lambda x,y : (x-2)**2 + (y-2)**2
z = f(x,y)
minima = np.array([2.,2.])

f(*minima)

minima_ = minima.reshape(-1,1)
print(minima,minima_)
surf(f,x,y,minima=minima_)

grad_f_x = lambda x,y : 2 * (x-2)
grad_f_y = lambda x,y : 2 * (y-2)

contour_with_quiver(f,x,y,grad_f_x,grad_f_y,minima=minima_)

x0 = np.array([-2.,-2.])
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x,grad_f_y,x0)

contour_with_path(f,x,y,paths,minima=np.array([[2],[2]]))

np.random.seed(320)

x_train = np.linspace(-1,1,51)
f = lambda x : 0.5*x+1.0
y_train = f(x_train) + 0.4 *np.random.rand(len(x_train))
plt.plot(x_train,y_train,'o')
plt.grid()
plt.show()

np.random.seed(303)
shuffled_id = np.arange(0,len(x_train))
np.random.shuffle(shuffled_id)
x_train = x_train[shuffled_id]
y_train = y_train[shuffled_id]


######GD
batch_size = 10
lr = 0.01
MaxEpochs = 51
alpha = .9

w0 = np.array([4.0,-1.0])
path_sgd = []
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch,w0,loss(w0,x_train,y_train))
    for x_batch,y_batch in generate_batches(batch_size,x_train,y_train):

        path_sgd.append(w0)
        grad = loss_grad(w0,x_batch,y_batch)
        w1 = w0 - lr * grad
        w0 = w1

#####Adagrad
w0 = np.array([4.0,-1.0])
path_mm = []
velocity = np.zeros_like(w0)
for epoch in range(MaxEpochs):
    if epoch % 10 == 0:
        print(epoch,w0,loss(w0,x_train,y_train))
    for x_batch,y_batch in generate_batches(batch_size,x_train,y_train):
        path_mm.append(w0)
        grad = loss_grad(w0,x_batch,y_batch)
        velocity = alpha * velocity - lr * grad
        w1 = w0 + velocity
        w0 = w1
w0 = np.linspace(-2,5,101)
w1 = np.linspace(-2,5,101)
w0,w1 = np.meshgrid(w0,w1)
LOSSW = w0*0
for i in range(w0.shape[0]):
    for j in range(w0.shape[1]):
        wij = np.array([w0[i,j],w1[i,j]])
        LOSSW[i,j] = loss(wij,x_train,y_train)
fig, ax = plt.subplots(figsize=(6,6))

ax.contour(w0, w1, LOSSW, cmp=plt.cm.jet, levels=np.linspace(0, np.max(LOSSW.flatten()), 20))

paths = path_sgd
paths = np.array(np.matrix(paths).T)
ax.quiver(paths[0, :-1], paths[1, :-1], paths[0, 1:] - paths[0, :-1],  paths[1, 1:] - paths[1, :-1],scale_units='xy', angles='xy', scale=1, color='r')
plt.legend(['GD','Moment'])
plt.show()