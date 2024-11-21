import numpy as np
import matplotlib.pyplot as plt

def f(x):
    return x**2 - 4*x + 6

def grad_fx(x):
    return 2*x - 4

def steepest_descent(func, grad_func, x0, learning_rate=0.01, Maxiter=15, verbose=True):
    paths = []
    for i in range(Maxiter): #설정한 15만큼 반복
        x1 = x0 - learning_rate * grad_func(x0) #임의로설정한초기값에서 임의설정한 비율만큼 빼준다
        if verbose:
            print('{0:03d} : {1:4.3f}, {2:4.2E}'.format(i, x1, func(x1))) # ?번째 : x좌표, y좌표
        x0 = x1 #x1값을 새로운값으로변환
        paths.append(x0)
    return(x0, func(x0), paths) #세값을 반환하고

xopt, fopt, paths = steepest_descent(f, grad_fx, 1.0, learning_rate=0.9) #최종반환값이 여기들어간다

x = np.linspace(0.5, 3.5, 1000) #x값이 0.5~3.5까지 1000번쪼개서 나타낸다
paths = np.array(paths)
plt.plot(x,f(x)) #그래프그림
plt.grid() #격자그림
plt.xlabel('x') 
plt.ylabel('f(x)')
plt.title('plot of f(X)')
plt.plot(paths, f(paths), 'o-') #해당부분 점찍음
plt.show()

plt.plot(f(paths), 'o-')
plt.grid()
plt.xlabel('x')
plt.ylabel('cost')
plt.title('plot of cost')
plt.show()
