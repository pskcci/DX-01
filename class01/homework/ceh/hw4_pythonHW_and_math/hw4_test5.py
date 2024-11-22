import numpy as np
import matplotlib.pylab as plt
from matplotlib.colors import LogNorm
from visualize import contour_with_quiver
from visualize import contour_with_path
from visualize import surf

# x와 y의 범위 및 스텝 설정
xmin, xmax, xstep = -4.0, 4.0, .25
ymin, ymax, ystep = -4.0, 4.0, .25

# x와 y의 메쉬 그리드 생성
x, y = np.meshgrid(np.arange(xmin, xmax + xstep, xstep),
                   np.arange(ymin, ymax + ystep, ystep))

# 목표 함수 정의: (x-2)² + (y-2)²
f = lambda x,y : (x-2)**2 + (y-2)**2
z = f(x, y)  # 함수 값 계산
minima = np.array([2., 2.])  # 최소값 위치

# 최소값 계산
f(*minima)

# 최소값을 2D 배열로 변환
minima_ = minima.reshape(-1, 1)
print(minima_)  # 최소값 출력
surf(f, x, y, minima=minima_)  # 함수의 3D 표면 시각화

# 기울기 함수 정의
grad_f_x = lambda x, y : 2 * (x-2)  # x에 대한 기울기
grad_f_y = lambda x, y : 2 * (y-2)  # y에 대한 기울기

# 등고선과 기울기 화살표 시각화
contour_with_quiver(f, x, y, grad_f_x, grad_f_y, minima=minima_)

# 경사 하강법 함수 정의
def steepest_descent_twod(func, gradx, grady, x0, Maxlter=10, learn_rate=0.25, verbose=True):
    paths = [x0]  # 경로 저장 리스트
    fval_paths = [f(x0[0], x0[1])]  # 함수 값 저장 리스트
    for i in range(Maxlter):
        grad = np.array([grad_f_x(*x0), grad_f_y(*x0)])  # 현재 점에서의 기울기 계산
        x1 = x0 - learn_rate * grad  # 다음 점 계산
        fval = f(*x1)  # 다음 점에서의 함수 값 계산
        if verbose:
            print(i, x1, fval)  # 현재 반복, 점, 함수 값 출력
        x0 = x1  # 현재 점 업데이트
        paths.append(x0)  # 경로에 추가
        fval_paths.append(fval)  # 함수 값에 추가
    paths = np.array(paths)  # 경로를 배열로 변환
    paths = np.array(np.matrix(paths).T)  # 경로를 전치
    fval_paths = np.array(fval_paths)  # 함수 값 배열로 변환
    return (x0, fval, paths, fval_paths)  # 최적화 결과 반환

# 초기 점 설정
x0 = np.array([-2, -2])
# 경사 하강법 실행
xopt, fopt, paths, fval_paths = steepest_descent_twod(f, grad_f_x, grad_f_y, x0)

# 최적화 경로 시각화
contour_with_path(f, x, y, paths, minima=np.array(([2], [2])))

