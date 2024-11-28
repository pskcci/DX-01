import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense
from tensorflow.keras.datasets import mnist

# 1. 데이터 로드
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

# 2. 데이터 전처리
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0  # 0-255 값을 0-1로 정규화

# 3. 모델 정의 (Sequential 모델로 층을 쌓기)
model = Sequential()

# Flatten: 28x28 이미지를 1차원 벡터로 변환
model.add(Flatten(input_shape=(28, 28)))

# Dense: fully connected (완전 연결) 층
 # 첫 번째 은닉층 (64개의 뉴런)
model.add(Dense(64, activation='relu'))   # 두 번째 은닉층 (64개의 뉴런)
model.add(Dense(10, activation='softmax'))  # 출력층 (10개의 클래스)

# 4. 모델 컴파일 
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',  # one-hot 인코딩으로 0~9까지 숫자를 바꿔주는함수
    metrics=['accuracy']  # 정확도를 평가 지표로 사용
)

# 5. 모델 학습
model.fit(f_image_train, f_label_train, epochs=10, batch_size=32)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(f_image_test, f_label_test, verbose=2)
print(f"테스트 정확도: {test_acc}")

# 7. 모델 저장
model.save('mnist.h5')

# 8. 예측 예시
num = 10
predict = model.predict(f_image_test[:num])
print(f"실제 레이블: {f_label_test[:num]}")
print("예측 결과:", np.argmax(predict, axis=1))