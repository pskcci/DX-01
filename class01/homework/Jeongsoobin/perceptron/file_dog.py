import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# 1. CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 2. 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0  # 0-255 범위 값을 0-1로 정규화
y_train = to_categorical(y_train, 10)  # 원-핫 인코딩
y_test = to_categorical(y_test, 10)

# 3. CNN 모델 구축
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 첫 번째 합성곱층
model.add(MaxPooling2D((2, 2)))  # 최대 풀링층
model.add(Conv2D(64, (3, 3), activation='relu'))  # 두 번째 합성곱층
model.add(MaxPooling2D((2, 2)))  # 최대 풀링층
model.add(Flatten())  # 1차원 배열로 변환
model.add(Dense(64, activation='relu'))  # 은닉층
model.add(Dense(10, activation='softmax'))  # 출력층 (10개의 클래스)

# 4. 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 5. 모델 학습
model.fit(x_train, y_train, epochs=20, batch_size=64)

# 6. 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc}")

# 7. 예측 함수 정의
def predict_image(image_path):
    # 이미지를 파일에서 불러오기
    img = cv2.imread(image_path)
    img = cv2.resize(img, (32, 32))  # CIFAR-10 이미지 크기에 맞게 리사이즈
    img = img / 255.0  # 0-1 범위로 정규화
    img = np.expand_dims(img, axis=0)  # 배치 차원 추가
    
    # 예측
    prediction = model.predict(img)
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    predicted_class = class_names[np.argmax(prediction)]

    return predicted_class

# 8. 예시로 로컬 파일을 예측하는 코드
image_path = '/home/intel/다운로드/dog.jpeg'  # 예측할 이미지 파일 경로 지정
predicted_class = predict_image(image_path)
print(f"예측된 클래스: {predicted_class}")

img = Image.open(image_path)
plt.imshow(img, cmap='gray')
plt.title(f"predict_animal: {predicted_class}")
plt.show()