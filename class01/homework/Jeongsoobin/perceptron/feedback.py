import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.utils import to_categorical

# CIFAR-10 데이터셋 로드
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 데이터 전처리
x_train, x_test = x_train / 255.0, x_test / 255.0  # 0-255 범위 값을 0-1로 정규화
y_train = to_categorical(y_train, 10)  # 원-핫 인코딩
y_test = to_categorical(y_test, 10)

# CNN 모델 구축
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 첫 번째 합성곱층
model.add(BatchNormalization())  # 배치 정규화 추가
model.add(MaxPooling2D((2, 2)))  # 최대 풀링층
model.add(Conv2D(64, (3, 3), activation='relu'))  # 두 번째 합성곱층
model.add(BatchNormalization())  # 배치 정규화 추가
model.add(MaxPooling2D((2, 2)))  # 최대 풀링층
model.add(Flatten())  # 1차원 배열로 변환
model.add(Dense(64, activation='relu'))  # 은닉층
model.add(Dropout(0.5))  # 드롭아웃 추가 (50% 확률로 뉴런 비활성화)
model.add(Dense(10, activation='softmax'))  # 출력층 (10개의 클래스)

# 모델 컴파일
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 모델 학습
model.fit(x_train, y_train, epochs=10, batch_size=64)

# 모델 평가
test_loss, test_acc = model.evaluate(x_test, y_test)
print(f"테스트 정확도: {test_acc}")

# 예측 함수 정의
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

    return predicted_class, prediction

# 사용자 인터페이스로 정답을 학습에 반영하는 부분
def interactive_learning(image_path, x_train, y_train):
    predicted_class, prediction = predict_image(image_path)
    print(f"예측된 클래스: {predicted_class}")
    
    # 사용자에게 정답을 입력받음
    correct_class = input(f"정답을 입력하세요 (예: {predicted_class}): ")
    
    # 정답이 틀렸을 경우 모델 학습 데이터에 추가
    if correct_class != predicted_class:
        # CIFAR-10 class_names의 인덱스를 가져오기 위해 추가
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
        correct_class_index = class_names.index(correct_class)
        
        # 이미지를 정답으로 라벨링하여 학습 데이터에 추가
        img = cv2.imread(image_path)
        img = cv2.resize(img, (32, 32))  # CIFAR-10 이미지 크기에 맞게 리사이즈
        img = img / 255.0  # 0-1 범위로 정규화
        img = np.expand_dims(img, axis=0)  # 배치 차원 추가
        
        # 모델에 새로 추가된 이미지를 학습 데이터로 추가
        x_train = np.append(x_train, img, axis=0)
        y_train = np.append(y_train, to_categorical(np.array([correct_class_index]), 10), axis=0)
        
        # 모델을 추가 학습 (fine-tuning)
        model.fit(x_train, y_train, epochs=1, batch_size=64)
        
        print("모델이 업데이트되었습니다.")
    
    else:
        print("예측이 정확했습니다.")
    
    return x_train, y_train

# 예시로 로컬 파일을 예측하는 코드
image_path = '/home/intel/다운로드/dog.jpeg'  # 예측할 이미지 파일 경로 지정
x_train, y_train = interactive_learning(image_path, x_train, y_train)

# 이미지 출력
img = Image.open(image_path)
plt.imshow(img)
plt.title(f"predict_animal: {predicted_class}")
plt.show()
