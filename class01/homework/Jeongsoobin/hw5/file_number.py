import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# 1. MNIST 모델 불러오기 (이전과 동일)
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

# 이미지 전처리
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# 모델 정의 및 학습 (같은 모델 구조로 학습)
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')  # 10개의 클래스 (0~9)
])

# 모델 컴파일
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',#one-hot 인코딩으로 0~9까지 숫자를 바꿔주는함수
              metrics=['accuracy'])

# 모델 학습
model.fit(f_image_train, f_label_train, epochs=5)

# 2. 이미지 파일 로드 및 전처리
def load_and_preprocess_image(image_path):
    # 이미지 파일 로드
    img = Image.open(image_path).convert('L')  # 흑백 이미지로 변환
    img = img.resize((28, 28))  # 28x28 크기로 리사이즈
    img_array = np.array(img)  # 이미지 배열로 변환
    
    img_array = img_array / 255.0  # 0-255에서 0-1로 정규화
    img_array = img_array.reshape(1, 28, 28, 1)  # 모델 입력 형태로 변환
    return img_array

# 3. 예측 함수
def predict_digit(image_path):
    img_array = load_and_preprocess_image(image_path)  # 이미지 전처리
    prediction = model.predict(img_array)  # 예측 수행
    predicted_label = np.argmax(prediction)  # 예측된 숫자 라벨
    return predicted_label

# 4. 예측 실행
image_path = '/home/intel/다운로드/2.png'  # 예시 이미지 경로
predicted_digit = predict_digit(image_path)

# 5. 결과 출력
print(f"predict_number: {predicted_digit}")

# 이미지를 출력해서 확인
img = Image.open(image_path)
plt.imshow(img, cmap='gray')
plt.title(f"predict_number: {predicted_digit}")
plt.show()