import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

# MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()

# 이미지 정규화
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0
class_names = ['0','1', '2', '3', '4', '5', '6', '7', '8', '9']  # class_names의 크기는 10이어야 하므로, 0도 추가

# 이미지 출력
plt.figure(figsize=(10,10))
for i in range(10):
    plt.subplot(3,4,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_train[i])
    plt.xlabel(class_names[f_label_train[i]])
plt.show()

# ANN 모델 정의
model = tf.keras.Sequential()
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(128, activation='relu'))  
model.add(tf.keras.layers.Dense(64, activation='relu'))   
model.add(tf.keras.layers.Dense(10, activation='softmax')) 

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy'],
)

# 모델 훈련
model.fit(f_image_train, f_label_train, epochs=10, batch_size=10)
model.summary()
model.save('mnist.h5')  # 모델 저장 (저장된 파일 이름은 mnist.h5)

# 다른 코드에서 모델 불러오기
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2

# 저장된 모델 불러오기
model = tf.keras.models.load_model('mnist.h5')  # 수정: 파일 경로를 'mnist.h5'로 변경

# Fashion MNIST 데이터셋 로드
mnist = tf.keras.datasets.mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test / 255.0

# 예측 및 출력
num = 10
predict = model.predict(f_image_train[:num])

# 실제 라벨과 예측 결과 출력
print(f_label_train[:num])  
print(" * Prediction, ", np.argmax(predict, axis=1))

