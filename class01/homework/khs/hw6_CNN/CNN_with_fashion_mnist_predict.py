import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
#import cv2

model = tf.keras.models.load_model('./fashion_mnist.h5')
fashion_mnist = tf.keras.datasets.fashion_mnist
(f_image_train, f_label_train), (f_image_test, f_label_test) = fashion_mnist.load_data()
f_image_train, f_image_test = f_image_train / 255.0, f_image_test /255.0

num = 10
predict = model.predict(f_image_test[:num])
print(f_label_test[:num])
print(" * prediction, ", np.argmax(predict, axis=1))

# 실제 레이블과 예측 레이블 출력
print("Actual labels: ", f_label_test[:num])
print("Predicted labels: ", np.argmax(predict, axis=1))

# 정확도 계산
correct_predictions = np.sum(np.argmax(predict, axis=1) == f_label_test[:num])
accuracy = correct_predictions / num
print(f"Accuracy: {accuracy * 100:.2f}%")

# 예측 결과 시각화
plt.figure(figsize=(10, 10))
for i in range(num):
    plt.subplot(2, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(f_image_test[i], cmap=plt.cm.binary)
    plt.xlabel(f"True: {f_label_test[i]}, Pred: {np.argmax(predict[i])}")
plt.show()