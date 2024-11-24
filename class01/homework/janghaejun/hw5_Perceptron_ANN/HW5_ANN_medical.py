import numpy as np # forlinear algebra
import matplotlib.pyplot as plt #for plotting things
import os
from PIL import Image# for reading images

# KerasLibraries <-CNN
import tensorflow as tf
from tensorflow.keras import datasets, layers, models, Model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, Flatten, Dense, Input, BatchNormalization, Concatenate, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
#from sklearn.metrics import classification_report, confusion_matrix# <-define evaluation metrics

mainDIR = os.listdir('./chest_xray')
print(mainDIR)
train_folder= './chest_xray/train/'
val_folder = './chest_xray/val/'
test_folder = './chest_xray/test/'
# train 
os.listdir(train_folder)
train_n = train_folder+'NORMAL/'
train_p = train_folder+'PNEUMONIA/'
#Normal pic 
print(len(os.listdir(train_n)))
rand_norm= np.random.randint(0,len(os.listdir(train_n)))
norm_pic = os.listdir(train_n)[rand_norm]
print('normal picture title: ',norm_pic)
norm_pic_address = train_n+norm_pic
#Pneumonia
rand_p = np.random.randint(0,len(os.listdir(train_p)))
sic_pic = os.listdir(train_p)[rand_norm]
sic_address = train_p+sic_pic
print('pneumonia picture title:', sic_pic)
# Load the images
norm_load =Image.open(norm_pic_address)
sic_load =Image.open(sic_address)
#Let's plt these images
f=plt.figure(figsize=(10,6))
a1 =f.add_subplot(1,2,1)
img_plot =plt.imshow(norm_load)
a1.set_title('Normal')
a2 =f.add_subplot(1, 2, 2)
img_plot =plt.imshow(sic_load)
a2.set_title('Pneumonia(폐렴)')
plt.show()
# let's build the CNN model
#hw3
# 첫 번째 Conv2D 층과 MaxPooling2D 층
model = tf.keras.Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))  # 예시 input shape: (64, 64, 3)
model.add(MaxPooling2D(pool_size=(2, 2)))

# 두 번째 Conv2D 층과 MaxPooling2D 층
model.add(Conv2D(32, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Flatten 층
model.add(Flatten())

# 첫 번째 Dense 층
model.add(Dense(128, activation='relu'))

# 두 번째 Dense 층 (출력층)
model.add(Dense(1, activation='sigmoid'))  # 이진 분류인 경우 sigmoid, 다중 분류의 경우 softmax 사용

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy'],
)


# Fitting the CNN to the images
# The function ImageDataGenerator augments your image by iterating through image as your CNN is getting ready to process that image
train_datagen = ImageDataGenerator(rescale = 1./255,shear_range = 0.2,zoom_range = 0.2,horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255) #Image normalization.
training_set = train_datagen.flow_from_directory('./chest_xray/train',target_size = (64, 64),batch_size = 32,class_mode = 'binary')
validation_generator =test_datagen.flow_from_directory('./chest_xray/val/',target_size=(64, 64), batch_size=32, class_mode='binary')
test_set =test_datagen.flow_from_directory('./chest_xray/test', target_size=(64, 64), batch_size=32, class_mode='binary')

model.summary()

# 훈련 데이터셋 크기
train_size = len(training_set.filenames)
batch_size = 32  # flow_from_directory에서 설정한 배치 크기

# 검증 데이터셋 크기
val_size = len(validation_generator.filenames)

# steps_per_epoch와 validation_steps 계산
steps_per_epoch = train_size // batch_size
validation_steps = val_size // batch_size


medical_model = model.fit(
    training_set,                  # 훈련 데이터셋
    steps_per_epoch=steps_per_epoch,  # 에포크당 스텝 수 (훈련 데이터 샘플 수 / 배치 크기)
    epochs= 10,                     # 훈련할 에포크 횟수 (자유롭게 조정 가능)
    validation_data=validation_generator,  # 검증 데이터셋
    validation_steps=validation_steps  # 검증 데이터셋의 스텝 수
)

model.save('medical.h5')

model = tf.keras.models.load_model('./medical.h5')
predictions = model.predict(test_set, steps=len(test_set))
predictions = (predictions > 0.5).astype(int)
actual_labels = test_set.labels
cnt = 0
total = 0

print(f"actual || predict")
for i in range(len(actual_labels)):
    actual = 'NORMAL' 
    predicted = 'NORMAL'
    if actual_labels[i] == 0:
        pass
    else:
        actual = 'PNEUMONIA'
    if predictions[i] == 0:
        pass
    else:
        predicted = 'PNEUMONIA'
    if actual == predicted:
        cnt += 1
    print(f"{i+1} : {actual} || {predicted}")

percent = (cnt/len(actual_labels))*100
print(f"실제 데이터와 동일한 예측은 {cnt}개")
print(f"정확도는 {percent:.2f}%")

"""
학습중에 나온 에러(?) 나중에 수정할  것 => 짝수번 학습에서만 나옴(2,4,6,8)
UserWarning: Your input ran out of data; interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches. You may need to use the `.repeat()` function when building your dataset. self.gen.throw(typ, value, traceback)
Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence [[{{node IteratorGetNext}}]]
I tensorflow/core/framework/local_rendezvous.cc:405] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence[[{{node IteratorGetNext}}]]
"""