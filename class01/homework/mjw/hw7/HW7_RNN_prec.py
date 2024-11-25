import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout


def MinMaxScaler(data):
    # 최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환"
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)


df = fdr.DataReader('005930','2022-11-01','2024-11-20')
dfx = df[['Open','High','Low','Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]

x = dfx.values.tolist() # open, high, log, volume, 데이터
y = dfy.values.tolist() # close 데이터


window_size = 10
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]
    # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)




# Open, Close, Low, High, Volume 데이터를 따로 준비했다고 가정
open_data = np.array([item[0] for item in x])  # Open
high_data = np.array([item[1] for item in x])  # High
low_data = np.array([item[2] for item in x])   # Low
volume_data = np.array([item[3] for item in x])  # Volume
close_data = np.array([item[0] for item in y])  # Close (y에는 Close 데이터만 있음)

# 예시로 날짜 인덱스를 생성
date_range = pd.date_range(start='2022-11-01', periods=len(dfx), freq='B')  # Business day로 생성
df['Date'] = date_range  # Date 컬럼 추가
dates = df['Date']

# 그래프 크기 설정
plt.figure(figsize=(10, 10))

# 첫 번째 서브플롯 (Open)
plt.subplot(3, 1, 1)
plt.plot(dates, open_data, label='Open Price', color='blue')
plt.title('Open Price')
plt.legend()
plt.xticks([])  # x축 날짜 회전
plt.grid(False)

# 두 번째 서브플롯 (Close)
plt.subplot(3, 1, 2)
plt.plot(dates, close_data, label='Close Price', color='green')
plt.title('Close Price')
plt.legend()
plt.xticks([])  # x축 날짜 회전
plt.grid(False)

# 세 번째 서브플롯 (Low, High, Volume)
plt.subplot(3, 1, 3)
plt.plot(dates, low_data, label='Low', color='orange')
plt.plot(dates, high_data, label='High', color='green')
plt.plot(dates, volume_data, label='Volume', color='red')
plt.title('Low, High, Volume')
plt.legend()
plt.xticks(rotation=45)  # x축 날짜 회전
plt.grid(False)

# 레이아웃을 자동으로 조정하여 겹치지 않게 만들기
plt.tight_layout()

# 그래프 표시
plt.savefig('./Stock.png')
plt.show()

# 훈련데이터셋
train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0 : train_size])
train_y = np.array(data_y[0 : train_size])
val_x = np.array(data_x[train_size:train_size+val_size])
val_y = np.array(data_y[train_size:train_size+val_size])
test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size+val_size: len(data_x)])
test_y = np.array(data_y[train_size+val_size: len(data_y)])
print('훈련 데이터의 크기 :', train_x.shape, train_y.shape)
print('검증 데이터의 크기 :', val_x.shape, val_y.shape)
print('테스트 데이터의 크기 :', test_x.shape, test_y.shape)


# RNN 모델
model = Sequential()
model.add(SimpleRNN(units=20, activation='tanh',return_sequences=True,input_shape=(10, 4)))
model.add(Dropout(0.1))
model.add(SimpleRNN(units=20, activation='tanh'))
model.add(Dropout(0.1))
model.add(Dense(units=1))
model.summary()
model.compile(optimizer='adam',loss='mean_squared_error')
history = model.fit(train_x, train_y, validation_data = (val_x, val_y),epochs=70, batch_size=30)
print("RNN done")

# RNN 모델 학습 및 예측
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(units=20, activation='tanh'))
rnn_model.add(Dropout(0.1))
rnn_model.add(Dense(units=1))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)
rnn_pred = rnn_model.predict(test_x)

# GRU 모델 학습 및 예측
gru_model = Sequential()
gru_model.add(GRU(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
gru_model.add(Dropout(0.1))
gru_model.add(GRU(units=20, activation='tanh'))
gru_model.add(Dropout(0.1))
gru_model.add(Dense(units=1))
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)
gru_pred = gru_model.predict(test_x)

# LSTM 모델 학습 및 예측
lstm_model = Sequential()
lstm_model.add(LSTM(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(units=20, activation='tanh'))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)
lstm_pred = lstm_model.predict(test_x)

# 실제값
actual = test_y

# 그래프 출력
plt.figure(figsize=(10, 6))
plt.plot(actual, label='Actual', color='red')  # 실제 값
plt.plot(rnn_pred, label='predicted (rnn)', color='blue')  # RNN 예측
plt.plot(gru_pred, label='predicted (gru)', color='orange')  # GRU 예측
plt.plot(lstm_pred, label='predicted (lstm)', color='green')  # LSTM 예측

# 그래프 꾸미기
plt.title('SEC stock price prediction')
plt.xlabel('time')
plt.ylabel('stock price')
plt.legend()
plt.savefig('./train.png')
plt.show()