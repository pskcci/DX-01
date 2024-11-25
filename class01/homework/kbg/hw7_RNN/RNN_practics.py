import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

print("hello world")

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
    # 최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2022-01-01', '2024-11-01')
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
    _y = y[i + window_size] # 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

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
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=20, activation='tanh',
return_sequences=True,
input_shape=(10, 4)))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(units=20, activation='tanh'))
rnn_model.add(Dropout(0.1))
rnn_model.add(Dense(units=1))
rnn_model.summary()
rnn_model.compile(optimizer='adam',
loss='mean_squared_error')
history = rnn_model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=70, batch_size=30)

with open('history_rnn', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# GRU 모델
gru_model = Sequential()
gru_model.add(GRU(units=20, activation='tanh',
return_sequences=True,
input_shape=(10, 4)))
gru_model.add(Dropout(0.1))
gru_model.add(GRU(units=20, activation='tanh'))
gru_model.add(Dropout(0.1))
gru_model.add(Dense(units=1))
gru_model.summary()
gru_model.compile(optimizer='adam',
loss='mean_squared_error')
history = gru_model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=70, batch_size=30)

with open('history_gru', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# LSTM 모델
lstm_model = Sequential()
lstm_model.add(LSTM(units=20, activation='tanh',
return_sequences=True,
input_shape=(10, 4)))
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(units=20, activation='tanh'))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(units=1))
lstm_model.summary()
lstm_model.compile(optimizer='adam',
loss='mean_squared_error')
history = lstm_model.fit(train_x, train_y, validation_data = (val_x, val_y), epochs=70, batch_size=30)

with open('history_lstm', 'wb') as file_pi:
    pickle.dump(history.history, file_pi)

# 예측을 수행하는 함수
def plot_predictions(model, test_x, test_y):
    # 예측값 생성
    predictions = model.predict(test_x)
    
    # 실제값과 예측값을 비교하는 그래프 그리기
    plt.figure(figsize=(12, 6))
    plt.plot(test_y, label="Actual Close", color='blue')  # 실제 종가
    plt.plot(predictions, label="Predicted Close", color='red')  # 예측된 종가
    plt.title(f"{model.name} Model - Actual vs Predicted")
    plt.xlabel("Time Step")
    plt.ylabel("Close Price")
    plt.legend()
    plt.show()

# # RNN 모델 예측 결과 시각화
# plot_predictions(rnn_model, test_x, test_y)
# # GRU 모델 예측 결과 시각화
# plot_predictions(gru_model, test_x, test_y)
# # LSTM 모델 예측 결과 시각화
# plot_predictions(lstm_model, test_x, test_y)


def plot_predictions_m(modelA, modelB, modelC, test_x, test_y):
    # 각 모델에 대해 예측값을 생성
    pred_A = modelA.predict(test_x)
    pred_B = modelB.predict(test_x)
    pred_C = modelC.predict(test_x)
    
    # 실제값을 파란색으로, 모델 예측값을 각각 빨강, 주황, 노랑으로 시각화
    plt.figure(figsize=(14, 8))
    
    # 실제 종가
    plt.plot(test_y, label="Actual Close", color='blue', linewidth=2)
    
    # 모델 A 예측값
    plt.plot(pred_A, label="Model A (RNN)", color='red', linewidth=1)
    
    # 모델 B 예측값
    plt.plot(pred_B, label="Model B (GRU)", color='orange', linewidth=1)
    
    # 모델 C 예측값
    plt.plot(pred_C, label="Model C (LSTM)", color='green', linewidth=1)
    
    # 그래프 제목과 레이블 설정
    plt.title("Actual vs Predicted Close Prices", fontsize=16)
    plt.xlabel("Time Step", fontsize=14)
    plt.ylabel("Close Price", fontsize=14)
    
    # 범례 추가
    plt.legend()
    
    # 그래프 표시
    plt.show()

plot_predictions_m(rnn_model, gru_model, lstm_model, test_x, test_y)
