import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
    """최솟값과 최대값을 이용하여 0 ~ 1 값으로 변환"""
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값 (1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

df = fdr.DataReader('005930', '2020-01-01', '2024-11-14')
dfx = df[['Open','High','Low','Volume','Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open','High','Low','Volume']]


# 두 데이터를 리스트 형태로 저장
x = dfx.values.tolist() # open, high, log, volume, 데이터
y = dfy.values.tolist() # close 데이터

#ex) 1월 1일 ~ 1월 10일까지의 OHLV 데이터로 1월 11일 종가 (Close) 예측
#ex) 1월 2일 ~ 1월 11일까지의 OHLV 데이터로 1월 12일 종가 (Close) 예측
window_size = 120
data_x = []
data_y = []
for i in range(len(y) - window_size):
    _x = x[i : i + window_size] # 다음 날 종가(i+windows_size)는 포함되지 않음
    _y = y[i + window_size]
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

# train_x의 차원 확인
print("train_x shape:", train_x.shape)

# train_x가 1차원인 경우 3차원으로 변환
if len(train_x.shape) == 1:
    train_x = train_x.reshape(-1, 1, 1)  # (samples, time steps, features) 형태로 변환
elif len(train_x.shape) == 2:
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], 1)  # (samples, time steps, features) 형태로 변환

# test_x도 같은 방식으로 변환
print("test_x shape:", test_x.shape)
if len(test_x.shape) == 1:
    test_x = test_x.reshape(-1, 1, 1)  # (samples, time steps, features) 형태로 변환
elif len(test_x.shape) == 2:
    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], 1)  # (samples, time steps, features) 형태로 변환

# RNN 모델 정의
rnn_model = Sequential()
rnn_model.add(SimpleRNN(50, input_shape=(train_x.shape[1], train_x.shape[2])))
rnn_model.add(Dense(1))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')

# RNN 모델 훈련
rnn_model.fit(train_x, train_y, epochs=70, batch_size=30)

# GRU 모델 정의
gru_model = Sequential()
gru_model.add(GRU(50, input_shape=(train_x.shape[1], train_x.shape[2])))
gru_model.add(Dense(1))
gru_model.compile(optimizer='adam', loss='mean_squared_error')

# GRU 모델 훈련
gru_model.fit(train_x, train_y, epochs=70, batch_size=30)

# LSTM 모델 정의
lstm_model = Sequential()
lstm_model.add(LSTM(50, input_shape=(train_x.shape[1], train_x.shape[2])))
lstm_model.add(Dense(1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')

# LSTM 모델 훈련
lstm_model.fit(train_x, train_y, epochs=70, batch_size=30)

# 예측 결과 시각화
# RNN 모델 예측
rnn_predicted_y = rnn_model.predict(test_x).flatten()

# GRU 모델 예측
gru_predicted_y = gru_model.predict(test_x).flatten()

# LSTM 모델 예측
lstm_predicted_y = lstm_model.predict(test_x).flatten()

# 실제 종가
actual_y = test_y.flatten()

# MinMaxScaler 역변환 함수 추가
def inverse_minmax_scaler(scaled_data, original_data):
    """MinMaxScaler 역변환"""
    min_val = np.min(original_data)
    max_val = np.max(original_data)
    return scaled_data * (max_val - min_val) + min_val

# 예측 결과 역변환
rnn_predicted_y = inverse_minmax_scaler(rnn_predicted_y, df['Close'])
gru_predicted_y = inverse_minmax_scaler(gru_predicted_y, df['Close'])
lstm_predicted_y = inverse_minmax_scaler(lstm_predicted_y, df['Close'])
actual_y = inverse_minmax_scaler(actual_y, df['Close'])

# 시각화
plt.figure(figsize=(14, 7))
plt.plot(actual_y, color='blue', label='Actual C.P')
plt.plot(rnn_predicted_y, color='red', label='RNN Predicted C.P')
plt.plot(gru_predicted_y, color='green', label='GRU Predicted C.P')
plt.plot(lstm_predicted_y, color='orange', label='LSTM Predicted C.P')
plt.title('Actual C.P vs Model Predicted C.P')
plt.xlabel('Time')
plt.ylabel('C.P')

# x축을 일자별로 표현
step = max(1, len(actual_y) // 10)  # 10개 간격으로 눈금 수 설정
ticks = np.arange(0, len(actual_y), step)
labels = df.index[train_size + val_size::step].date

plt.xticks(ticks=ticks, labels=labels[:len(ticks)], rotation=45)  # x축 눈금 조정 및 일자 레이블 추가

# y축을 원 기준으로 설정
plt.gca().set_ylim(bottom=0)  # y축의 하한을 0으로 설정


plt.legend()
plt.show()



