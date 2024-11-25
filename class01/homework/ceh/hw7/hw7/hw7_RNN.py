import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout
from sklearn.model_selection import train_test_split

# 범위를 0 ~ 1 로 normalized
def MinMaxScaler(data):
    # 최솟값과 최댓값을 이용하여 0 ~ 1 값으로 변환
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # 0으로 나누기 에러가 발생하지 않도록 매우 작은 값(1e-7)을 더해서 나눔
    return numerator / (denominator + 1e-7)

# 삼성전자 주식 데이터 (기존 데이터 기간 유지)
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')  # 기존 날짜 유지
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx_values = MinMaxScaler(dfx[['Open', 'High', 'Low', 'Volume']].values)  # OHLV 데이터 정규화
dfy_values = MinMaxScaler(dfx[['Close']].values)  # 종가 데이터 정규화

# 윈도우 크기 설정
window_size = 10
data_x = []
data_y = []

# 데이터 생성: window_size 만큼 이전 데이터를 사용하여 다음 날 종가 예측
for i in range(len(dfy_values) - window_size):
    _x = dfx_values[i: i + window_size]  # X 값: window_size 만큼의 데이터
    _y = dfy_values[i + window_size]     # Y 값: 해당 윈도우 다음 날 종가
    data_x.append(_x)
    data_y.append(_y)

# 배열로 변환
data_x = np.array(data_x)
data_y = np.array(data_y)

# 훈련 데이터와 검증 데이터 분리
train_x, val_x, train_y, val_y = train_test_split(data_x, data_y, test_size=0.2, shuffle=False)

# RNN 모델
model_rnn = Sequential()
model_rnn.add(SimpleRNN(units=20, activation='tanh', return_sequences=True, input_shape=(window_size, 4)))
model_rnn.add(Dropout(0.1))
model_rnn.add(SimpleRNN(units=20, activation='tanh'))
model_rnn.add(Dropout(0.1))
model_rnn.add(Dense(units=1))
model_rnn.compile(optimizer='adam', loss='mean_squared_error')

history_rnn = model_rnn.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)

# GRU 모델
model_gru = Sequential()
model_gru.add(GRU(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
model_gru.add(Dropout(0.1))
model_gru.add(GRU(units=20, activation='tanh'))
model_gru.add(Dropout(0.1))
model_gru.add(Dense(units=1))
model_gru.compile(optimizer='adam', loss='mean_squared_error')

history_gru = model_gru.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)

# LSTM 모델
model_lstm = Sequential()
model_lstm.add(LSTM(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
model_lstm.add(Dropout(0.1))
model_lstm.add(LSTM(units=20, activation='tanh'))
model_lstm.add(Dropout(0.1))
model_lstm.add(Dense(units=1))
model_lstm.compile(optimizer='adam', loss='mean_squared_error')

history_lstm = model_lstm.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30)

# 모델별 예측값을 구하기
rnn_predictions = model_rnn.predict(val_x)
gru_predictions = model_gru.predict(val_x)
lstm_predictions = model_lstm.predict(val_x)

# 예측 결과를 원래 스케일로 되돌리기 (정규화된 값 복원)
def invert_scaler(scaled_data, original_data):
    return scaled_data * (np.max(original_data) - np.min(original_data)) + np.min(original_data)

# 예측값을 원래 스케일로 복원
rnn_predictions_original = invert_scaler(rnn_predictions, dfx['Close'].values)
gru_predictions_original = invert_scaler(gru_predictions, dfx['Close'].values)
lstm_predictions_original = invert_scaler(lstm_predictions, dfx['Close'].values)
val_y_original = invert_scaler(val_y, dfx['Close'].values)

# 그래프를 그리기 위해 실제값과 예측값을 하나의 그래프에 시각화
plt.figure(figsize=(14, 8))

# 실제 종가 값
plt.plot(df.index[-len(val_y_original):], val_y_original, label='실제 종가', color='black', linewidth=2)

# RNN 예측값
plt.plot(df.index[-len(rnn_predictions_original):], rnn_predictions_original, label='RNN 예측 종가', color='blue', linewidth=2)

# GRU 예측값
plt.plot(df.index[-len(gru_predictions_original):], gru_predictions_original, label='GRU 예측 종가', color='green', linewidth=2)

# LSTM 예측값
plt.plot(df.index[-len(lstm_predictions_original):], lstm_predictions_original, label='LSTM 예측 종가', color='red', linewidth=2)

# 그래프 제목과 레이블 설정
plt.title('SEC stock price prediction', fontsize=16)
plt.xlabel('날짜', fontsize=14)
plt.ylabel('종가', fontsize=14)
plt.legend()

# 그래프 표시
plt.show()
