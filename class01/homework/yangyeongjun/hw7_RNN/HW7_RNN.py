import FinanceDataReader as fdr
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, SimpleRNN, GRU, LSTM, Dropout

def MinMaxScaler(data):
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    return numerator / (denominator + 1e-7)

# 삼성전자 주식 데이터
df = fdr.DataReader('005930', '2018-05-04', '2020-01-22')
dfx = df[['Open', 'High', 'Low', 'Volume', 'Close']]
dfx = MinMaxScaler(dfx)
dfy = dfx[['Close']]
dfx = dfx[['Open', 'High', 'Low', 'Volume']]

x = np.array(dfx)
y = np.array(dfy)

# 범위를 0 ~ 1로 normalized
window_size = 10
data_x = []
data_y = []

for i in range(len(y) - window_size):
    _x = x[i:i + window_size]
    _y = y[i + window_size]
    data_x.append(_x)
    data_y.append(_y)

train_size = int(len(data_y) * 0.7)
val_size = int(len(data_y) * 0.2)
train_x = np.array(data_x[0:train_size])
train_y = np.array(data_y[0:train_size])
val_x = np.array(data_x[train_size:train_size + val_size])
val_y = np.array(data_y[train_size:train_size + val_size])

test_size = len(data_y) - train_size - val_size
test_x = np.array(data_x[train_size + val_size:len(data_x)])
test_y = np.array(data_y[train_size + val_size:len(data_y)])

# RNN 모델 생성 및 학습
rnn_model = Sequential()
rnn_model.add(SimpleRNN(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
rnn_model.add(Dropout(0.1))
rnn_model.add(SimpleRNN(units=20, activation='tanh'))
rnn_model.add(Dropout(0.1))
rnn_model.add(Dense(units=1))
rnn_model.compile(optimizer='adam', loss='mean_squared_error')
rnn_history = rnn_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)

# GRU 모델 생성 및 학습
gru_model = Sequential()
gru_model.add(GRU(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
gru_model.add(Dropout(0.1))
gru_model.add(GRU(units=20, activation='tanh'))
gru_model.add(Dropout(0.1))
gru_model.add(Dense(units=1))
gru_model.compile(optimizer='adam', loss='mean_squared_error')
gru_history = gru_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)

# LSTM 모델 생성 및 학습
lstm_model = Sequential()
lstm_model.add(LSTM(units=20, activation='tanh', return_sequences=True, input_shape=(10, 4)))
lstm_model.add(Dropout(0.1))
lstm_model.add(LSTM(units=20, activation='tanh'))
lstm_model.add(Dropout(0.1))
lstm_model.add(Dense(units=1))
lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_history = lstm_model.fit(train_x, train_y, validation_data=(val_x, val_y), epochs=70, batch_size=30, verbose=0)

# 예측값 생성
rnn_predictions = rnn_model.predict(test_x)
gru_predictions = gru_model.predict(test_x)
lstm_predictions = lstm_model.predict(test_x)

# 그래프 그리기
plt.figure(figsize=(15, 8))

# 실제 주가 값
plt.plot(test_y, label='Actual Price', color='black')

# RNN 예측 값
plt.plot(rnn_predictions, label='RNN Predicted Price', color='blue')

# GRU 예측 값
plt.plot(gru_predictions, label='GRU Predicted Price', color='red')

# LSTM 예측 값
plt.plot(lstm_predictions, label='LSTM Predicted Price', color='green')

plt.title('Stock Price Prediction Comparison (RNN vs GRU vs LSTM)')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()

# 그래프 저장
plt.savefig('stock_price_prediction_comparison.png', dpi=300)  # 고해상도로 저장
plt.show()  # 화면에 그래프 표시
