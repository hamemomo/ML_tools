import numpy as np
import pandas as pd
from keras.models import Model
from keras.layers import Input,Dense,Flatten
from keras.layers import LSTM
import keras

df_train = pd.read_csv('./Train_X.csv',encoding= 'cp950')
df_test = pd.read_csv('./Test_X.csv',encoding= 'cp950')

df_train = df_train.sort_values('日期')
df_test = df_test.sort_values('日期')

stock_list_train = (df_train['股票代號'].unique()).tolist()
stock_list_test = (df_test['股票代號'].unique()).tolist()
data_train = {}
data_test ={}
for name in stock_list_train:
    data_train[name]=df_train[df_train['股票代號']==name].loc[:,['開盤價', '最高價','最低價','收盤價','成交量.股.']]
    data_test[name]=df_test[df_test['股票代號']==name].loc[:,['開盤價', '最高價','最低價','收盤價','成交量.股.']]

#定義正規化函式
def normalize(train):
    train_norm = train.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
    return train_norm

#定義餵入的feature跟label序列
def train_(df, ref_day, predict_day):
    X_train, Y_train = [], []
    for i in range(df.shape[0]-predict_day-ref_day):
        X_train.append(np.array(df.iloc[i:i+ref_day,:]))
        Y_train.append(np.array(df.iloc[i+ref_day:i+ref_day+predict_day]['收盤價']))
    return np.array(X_train), np.array(Y_train)

def denormalize(train):
    denorm = train*(np.max(data_train['TWA00']['收盤價'])-np.min(data_train['TWA00']['收盤價']))+np.mean(data_train['TWA00']['收盤價'])
    return denorm

TWA00_train= normalize(data_train['TWA00'])
TWA00_test= normalize(data_test['TWA00'])
ref_days = 7
X_train,Y_train=train_(TWA00_train,ref_days,1)
X_test,Y_test=train_(TWA00_test,ref_days,1)

#split_boundary = int(X.shape[0] * 0.9)
'''
train_x = X[: split_boundary]
test_x = X[split_boundary:]
train_y = Y[: split_boundary]
test_y = Y[split_boundary:]
'''

# define model

units = 50
inputs1 = Input(shape=(X_train.shape[1], X_train.shape[2]))
lstm1=LSTM(units, activation='relu', recurrent_activation='hard_sigmoid',return_sequences=True)(inputs1)
lstm2=LSTM(units, activation='relu', recurrent_activation='hard_sigmoid',return_sequences=True)(lstm1)
lstm3=LSTM(units, activation='relu', recurrent_activation='hard_sigmoid',return_sequences=True)(lstm2)
final=LSTM(1, activation='tanh', recurrent_activation='hard_sigmoid',return_sequences = False)(lstm3)
adam = keras.optimizers.Adam(lr=0.01, beta_1=0.99, beta_2=0.999, amsgrad=False)

model = Model(inputs=inputs1, outputs=final)
model.compile(optimizer=adam, loss='mean_squared_error')

model.fit(X_train, Y_train, epochs = 100, batch_size = 20)

a = X_test[0].reshape(1,ref_days,5)

predict_y = model.predict(X_test)

Y_test = pd.DataFrame(Y_test)
test_y = denormalize(Y_test)
z = denormalize(predict_y)

import matplotlib.pyplot as plt
plt.plot(test_y, color = 'red', label = 'Real Price')  # 紅線表示真實股價
plt.plot(z, color = 'blue', label = 'Predicted Price')  # 藍線表示預測股價
plt.title('Price Prediction')
plt.xlabel('Time')
plt.ylabel('Stock Price')
plt.legend()
plt.show()