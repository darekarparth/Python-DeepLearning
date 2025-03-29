'''
 Develop an algorithm for predicting the number of car sales per month. The dataset will
be downloaded automatically using Pandas, but you can learn more about the dataset in
Appendix.

Formulate the problem taking a window of the last few months of data to predict the
current month’s data.

To achieve this, a new function named split_sequence() is provided blow that will split
the input sequence into windows of data appropriate for fitting a LSTM model.

Extend the above code with the Sequential model and LSTM to develop an algorithm for
predicting the number of car sales based on the input sequence [18024.0, 16722.0, 14385.0,
21342.0, 17180.0].

Appendix

Car Sales Dataset Description

Monthly car sales in Quebec 1960-1968.
Source: Time Series Data Library (citing: Abraham & Ledolter (1983))
Car Sales Dataset
"Month","Sales"
"1960-01",6550
"1960-02",8728
"1960-03",12026
"1960-04",14395
"1960-05",14587
"1960-06",13791
"1960-07",9498
"1960-08",8251
"1960-09",7049
"1960-10",9545
"1960-11",9364
"1960-12",8456
"1961-01",7237
"1961-02",9374
"1961-03",11837
"1961-04",13784
"1961-05",15926
"1961-06",13821
"1961-07",11143
.
.
.

'''

from numpy import sqrt
from numpy import asarray
from pandas import read_csv
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM
# split a univariate sequence into samples

def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
    # find the end of this pattern
        end_ix = i + n_steps
    # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
    # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return asarray(X), asarray(y)
# load the dataset
path = 'https://raw.githubusercontent.com/jbrownlee/Datasets/master/monthly-carsales.csv'
df = read_csv(path, header=0, index_col=0)
# retrieve the values
values = df.values.astype('float32')
# specify the window size
n_steps = 5
# split into samples
X, y = split_sequence(values, n_steps)
# reshape into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], 1))
# split into train/test
n_test = 12
X_train, X_test, y_train, y_test = X[:-n_test], X[-n_test:], y[:-n_test], y[
n_test:]
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)
# define model
model = Sequential()
model.add(LSTM(100, activation='relu', kernel_initializer='he_normal',
input_shape=(n_steps,1)))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1))
# compile the model
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
# fit the model
model.fit(X_train, y_train, epochs=350, batch_size=32, verbose=2,
validation_data=(X_test, y_test))
# evaluate the model
mse, mae = model.evaluate(X_test, y_test, verbose=0)
print('MSE: %.3f, RMSE: %.3f, MAE: %.3f' % (mse, sqrt(mse), mae))
# make a prediction
row = asarray([18024.0, 16722.0, 14385.0, 21342.0, 17180.0]).reshape((1, n_steps,
1))
yhat = model.predict(row)
print('Predicted: %.3f' % (yhat))