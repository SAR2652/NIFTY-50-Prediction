import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping


def main():
    X_train = np.load("X_train.npy")
    X_val = np.load("X_val.npy")
    X_test = np.load("X_test.npy")
    y_train = np.load("y_train.npy")
    y_val = np.load("y_val.npy")
    y_test = np.load("y_test.npy")
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(256, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer = 'adam', loss = 'mse',
                  metrics = [RootMeanSquaredError(), MeanAbsoluteError()])
    
    history = model.fit(X_train, y_train, batch_size=32,
                        epochs=input_dim * 2, validation_data=(X_val, y_val),
                        callbacks=[EarlyStopping()])

    plt.figure(figsize=(8, 5))
    pd.DataFrame(history.history).plot()
    plt.savefig('metric_plots.pdf')
    plt.show()

    X_test_tensor = tf.constant(X_test)
    y_pred = model.predict(X_test_tensor)

    y_pred_np = y_pred.numpy()

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_np))
    print(f'Root Mean Squared Error = {rmse}')

    r2 = r2_score(y_test, y_pred_np)
    print(f'R^2 Score = {r2}')


if __name__ == '__main__':
    main()
