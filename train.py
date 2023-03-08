import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense
from tensorflow.keras.metrics import RootMeanSquaredError, MeanAbsoluteError
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint


def main():
    df = pd.read_csv('data.csv')
    y = df['close'].values
    X = df.iloc[:, 1:].values
    mms = MinMaxScaler()
    t_mms = MinMaxScaler()
    X_scaled = mms.fit_transform(X)
    y_scaled = t_mms.fit_transform(y.reshape(-1, 1))
    X_train, X_val_test, y_train, y_val_test = train_test_split(X_scaled,
                                                                y_scaled,
                                                                test_size=0.4)
    X_val, X_test, y_val, y_test = train_test_split(X_val_test,
                                                    y_val_test,
                                                    test_size=0.5)
    with open('X_train.npy', 'wb') as f:
        np.save(f, X_train)
    with open('X_val.npy', 'wb') as f:
        np.save(f, X_val)
    with open('X_test.npy', 'wb') as f:
        np.save(f, X_test)
    with open('y_train.npy', 'wb') as f:
        np.save(f, y_train)
    with open('y_val.npy', 'wb') as f:
        np.save(f, y_val)
    with open('y_test.npy', 'wb') as f:
        np.save(f, y_test)
    with open('feature_mms.pkl', 'wb') as f:
        pickle.dump(mms, f)
    with open('target_mms.pkl', 'wb') as f:
        pickle.dump(t_mms, f)
    input_dim = X_train.shape[1]
    model = Sequential()
    model.add(Dense(128, input_shape=(input_dim,), activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='linear'))
    model.compile(optimizer='adam', loss='mse',
                  metrics=[RootMeanSquaredError(), MeanAbsoluteError()])
    filepath = "best_dnn.h5"
    checkpoint = ModelCheckpoint(filepath,
                                 monitor='val_root_mean_squared_error',
                                 verbose=1,
                                 save_best_only=True, mode='min')
    history = model.fit(X_train, y_train, batch_size=32,
                        epochs=input_dim * 2,
                        validation_data=(X_val, y_val),
                        callbacks=[checkpoint])

    plt.figure(figsize=(8, 5))
    pd.DataFrame(history.history).plot()
    plt.savefig('metric_plots.pdf')
    plt.show()


if __name__ == '__main__':
    main()
