import pickle
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import load_model

X_test = np.load('X_test.npy')
y_test = np.load('y_test.npy')

X_test_tensor = tf.constant(X_test)
best_model = load_model('best_dnn.h5')
y_pred = best_model.predict(X_test_tensor)
y_pred_np = y_pred.reshape(-1, 1)


with open('target_mms.pkl', 'rb') as f:
    mms = pickle.load(f)

y_test_scaled = mms.inverse_transform(y_test.reshape(-1, 1))
y_pred_scaled = mms.inverse_transform(y_pred_np.reshape(-1, 1))

rmse = np.sqrt(mean_squared_error(y_test_scaled, y_pred_scaled))
print(f'Root Mean Squared Error = {rmse}')

r2 = r2_score(y_test_scaled, y_pred_scaled)
print(f'R^2 Score = {r2}')