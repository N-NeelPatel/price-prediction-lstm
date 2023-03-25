import os
import tensorflow as tf
from tensorflow.python.keras import Sequential
from tensorflow.python.keras.layers import Dropout, Dense, LSTM


class LongShortTermMemory:
    def __init__(self, project_folder):
        self.project_folder = project_folder

    def get_defined_metrics(self):
        defined_metrics = [tf.keras.metrics.MeanSquaredError(name="MSE")]
        return defined_metrics

    def get_callback(self):
        callback = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=3, mode="min", verbose=1
        )
        return callback

    def create_model(self, x_train):
        model = Sequential()
        # 1st layer with Dropout regularisation
        model.add(
            LSTM(units=100, return_sequences=True, input_shape=(x_train.shape[1], 1))
        )
        model.add(Dropout(0.2))
        # 2nd LSTM layer
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.2))
        # 3rd LSTM layer
        model.add(LSTM(units=50, return_sequences=True))
        model.add(Dropout(0.5))
        # 4th LSTM layer
        model.add(LSTM(units=50))
        model.add(Dropout(0.5))
        # Dense layer that specifies an output of one unit
        model.add(Dense(units=1))
        model.summary()
        return model
