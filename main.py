import pandas as pd
import os
import secrets
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from plotter import Plotter
from lstm import LongShortTermMemory
import numpy as np

TIME_STEPS = 3

pd.options.mode.chained_assignment = None

fields = ["timestamp", "Asset_ID", "Close"]
train_df = pd.read_csv(
    "C:/Users/neel1/Downloads/supplemental_train.csv",
    low_memory=False,
    dtype={
        "Asset_ID": "int8",
        "Count": "int32",
        "row_id": "int32",
        "Open": "float64",
        "High": "float64",
        "Low": "float64",
        "Close": "float64",
        "Volume": "float64",
        "VWAP": "float64",
    },
    nrows=1000000,
    usecols=fields,
)

train_df.dropna(axis=0, inplace=True)
print(train_df.head(3))

# Print the list of Assets
print(train_df["Asset_ID"].unique())

asset_id_map = [
    "Binance Coin",
    "Bitcoin",
    "Bitcoin Cash",
    "Cardano",
    "Dogecoin",
    "EOS.IO",
    "Ethereum",
    "Ethereum Classic",
    "IOTA",
    "Litecoin",
    "Maker",
    "Monero",
    "Stellar",
    "TRON",
]
asset_id_ticker_map = [
    "BNB",
    "BTC",
    "BCH",
    "ADA",
    "DOGE",
    "EOS",
    "ETH",
    "ETC",
    "MIOTA",
    "LTC",
    "MKR",
    "XMR",
    "XLM",
    "TRX",
]

current_asset_id = 4

# Let's try BTC
train_data_set = train_df[train_df["Asset_ID"] == current_asset_id]
del train_data_set["Asset_ID"]
print(train_data_set.head(3))

# 70% of the data for training, the rest for validation
training_data = train_data_set[: int(train_data_set.shape[0] * 0.7)].copy()
testing_data = train_data_set[int(train_data_set.shape[0] * 0.7) :].copy()
print(training_data.head(3))
print(testing_data.head(3))

training_data = training_data.set_index("timestamp")
testing_data = testing_data.set_index("timestamp")

min_max = MinMaxScaler(feature_range=(0, 1))
train_scaled = min_max.fit_transform(training_data)

print("mean:", train_scaled.mean(axis=0))
print("max", train_scaled.max())
print("min", train_scaled.min())
print("Std dev:", train_scaled.std(axis=0))

# Training Data Transformation
x_train = []
y_train = []
for i in range(TIME_STEPS, train_scaled.shape[0]):
    x_train.append(train_scaled[i - TIME_STEPS : i])
    y_train.append(train_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

total_data = pd.concat((training_data, testing_data), axis=0)
inputs = total_data[len(total_data) - len(testing_data) - TIME_STEPS :]
test_scaled = min_max.fit_transform(inputs)

# Testing Data Transformation
x_test = []
y_test = []
for i in range(TIME_STEPS, test_scaled.shape[0]):
    x_test.append(test_scaled[i - TIME_STEPS : i])
    y_test.append(test_scaled[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

print(x_train)
print(y_train)

STOCK_TICKER = asset_id_ticker_map[current_asset_id]
TODAY_RUN = datetime.today().strftime("%Y%m%d")
TOKEN = STOCK_TICKER + "_" + TODAY_RUN + "_" + secrets.token_hex(16)
PROJECT_FOLDER = os.path.join(os.getcwd(), TOKEN)

if not os.path.exists(PROJECT_FOLDER):
    os.makedirs(PROJECT_FOLDER)

plotter = Plotter(True, PROJECT_FOLDER, STOCK_TICKER, STOCK_TICKER, STOCK_TICKER)
plotter.plot_histogram_data_split(training_data, testing_data, 70)

# Train the model
EPOCHS = 100
BATCH_SIZE = 32
lstm = LongShortTermMemory(PROJECT_FOLDER)
model = lstm.create_model(x_train)
model.compile(
    optimizer="adam", loss="mean_squared_error", metrics=lstm.get_defined_metrics()
)
history = model.fit(
    x_train,
    y_train,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    validation_data=(x_test, y_test),
    callbacks=[lstm.get_callback()],
)
print("saving weights")
model.save(os.path.join(PROJECT_FOLDER, "model_weights.h5"))
plotter.plot_loss(history)
plotter.plot_mse(history)
