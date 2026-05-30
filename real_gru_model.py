#Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


#Loading the Dataset
df = pd.read_excel(
    "Scats Data October 2006.xls",
    sheet_name="Data"
)


df = df.iloc[1:]


df.columns = df.columns.astype(str).str.strip()


print(df.columns)


traffic_flow = df["00:00:00"].values


traffic_flow = pd.to_numeric(
    traffic_flow,
    errors='coerce'
)


traffic_flow = pd.Series(traffic_flow).dropna()


traffic_flow = traffic_flow.values.reshape(-1,1)

print(traffic_flow[:10])


#Preprocessing the Data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(traffic_flow)



X = []
y = []

sequence_length = 5

for i in range(sequence_length, len(scaled_data)):
    X.append(scaled_data[i-sequence_length:i])
    y.append(scaled_data[i])

X = np.array(X)
y = np.array(y)

print("X shape:", X.shape)
print("y shape:", y.shape)


#Building the GRU Model
model = Sequential()

#GRU(units=50): Adds a GRU layer with 50 units (neurons).
model.add(
    GRU(
        units=50,
        activation='relu',
        input_shape=(X.shape[1], X.shape[2])
    )
)

#Dense(units=1): The output layer which predicts a single value for the next time step.
model.add(Dense(1))

#Adam(): An adaptive optimizer commonly used in deep learning.
model.compile(
    optimizer='adam',
    loss='mse'
)

#Training the Model
#The epochs=10 specifies the number of iterations over the entire dataset, and batch_size=32 defines the number of samples per batch.
# model.fit(X, y, epochs=10, batch_size=32)

model.summary()

history = model.fit(
    X,
    y,
    epochs=10,
    batch_size=32
)

#Making Predictions For the Future traffic flow values, we use the trained model to predict the traffic flow based on the input sequences in X. The predictions are then inverse transformed to get the actual traffic flow values.
predictions = model.predict(X)

predictions = scaler.inverse_transform(predictions)

actual = scaler.inverse_transform(y)


results = pd.DataFrame({
    "Actual Flow": actual.flatten(),
    "Predicted Flow": predictions.flatten()
})

results.to_csv(
    "gru_predictions.csv",
    index=False
)



plt.plot(actual, label='Actual Flow')

plt.plot(predictions,
         label='Predicted Flow')

plt.title("REAL SCATS GRU Prediction")

plt.xlabel("Time")

plt.ylabel("Traffic Flow")

plt.legend()
plt.savefig("gru_prediction_graph.png")

# plt.show()

#testing time
#Test 1: Data Processing test
def testDataProcessing(data):
    if len(data) > 0:
        print("Test case(Data Processing): Pass")
    else:
        print("Test case(Data Processing): Failed")
testDataProcessing(traffic_flow)
#Test 2: Sequence Creation test
def testSequenceCreation(X, y):
    if len(X) > 0 and len(y) > 0:
        print("Test case(Sequence Creation): Pass.", X.shape, y.shape)
    else:
        print("Test case 2: Fail. X or y is missing")
testSequenceCreation(X, y)

#Test 3: Prediction Output test
def testPrediction(predictions):
    if len(predictions) > 0:
        print("Test case 3(Prediction Output): Pass")
    else:
        print("Test case 3(Prediction Output): Failed")
testPrediction(predictions)
#Test 4: Model training test
def testTraining(history):
    if history is not None:
        print("Test case 4(Model Training): Pass")
    else:
        print("Test case 4(Model Training): Fail")
testTraining(history)
#Test 5: Evalutation Metrics 

def testMetrics(actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    mse = mean_squared_error(actual, predictions)
    r2 = r2_score(actual, predictions)
    actualSafe = np.where(actual == 0, 1, actual)
    mape = np.mean(np.abs((actual - predictions) / actualSafe)) * 100

    rmse = np.sqrt(mse)
    if mae >= 0 and mse >= 0 and rmse >= 0:
        print("Test case 5(Evaluation Metrics): Pass")
        print("MAE:", mae, "MSE:", mse, "RMSE:", rmse, "MAPE:", mape, "R2:", r2)    
    else :
        print("Test case 5(Evaluation Metrics): Fail")
testMetrics(actual, predictions)

gruResult = pd.DataFrame({
    "Actual": actual.flatten(),
    "GRU": predictions.flatten()
})

gruResult.to_csv("gru_predictions.csv", index=False)