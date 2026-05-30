import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the cleaned SCATS traffic dataset
df = pd.read_excel("Scats Data October 2006.xls",sheet_name="Data", header=1)

#print(df.head())
#print(df.columns)

volumeColumns = [] #first filter
#second filter
scatNum = []
dateList = []
volList = []
timeList = []


for i in df.columns:
    if str(i).startswith("V") and str(i)[1:].isdigit():
        volumeColumns.append(i)
filterData = df[["SCATS Number", "Date"] + volumeColumns]
#print(filterData.head())


for i, row in filterData.iterrows():
    scat = row["SCATS Number"]
    date = row["Date"]
    for col in volumeColumns:
        vol = row[col]

        scatNum.append(scat)
        dateList.append(date)
        timeList.append(col)
        volList.append(vol)

longData = pd.DataFrame({
    "SCATS Number": scatNum,
    "Date": dateList,
    "TimeInterval": timeList,
    "Volume": volList
})

#print(longData.head())

# for i in longData:
#     if str(i).startswith("V"):
#         num = int(i.replace("V", ""))
#         num = num*15
#         formatedTimeHour = num//60
#         formatedTimeMinute = num%60
#         formated = ("f{formatedTimeHour:02}:{formatedTimeMinute:02}")

def getTime(TimeIntervalValue):
    num = int(TimeIntervalValue.replace("V", ""))
    minutes = num * 15 

    #covert to format 
    hour = minutes//60
    minute = minutes%60
    return f"{hour:02}:{minute:02}"


longData["Time"] = longData["TimeInterval"].apply(getTime)

print("Dataset loaded successfully")
df = longData
# Combine Date and Time columns into one DateTime column
df["DateTime"] = pd.to_datetime(df["Date"].astype(str) + " " + df["Time"].astype(str))

# Extract useful time features for prediction
df["Hour"] = df["DateTime"].dt.hour
df["Minute"] = df["DateTime"].dt.minute
df["DayOfWeek"] = df["DateTime"].dt.dayofweek

# Sort records by SCATS site and time order
df = df.sort_values(["SCATS Number", "DateTime"])

# Create window size of 5 using previous traffic volumes
df["PreviousVolume_1"] = df.groupby("SCATS Number")["Volume"].shift(1)
df["PreviousVolume_2"] = df.groupby("SCATS Number")["Volume"].shift(2)
df["PreviousVolume_3"] = df.groupby("SCATS Number")["Volume"].shift(3)
df["PreviousVolume_4"] = df.groupby("SCATS Number")["Volume"].shift(4)
df["PreviousVolume_5"] = df.groupby("SCATS Number")["Volume"].shift(5)

# Use next traffic volume as the prediction target
df["NextVolume"] = df.groupby("SCATS Number")["Volume"].shift(-1)

# Remove rows with missing values
df = df.dropna()

# Select model input features and target output
X = df[
    [
        "SCATS Number",
        "Hour",
        "Minute",
        "DayOfWeek",
        "PreviousVolume_1",
        "PreviousVolume_2",
        "PreviousVolume_3",
        "PreviousVolume_4",
        "PreviousVolume_5",
    ]
]

y = df["NextVolume"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# Create the Random Forest regression model
model = RandomForestRegressor(n_estimators=100, random_state=42)

print("Training Random Forest model...")

# Train the model using the training data
model.fit(X_train, y_train)

# Predict traffic volume using the test data
y_pred = model.predict(X_test)

# Calculate error values for model evaluation
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)

print("\nRandom Forest Results")
print("MSE:", mse)
print("MAE:", mae)
print("RMSE:", rmse)

# Save actual and predicted values for comparison
results = X_test.copy()
results["ActualVolume"] = y_test.values
results["PredictedVolume"] = y_pred
results.to_csv("random_forest_predictions.csv", index=False)

print("\nPredictions saved as random_forest_predictions.csv")

# Plot actual vs predicted traffic volumes
plt.figure(figsize=(10, 5))
plt.plot(y_test.values[:100], label="Actual Volume")
plt.plot(y_pred[:100], label="Predicted Volume")
plt.title("Random Forest Traffic Flow Prediction")
plt.xlabel("Sample")
plt.ylabel("Traffic Volume")
plt.legend()
# plt.show()

#testing time
#Test 1: Data Processing test
def testDataProcessing(data):
    if len(data) > 0:
        print("Test case(Data Processing): Pass")
    else:
        print("Test case(Data Processing): Fail")
#Test 2: Sequence Creation test
def testFeatureCreation(x, y):
    if len(x) > 0 and len(y) > 0:
        print("Test case(Sequence Creation): Pass.", x.shape, y.shape)
    else:
        print("Test case 2: Fail. x or y is missing")
#Test 3: Prediction Output test
def testPrediction(predictions):
    if len(predictions) > 0:
        print("Test case 3(Prediction Output): Pass")
    else:
        print("Test case 3(Prediction Output): Fail")
#Test 4: Model Training test
def testTraining(history):
    if history is not None:
        print("Test case 4(Model Training): Pass")
    else:
        print("Test case 4(Model Training): Fail")
# Test 5: Evaluation Metrics test
def testMetrics(y_test, y_pred):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    actualSafe = np.where(y_test == 0, 1, y_test)
    mape = np.mean(np.abs((y_test - y_pred) / actualSafe)) * 100
    r2 = r2_score(y_test, y_pred)

    print("Test case 5(Evaluation Metrics): Pass")
    print("MAE:", mae, "MSE:", mse, "RMSE:", rmse, "MAPE:", mape, "R2:", r2)

testDataProcessing(df)
testFeatureCreation(X, y)
testPrediction(y_pred)
testTraining(model)
testMetrics(y_test, y_pred)

rfResult = pd.DataFrame({
    "Actual": y_test.values,
    "Random Forest": y_pred
})

rfResult.to_csv("random_forest_predictions.csv", index=False)