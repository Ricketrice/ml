#https://www.geeksforgeeks.org/python/working-with-excel-files-using-pandas/ 
#https://www.w3schools.com/python/pandas/trypython.asp?filename=demo_ref_df_iterrows 
#https://www.w3schools.com/python/pandas/ref_df_apply.asp
#https://www.w3schools.com/python/numpy/numpy_intro.asp 
#https://www.geeksforgeeks.org/deep-learning/tf-keras-layers-lstm-in-tensorflow/ 

#pip install pandas numpy matplotlib scikit-learn
#pip install xlrd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

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

# print(longData)

# #time series prep
# #further sort data by forcing it 
# firstTest = longData[longData["SCATS Number"] == 970]
# firstScats = firstTest.sort_values(by=["Date", "Time"])
# print(firstScats.head(20))
# #turned to a list
# firstVol = firstScats["Volume"].tolist() 
# #prediction
# x = []
# y = []
# i = 3
# for i in range(len(firstVol) -3):
#     x.append(firstVol[i:i+3])
#     y.append(firstVol[i+3])
def timeSequence(longData, scatsNumber, window=5):
    x = []
    y = []
    
    selectedScats = longData[longData["SCATS Number"] == scatsNumber]
    selectedScats = selectedScats.sort_values(by=["Date", "Time"]) 
    volList = selectedScats["Volume"].tolist()

    for i in range(len(volList) - window):
        x.append(volList[i:i + window])
        y.append(volList[i + window])

    return x, y


allScats = longData["SCATS Number"].unique()
x = []
y = []
for i in allScats:
    tempX, tempY = timeSequence(longData, i)
    x = x+tempX
    y = y+tempY

x = np.array(x)
y = np.array(y)


minVal = x.min()
maxValue = x.max()
#covert it smaller for nn to learn as value are small and consisitent, creating a pattern 
x = (x-minVal) / (maxValue-minVal)
y = (y - minVal) / (maxValue-minVal)

#split data to learn from previous and future
split = int(len(x) * 0.7) #70% training
xTrain = x[:split]
xTest = x[split:]

yTrain = y[:split]
yTest = y[split:]

samples = xTrain.shape[0]
timeStep = xTrain.shape[1]
#Conver tot 3d input, the last feature is traffic volume
xTrain = xTrain.reshape(samples, timeStep, 1)
xTest = xTest.reshape(xTest.shape[0], xTest.shape[1],1)

#build LSTM model 
model = Sequential()
model.add(LSTM(50, input_shape=(xTrain.shape[1], 1))) #50 for meduim power 
model.add(Dense(1))

model.compile(
    optimizer="adam",
    loss="mse"
)


history = model.fit(
    xTrain,
    yTrain,
    epochs=10,
    batch_size=32
)


#compare 
predictions = model.predict(xTest)
#traffic vol real scale
orignalPrediction = predictions * (maxValue - minVal) + minVal
orignalPrediction = orignalPrediction.flatten()
actualOriginal = yTest * (maxValue - minVal) + minVal
actualOriginal = actualOriginal.flatten()

mae = mean_absolute_error(actualOriginal, orignalPrediction)
mse = mean_squared_error(actualOriginal, orignalPrediction)
rmse = np.sqrt(mse)
actualSafe = np.where(actualOriginal == 0, 1, actualOriginal)
mape = np.mean(np.abs((actualOriginal - orignalPrediction) / actualSafe)) * 100
r2 = r2_score(actualOriginal, orignalPrediction)
evs = explained_variance_score(actualOriginal, orignalPrediction)

print("MAE:", mae)
print("MSE:", mse)
print("RMSE:", rmse)
print("MAPE:", mape)
print("R2:", r2)

#testing time
#Test 1: Data Processing test 
def testDataProcessing(data):
    checkColumn = ["SCATS Number", "Date", "TimeInterval", "Volume", "Time"]
    for col in checkColumn:
        if col not in data.columns:
            print("Test case 2: Failed ", col, "is missing")
            return 
    print("Test case(Data Processing): Pass")

testDataProcessing(longData)

#Test 2: Sequence Creation test
def testSequenceCreation(x,y):
    if len(x) > 0 and len(y) > 0:
        print("Test case(Sequence Creation): Pass.", x.shape, y.shape)
    else:
        print("Test case 2: Fail. x or y is missing")
testSequenceCreation(x,y)
#Test 3: Scaling test
def testPrediction(predictions): #Test to see if the model successfully generate prediction output 
    if len(predictions) > 0:
        print("Test case 3(Prediction Output): Pass")
    else:
        print("Test case 3(Prediction Output: Failed)")
testPrediction(predictions)
#Test 4: Model traning test
def testTraining(history):
    if history is not None:
        print("Test case 4(Model Training): Pass")
    else:
        print("Test case 4(Model Training): Fail")
testTraining(history)
#Test 5: Evalutation Metrics 
def testMetrics(mae,mse, rmse,mape,r2):
    if mae >= 0 and mse >= 0 and rmse >= 0:
        print("Test case 5(Evaluation Metrics): Pass")
        print("MAE:", mae, "MSE:", mse, "RMSE:", rmse, "MAPE:", mape, "R2:", r2)    
    else :
        print("Test case 5(Evaluation Metrics): Fail")

testMetrics(mae, mse, rmse, mape, r2)

lstmResult = pd.DataFrame({
    "Actual": actualOriginal,
    "LSTM": orignalPrediction
})

lstmResult.to_csv("lstm_predictions.csv", index=False)