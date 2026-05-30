import pandas as pd
from travel_time import calculate_travel_time
 
distance = 5
 
# GRU
gru_df = pd.read_csv("gru_predictions.csv")
gru_prediction = gru_df["GRU"].iloc[-1]
gru_time = calculate_travel_time(gru_prediction, distance)
 
# LSTM
lstm_df = pd.read_csv("lstm_predictions.csv")
lstm_prediction = lstm_df["LSTM"].iloc[-1]
lstm_time = calculate_travel_time(lstm_prediction, distance)
 
# Random Forest
rf_df = pd.read_csv("random_forest_predictions.csv")
rf_prediction = rf_df["Random Forest"].iloc[-1]
rf_time = calculate_travel_time(rf_prediction, distance)
 
print("===== Travel Time Results =====")
print()
 
print(f"GRU Prediction: {round(gru_prediction,2)}")
print(f"GRU Travel Time: {gru_time} mins")
print()
 
print(f"LSTM Prediction: {round(lstm_prediction,2)}")
print(f"LSTM Travel Time: {lstm_time} mins")
print()
 
print(f"Random Forest Prediction: {round(rf_prediction,2)}")
print(f"Random Forest Travel Time: {rf_time} mins")