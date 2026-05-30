import matplotlib
matplotlib.use("Agg")
import pandas as pd
import matplotlib.pyplot as plt
#read the prediction
lstm = pd.read_csv("lstm_predictions.csv")
gru = pd.read_csv("gru_predictions.csv")
rf = pd.read_csv("random_forest_predictions.csv")

limit = 100

plt.plot(lstm["Actual"][:limit], label="Actual")
plt.plot(lstm["LSTM"][:limit], label="LSTM")
plt.plot(gru["GRU"][:limit], label="GRU")
plt.plot(rf["Random Forest"][:limit], label="Random Forest")

plt.title("Traffic Prediction Comparison")
plt.xlabel("Sample")
plt.ylabel("Traffic Volume")
plt.legend()

plt.savefig("compareModel")
# plt.show()