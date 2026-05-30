import tkinter as tk
from tkinter import ttk
import pandas as pd

root = tk.Tk()
root.title("Traffic Flow Prediction System")
root.geometry("500x350")

tk.Label(root, text="Traffic Flow Prediction System", font=("Arial", 16, "bold")).pack(pady=10)
tk.Label(root, text="Select Model:").pack()

model_choice = tk.StringVar()
ttk.Combobox(root, textvariable=model_choice, values=["GRU", "LSTM", "Random Forest"]).pack(pady=5)

def predict():
    model = model_choice.get()
    try:
        if model == "GRU":
            df = pd.read_csv("gru_predictions.csv")
            prediction = df["GRU"].iloc[-1]
        elif model == "LSTM":
            df = pd.read_csv("lstm_predictions.csv")
            prediction = df["LSTM"].iloc[-1]
        elif model == "Random Forest":
            df = pd.read_csv("random_forest_predictions.csv")
            prediction = df.iloc[-1, -1]
        else:
            result_label.config(text="Please select a model")
            return
        prediction = round(prediction, 2)
        travel_time = round(prediction / 10, 2)
        result_label.config(
            text=f"Predicted Traffic Flow: {prediction}\nEstimated Travel Time: {travel_time} mins"
        )
    except Exception as e:
        result_label.config(text=f"Error: {e}")
tk.Button(root, text="Predict", command=predict).pack(pady=10)

result_label = tk.Label(root, text="Predicted Traffic Flow:\nEstimated Travel Time:", font=("Arial", 12))
result_label.pack(pady=10)

root.mainloop()