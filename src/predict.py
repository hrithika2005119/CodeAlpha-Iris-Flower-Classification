
import joblib
import pandas as pd

# Load the saved artifacts
loaded_model = joblib.load("models/iris_model.pkl")
loaded_scaler = joblib.load("models/scaler.pkl")
loaded_encoder = joblib.load("models/label_encoder.pkl")

# A new flower's measurements — sepal_length, sepal_width, petal_length, petal_width
sample = pd.DataFrame([[5.1, 3.5, 1.4, 0.2]],
                       columns=["SepalLengthCm", "SepalWidthCm", "PetalLengthCm", "PetalWidthCm"])

sample_scaled = loaded_scaler.transform(sample)
prediction = loaded_model.predict(sample_scaled)
predicted_species = loaded_encoder.inverse_transform(prediction)[0]

print(f"Predicted species: {predicted_species}")