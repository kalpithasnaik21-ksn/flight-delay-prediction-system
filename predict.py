import pickle
import pandas as pd

# 1. Load model
with open("models/flight_model.pkl", "rb") as f:
    model = pickle.load(f)

# 2. Load columns
cols = pd.read_csv("models/columns.csv").iloc[:, 0].tolist()

# 3. Take input (example)
# sample = pd.DataFrame([{
#     "DAY_OF_MONTH": 10,
#     "DAY_OF_WEEK": 3,
#     "OP_UNIQUE_CARRIER": "AA",
#     "DEP_TIME_BLK": "Morning",
#     "DISTANCE": 1500
# }])

sample = pd.DataFrame([{
    "DAY_OF_MONTH": 25,
    "DAY_OF_WEEK": 5,          # Friday (busy day)
    "OP_UNIQUE_CARRIER": "AA", # Major airline
    "DEP_TIME_BLK": "Evening", # Peak time
    "DISTANCE": 2500           # Long distance
}])

# 4. Encode
sample_encoded = pd.get_dummies(sample)

# 5. Match columns
sample_encoded = sample_encoded.reindex(columns=cols, fill_value=0)

# 6. Predict
prediction = model.predict(sample_encoded)

# 7. Output
if prediction[0] == 1:
    print("⚠️ Flight will be DELAYED")
else:
    print("✅ Flight will be ON TIME")