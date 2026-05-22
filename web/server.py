from flask import Flask, render_template, request
import pickle
import pandas as pd
import webbrowser

app = Flask(__name__)

# =====================================================
# LOAD MODEL
# =====================================================

with open("../models/flight_model.pkl", "rb") as f:
    model = pickle.load(f)

# =====================================================
# LOAD COLUMNS
# =====================================================

cols = pd.read_csv("../models/columns.csv").iloc[:, 0].tolist()

# =====================================================
# HOME ROUTE
# =====================================================

@app.route("/", methods=["GET", "POST"])
def home():

    prediction_text = ""
    probability_text = ""

    if request.method == "POST":

        try:
            # =====================================================
            # GET INPUT VALUES
            # =====================================================

            day = int(request.form["day"])
            weekday = int(request.form["weekday"])

            airline = request.form["airline"]

            time = request.form["time"]

            distance = int(request.form["distance"])

            # =====================================================
            # CREATE DATAFRAME
            # =====================================================

            sample = pd.DataFrame([{
                "DAY_OF_MONTH": day,
                "DAY_OF_WEEK": weekday,
                "OP_UNIQUE_CARRIER": airline,
                "DEP_TIME_BLK": time,
                "DISTANCE": distance
            }])

            # =====================================================
            # ENCODE INPUT
            # =====================================================

            sample_encoded = pd.get_dummies(sample)

            sample_encoded = sample_encoded.reindex(
                columns=cols,
                fill_value=0
            )

            # =====================================================
            # GET PROBABILITY
            # =====================================================

            prob = model.predict_proba(sample_encoded)[0][1]

            # =====================================================
            # CUSTOM THRESHOLD
            # =====================================================

            if prob >= 0.30:
                pred = 1
            else:
                pred = 0

            # =====================================================
            # OUTPUT MESSAGE
            # =====================================================

            if pred == 1:
                prediction_text = "⚠️ Flight will be DELAYED"

            else:
                prediction_text = "✅ Flight will be ON TIME"

            probability_text = f"Delay Probability: {prob:.2f}"

        except Exception as e:

            prediction_text = "❌ Error in input"

            probability_text = str(e)

    # =====================================================
    # RENDER PAGE
    # =====================================================

    return render_template(
        "index.html",
        prediction=prediction_text,
        probability=probability_text,
        form_data=request.form
    )

# =====================================================
# RUN SERVER
# =====================================================

if __name__ == "__main__":

    webbrowser.open("http://127.0.0.1:5000")

    app.run(debug=True)