from src.preprocessing import load_data, clean_data, prepare_features
from src.evaluate import evaluate_model
from src.utils import save_model, save_columns
from src.visualization import generate_all_visuals

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# 1. LOAD DATA
# =====================================================
data = load_data("data/raw/flights.csv")

# =====================================================
# 2. CLEAN DATA
# =====================================================
data = clean_data(data)

# =====================================================
# 3. PREPARE FEATURES
# =====================================================
X, y = prepare_features(data)

# Save columns for prediction
save_columns(X.columns, "models/columns.csv")

# =====================================================
# 4. TRAIN-TEST SPLIT
# =====================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# =====================================================
# 5. DECISION TREE
# =====================================================
dt_model = DecisionTreeClassifier(
    max_depth=6,
    class_weight='balanced',
    random_state=42
)

dt_model.fit(X_train, y_train)

y_pred_dt = dt_model.predict(X_test)

print("\n===== Decision Tree Results =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_dt))

print("Precision:",
      precision_score(y_test, y_pred_dt))

print("Recall:",
      recall_score(y_test, y_pred_dt))

print("F1 Score:",
      f1_score(y_test, y_pred_dt))

# =====================================================
# 6. RANDOM FOREST
# =====================================================
rf_model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

print("\n===== Random Forest Results =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_rf))

print("Precision:",
      precision_score(y_test, y_pred_rf))

print("Recall:",
      recall_score(y_test, y_pred_rf))

print("F1 Score:",
      f1_score(y_test, y_pred_rf))

# =====================================================
# 7. NAIVE BAYES
# =====================================================
nb_model = GaussianNB()

# GaussianNB requires dense arrays
X_train_dense = X_train.to_numpy()
X_test_dense = X_test.to_numpy()

nb_model.fit(X_train_dense, y_train)

y_pred_nb = nb_model.predict(X_test_dense)

print("\n===== Naive Bayes Results =====")

print("Accuracy:",
      accuracy_score(y_test, y_pred_nb))

print("Precision:",
      precision_score(y_test, y_pred_nb))

print("Recall:",
      recall_score(y_test, y_pred_nb))

print("F1 Score:",
      f1_score(y_test, y_pred_nb))

# =====================================================
# 8. SAVE FINAL MODEL (Random Forest)
# =====================================================
save_model(rf_model, "models/flight_model.pkl")

# =====================================================
# 9. GENERATE VISUALIZATIONS
# =====================================================
generate_all_visuals(
    y_test,
    y_pred_dt,
    y_pred_rf,
    y_pred_nb
)