import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# =====================================================
# SETUP OUTPUT DIRECTORY
# =====================================================

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

os.makedirs(OUTPUT_DIR, exist_ok=True)


def generate_all_visuals(y_test, y_pred_dt, y_pred_rf, y_pred_nb):

    # =====================================================
    # CALCULATE METRICS AUTOMATICALLY
    # =====================================================

    dt_accuracy = accuracy_score(y_test, y_pred_dt)
    dt_precision = precision_score(y_test, y_pred_dt)
    dt_recall = recall_score(y_test, y_pred_dt)
    dt_f1 = f1_score(y_test, y_pred_dt)

    rf_accuracy = accuracy_score(y_test, y_pred_rf)
    rf_precision = precision_score(y_test, y_pred_rf)
    rf_recall = recall_score(y_test, y_pred_rf)
    rf_f1 = f1_score(y_test, y_pred_rf)

    nb_accuracy = accuracy_score(y_test, y_pred_nb)
    nb_precision = precision_score(y_test, y_pred_nb)
    nb_recall = recall_score(y_test, y_pred_nb)
    nb_f1 = f1_score(y_test, y_pred_nb)

    # =====================================================
    # 1. CONFUSION MATRIX - Decision Tree
    # =====================================================

    cm_dt = confusion_matrix(y_test, y_pred_dt)

    disp_dt = ConfusionMatrixDisplay(confusion_matrix=cm_dt)
    disp_dt.plot()

    plt.title("Decision Tree Confusion Matrix")

    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_dt.png"))
    plt.close()

    # =====================================================
    # 2. CONFUSION MATRIX - Random Forest
    # =====================================================

    cm_rf = confusion_matrix(y_test, y_pred_rf)

    disp_rf = ConfusionMatrixDisplay(confusion_matrix=cm_rf)
    disp_rf.plot()

    plt.title("Random Forest Confusion Matrix")

    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_rf.png"))
    plt.close()

    # =====================================================
    # 3. CONFUSION MATRIX - Naive Bayes
    # =====================================================

    cm_nb = confusion_matrix(y_test, y_pred_nb)

    disp_nb = ConfusionMatrixDisplay(confusion_matrix=cm_nb)
    disp_nb.plot()

    plt.title("Naive Bayes Confusion Matrix")

    plt.savefig(os.path.join(OUTPUT_DIR, "confusion_nb.png"))
    plt.close()

    # =====================================================
    # 4. MODEL PERFORMANCE BAR GRAPH
    # =====================================================

    models = ['Decision Tree', 'Random Forest', 'Naive Bayes']

    accuracy = [dt_accuracy, rf_accuracy, nb_accuracy]

    precision = [dt_precision, rf_precision, nb_precision]

    recall = [dt_recall, rf_recall, nb_recall]

    f1 = [dt_f1, rf_f1, nb_f1]

    x = np.arange(len(models))

    plt.figure(figsize=(10, 5))

    plt.bar(x - 0.3, accuracy, width=0.2, label='Accuracy')
    plt.bar(x - 0.1, precision, width=0.2, label='Precision')
    plt.bar(x + 0.1, recall, width=0.2, label='Recall')
    plt.bar(x + 0.3, f1, width=0.2, label='F1 Score')

    plt.xticks(x, models)

    plt.legend()

    plt.title("Model Performance Comparison")

    plt.savefig(os.path.join(OUTPUT_DIR, "performance_bar.png"))
    plt.close()

    # =====================================================
    # 5. RADAR CHART
    # =====================================================

    labels = ['Accuracy', 'Precision', 'Recall', 'F1']

    dt = [dt_accuracy, dt_precision, dt_recall, dt_f1]

    rf = [rf_accuracy, rf_precision, rf_recall, rf_f1]

    nb = [nb_accuracy, nb_precision, nb_recall, nb_f1]

    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)

    dt = np.concatenate((dt, [dt[0]]))
    rf = np.concatenate((rf, [rf[0]]))
    nb = np.concatenate((nb, [nb[0]]))

    angles = np.concatenate((angles, [angles[0]]))

    plt.figure(figsize=(6, 6))

    ax = plt.subplot(111, polar=True)

    ax.plot(angles, dt, label='Decision Tree')
    ax.plot(angles, rf, label='Random Forest')
    ax.plot(angles, nb, label='Naive Bayes')

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)

    plt.legend(loc='upper right')

    plt.title("Performance Radar Chart")

    plt.savefig(os.path.join(OUTPUT_DIR, "radar_chart.png"))
    plt.close()

    # =====================================================
    # 6. METRIC BREAKDOWN
    # =====================================================

    metrics = ['Accuracy', 'Precision', 'Recall', 'F1']

    dt_scores = [dt_accuracy, dt_precision, dt_recall, dt_f1]

    rf_scores = [rf_accuracy, rf_precision, rf_recall, rf_f1]

    nb_scores = [nb_accuracy, nb_precision, nb_recall, nb_f1]

    plt.figure(figsize=(8, 5))

    plt.plot(metrics, dt_scores, marker='o', label='Decision Tree')

    plt.plot(metrics, rf_scores, marker='o', label='Random Forest')

    plt.plot(metrics, nb_scores, marker='o', label='Naive Bayes')

    plt.legend()

    plt.title("Detailed Metric Breakdown")

    plt.savefig(os.path.join(OUTPUT_DIR, "metric_breakdown.png"))
    plt.close()

    # =====================================================
    # 7. MODEL VERDICT
    # =====================================================

    plt.figure(figsize=(7, 5))

    plt.bar(
        ['Decision Tree', 'Random Forest', 'Naive Bayes'],
        [dt_f1, rf_f1, nb_f1]
    )

    plt.title("Final Model Verdict (F1 Score)")

    plt.ylabel("F1 Score")

    plt.savefig(os.path.join(OUTPUT_DIR, "model_verdict.png"))
    plt.close()

    # =====================================================
    # 8. SUMMARY TABLE
    # =====================================================

    df = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest', 'Naive Bayes'],
        'Accuracy': [dt_accuracy, rf_accuracy, nb_accuracy],
        'Precision': [dt_precision, rf_precision, nb_precision],
        'Recall': [dt_recall, rf_recall, nb_recall],
        'F1': [dt_f1, rf_f1, nb_f1]
    })

    fig, ax = plt.subplots(figsize=(8, 2))

    ax.axis('tight')
    ax.axis('off')

    table = ax.table(
        cellText=df.values,
        colLabels=df.columns,
        loc='center'
    )

    plt.title("Summary Table")

    plt.savefig(os.path.join(OUTPUT_DIR, "summary_table.png"))

    plt.close()