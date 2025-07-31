# evaluate.py: Loads the saved model and test data, makes predictions on test set, prints classification report, confusion matrix, and ROC AUC score.
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def main():
    X_test = np.load('data/X_test.npy')
    y_test = np.load('data/y_test.npy')

    model = load_model('models/churn_model.h5')

    y_pred_prob = model.predict(X_test)
    y_pred = (y_pred_prob > 0.5).astype(int)

    print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred))
    print("ROC AUC Score:", roc_auc_score(y_test, y_pred_prob))

if __name__ == '__main__':
    main()
