import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix
from data_loader import load_test_data
from config import MODEL_SAVE_PATH, RESULTS_PATH

def evaluate():
    X_test, y_test, class_names = load_test_data()
    model = tf.keras.models.load_model(MODEL_SAVE_PATH)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)

    print(classification_report(y_test, y_pred, target_names=class_names))

    os.makedirs(RESULTS_PATH, exist_ok=True)

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap='Greens')
    plt.title('Confusion Matrix - Ulcinj Salina')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'confusion_matrix.png'))
    plt.show()

if __name__ == "__main__":
    evaluate()