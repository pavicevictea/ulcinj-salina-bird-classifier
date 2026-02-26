import os
import matplotlib.pyplot as plt
from train import train
from evaluate import evaluate
from config import RESULTS_PATH

def plot_history(history):
    os.makedirs(RESULTS_PATH, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    
    ax1.plot(history.history['loss'], label='Train Loss')
    ax1.plot(history.history['val_loss'], label='Validation Loss')
    ax1.set_title('Loss Curve')
    ax1.legend()
    
    ax2.plot(history.history['accuracy'], label='Train Acc')
    ax2.plot(history.history['val_accuracy'], label='Validation Acc')
    ax2.set_title('Accuracy Curve')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_PATH, 'training_results.png'))
    plt.show()

if __name__ == "__main__":
    history, class_names = train()
    plot_history(history)
    evaluate()