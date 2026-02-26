import os
from data_loader import prepare_and_split
from model import create_model
from config import EPOCHS, BATCH_SIZE, MODEL_SAVE_PATH

def train():
    X_train, X_val, X_test, y_train, y_val, y_test, class_names = prepare_and_split()
    model = create_model(len(class_names))
    history = model.fit(X_train, y_train, epochs=EPOCHS, validation_data=(X_val, y_val), batch_size=BATCH_SIZE, shuffle=True, verbose=1)
    os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
    model.save(MODEL_SAVE_PATH)
    return history, class_names

if __name__ == "__main__":
    train()