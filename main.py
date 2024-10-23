import subprocess
import os
import torch
from src.data.load_data import load_data
from src.models.train import train_model
from src.models.evaluate import evaluate_model

MODEL_PATH = 'models/model_v1.pkl'


def train_and_save_model():
    train_loader, test_loader = load_data()

    model = train_model(train_loader, epochs=5)

    evaluate_model(model, test_loader)

    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Model saved to {MODEL_PATH}")


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print("No trained model found. Training a new model...")
        train_and_save_model()
    else:
        print("Model already exists. Skipping training.")

    subprocess.run(["streamlit", "run", "src/server/app.py"])
