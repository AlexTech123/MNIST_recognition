import streamlit as st
import torch
from src.models.model import MNISTModel
from src.models.predict import predict
from src.data.preprocess import get_transform
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset


def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load('models/model_v1.pkl'))
    model.eval()
    return model


def predict_image(model, image):
    transform = get_transform()
    image_tensor = transform(image).unsqueeze(0)

    dataset = TensorDataset(image_tensor, torch.zeros(1))
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    predictions = predict(model, loader)

    return predictions[0]


st.title("MNIST Digit Recognition")

uploaded_file = st.file_uploader("Choose an image...", type="png")

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert('L').resize((28, 28))
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    model = load_model()

    prediction = predict_image(model, image)

    st.write(f'Prediction: {prediction}')