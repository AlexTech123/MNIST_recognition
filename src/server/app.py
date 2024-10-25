import inspect
import os
import sys

import streamlit as st
import torch
from PIL import Image
from torch.utils.data import DataLoader, TensorDataset

cmd_folder = os.path.realpath(os.path.abspath(os.path.split(inspect.getfile( inspect.currentframe() ))[0]))
if cmd_folder not in sys.path:
    sys.path.insert(0, cmd_folder + "/..")

from models.model import MNISTModel
from models.predict import predict
from data.preprocess import get_transform



def load_model():
    model = MNISTModel()
    model.load_state_dict(torch.load('models/model_v2.pkl'))
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

    st.markdown(f'<h1>Prediction: {prediction}</h1>', unsafe_allow_html=True)