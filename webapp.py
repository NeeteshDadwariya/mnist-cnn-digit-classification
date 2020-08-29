import random
import string
import streamlit as st
import pandas as pd

from joblib import load
from sklearn.datasets import fetch_openml
from digitclassifier import DigitClassifier

@st.cache
def fetch_dataset():
    return fetch_openml('mnist_784', version=1, return_X_y=True)

@st.cache
def load_dataset():
    x, y = fetch_dataset()
    x = (x/255).astype('float32')
    y = pd.Series(y, dtype="category").cat.codes.values
    img_w = 28
    img_l = 28
    return x, y, img_w, img_l

#@st.cache
def load_model():
    model = load("model.joblib")
    return model

def show_image(x, img_w, img_l, caption):
    st.image(x.reshape(img_w, img_l), caption, width=40)


st.title("Inference Pipeline for digit recognition ")
st.header("Loading dataset...")
x, y, img_w, img_l = load_dataset()
st.header("Dataset loaded.")


images = []
captions = []
for i in range(5):
    rand = random.randint(0, x.shape[1] - 1)
    caption = string.ascii_uppercase[i]
    image = x[rand]

    captions.append(caption)
    images.append(image)

    show_image(image, img_w, img_l, caption)

select_radio = st.sidebar.radio("Please select a image which you want to get classified ?", captions)

if select_radio:
    model = load_model()
    image = images[captions.index(select_radio)]
    pred = model.predict(image)
    st.header("Selected image")
    show_image(image, img_w, img_l, select_radio)
    st.header(f"Predicted value for ({select_radio}) = {pred}")
