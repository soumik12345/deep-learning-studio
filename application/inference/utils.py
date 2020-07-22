import pandas as pd
from PIL import Image
import streamlit as st
import tensorflow as tf
from src.gradcam import GradCam, get_overlayed_image
from plotly import express as px


def visualize_classification_prediction(prediction):
    ids, class_names, probabilities = [], [], []
    for pred in prediction[0]:
        ids.append(pred[0])
        class_names.append(pred[1])
        probabilities.append(pred[2])
    df = pd.DataFrame(data={
        'id': ids,
        'class': class_names,
        'probability': probabilities
    })
    figure = px.bar(
        df, x='class', y='probability',
        hover_data=['id'], color='probability'
    )
    return figure


def get_model_summary(model):
    string_list = []
    model.summary(print_fn=lambda x: string_list.append(x))
    model_summary = "<br>".join(string_list)
    return model_summary
