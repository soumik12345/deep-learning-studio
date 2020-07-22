from PIL import Image
import streamlit as st
import tensorflow as tf
from src.gradcam import GradCam
from .utils import (
    get_overlayed_image, get_model_summary,
    visualize_classification_prediction
)


def model_explorer(model):
    st.markdown(
        '<hr><h3>Model Explorer</h3><br>',
        unsafe_allow_html=True
    )
    model_json = model.to_json()
    model_explorer_checkbox = st.checkbox(
        'Maximize', value=True, key='model_explorer_checkbox'
    )
    if model_explorer_checkbox:
        st.json(model_json)


def visualize_prediction_probablities(prediction):
    st.markdown(
        '<hr><h3>Prediction Probablities</h3><br>',
        unsafe_allow_html=True
    )
    plotly_figure = visualize_classification_prediction(prediction)
    pred_probability_checkbox = st.checkbox(
        'Maximize', value=True, key='pred_probability_checkbox'
    )
    if pred_probability_checkbox:
        st.plotly_chart(plotly_figure, use_container_width=True)


def visualize_gradcam(
        image_pil, model, preprocess_function, decode_function,
        last_layer, classifier_layers, size, original_size):

    grad_cam = GradCam(
        model, preprocess_function, decode_function, last_layer=last_layer,
        classifier_layers=classifier_layers, size=size
    )
    prediction = grad_cam.get_prediction(image_pil)
    visualize_prediction_probablities(prediction)

    st.markdown(
        '<hr><h3>GradCam</h3><br>',
        unsafe_allow_html=True
    )

    image_tensor = grad_cam.get_tensor(image_pil)
    gradcam_heatmap = grad_cam.apply_gradcam(image_tensor)

    overlayed_image = get_overlayed_image(
        image_pil, gradcam_heatmap,
        original_size, weightage=0.7
    )

    gradcam_checkbox = st.checkbox(
        'Maximize', value=True, key='gradcam_checkbox'
    )
    if gradcam_checkbox:
        st.image(overlayed_image)


def inference_branch_app():
    # ./assets/cat_1.jpg
    image_path = st.sidebar.file_uploader('Please Select a File')
    if image_path is not None:
        try:
            image_pil = Image.open(image_path)
            original_size = image_pil.size
            height, width = original_size
            image_pil = image_pil.resize((224, 224))
            st.image(image_pil, caption='Input Image')
        except:
            st.error('Invalid File')

        model_option = st.sidebar.selectbox(
            'Please Select the Model',
            ('', 'VGG16', 'VGG19', 'ResNet50')
        )

        model_dict = {
            'VGG16': [
                tf.keras.applications.vgg16.VGG16,
                tf.keras.applications.vgg16.preprocess_input,
                tf.keras.applications.vgg16.decode_predictions,
                'block5_conv3', (224, 224),
                ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
            ],
            'VGG19': [
                tf.keras.applications.vgg19.VGG19,
                tf.keras.applications.vgg19.preprocess_input,
                tf.keras.applications.vgg19.decode_predictions,
                'block5_conv4', (224, 224),
                ['block5_pool', 'flatten', 'fc1', 'fc2', 'predictions']
            ],
            'ResNet50': [
                tf.keras.applications.resnet50.ResNet50,
                tf.keras.applications.resnet50.preprocess_input,
                tf.keras.applications.resnet50.decode_predictions,
                'conv5_block3_out', (224, 224),
                ['avg_pool', 'predictions']
            ]
        }

        if model_option in list(model_dict.keys()):

            (
                model, preprocess_function, decode_function,
                last_layer, size, classifier_layers
            ) = model_dict[model_option]
            model = model(weights='imagenet')
            model_explorer(model)

            classify_button = st.sidebar.button('Classify')

            if classify_button:
                visualize_gradcam(
                    image_pil, model, preprocess_function, decode_function,
                    last_layer, classifier_layers, size, original_size
                )
