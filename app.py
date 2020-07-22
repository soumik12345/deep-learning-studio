import streamlit as st
from application.inference.inference_app import inference_branch_app


def run_app():
    st.markdown(
        '<h1>Deep Learning Studio</h1><hr><br>',
        unsafe_allow_html=True
    )
    action_option = st.selectbox(
        'What\'d yeh wanna do Harry?',
        ('', 'Inference on Pre-trained Model', 'Transfer Learning', 'Inference (Not Implemented)')
    )
    if action_option == 'Inference on Pre-trained Model':
        inference_branch_app()


if __name__ == '__main__':
    run_app()
