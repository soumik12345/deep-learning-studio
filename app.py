import streamlit as st
from application.inference.inference_app import inference_branch_app


def run_app():
    st.markdown(
        '<h1>Deep Learning Studio</h1><hr><br>',
        unsafe_allow_html=True
    )
    action_option = st.selectbox(
        'What\'d yeh wanna do Harry?',
        ('', 'Inference on Pre-trained Model', 'Transfer Learning', 'Inference')
    )
    if action_option == 'Inference on Pre-trained Model':
        inference_branch_app()
    elif action_option == 'Transfer Learning':
        st.warning('Under Development')
    elif action_option == 'Inference':
        st.error('Not yet implemented')
    elif action_option != '':
        st.error('Action Option Not recognized')


if __name__ == '__main__':
    run_app()
