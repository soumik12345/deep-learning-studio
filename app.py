import streamlit as st
from application.inference.inference_app import inference_branch_app


def run_app():
    st.markdown(
        '<h1>Deep Learning Studio</h1><hr><br>',
        unsafe_allow_html=True
    )
    inference_branch_app()


if __name__ == '__main__':
    run_app()
