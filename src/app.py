import streamlit as st
from src.predict import predict

st.title(" Phishing URL Detector")

url = st.text_input("Enter URL")

if st.button("Check"):
    result = predict(url)
    st.success(result)
