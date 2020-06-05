import streamlit as st
st.title("My App")

clicked = st.button("CLICK ME")
if clicked:
    st.balloons()
