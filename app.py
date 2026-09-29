import streamlit as st
import joblib

# Load the trained model
model = joblib.load("spam_detection_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Spam Detector",
    page_icon="📩"
)

st.title("📩 Spam Message Detector")
st.write("Enter a message below to check whether it is spam or not.")

# User input
message = st.text_area(
    "Enter your message:",
    placeholder="Type your message here..."
)

# Prediction
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter a message.")
    else:
        prediction = model.predict([message])[0]

        if prediction == 1:
            st.error("This message is SPAM")
        else:
            st.success("This message is NOT SPAM")