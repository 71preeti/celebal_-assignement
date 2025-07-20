import streamlit as st
import numpy as np
import pickle
from sklearn.datasets import load_iris
import pandas as pd

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# App Title
st.title("🌼 Iris Flower Prediction App")
st.markdown("Enter the measurements of the flower to predict its species.")

# Sidebar inputs
st.sidebar.header("User Input Features")
def user_input_features():
    sepal_length = st.sidebar.slider('Sepal Length (cm)', 4.0, 8.0, 5.1)
    sepal_width  = st.sidebar.slider('Sepal Width (cm)', 2.0, 4.5, 3.5)
    petal_length = st.sidebar.slider('Petal Length (cm)', 1.0, 7.0, 1.4)
    petal_width  = st.sidebar.slider('Petal Width (cm)', 0.1, 2.5, 0.2)
    data = {
        'sepal length (cm)': sepal_length,
        'sepal width (cm)': sepal_width,
        'petal length (cm)': petal_length,
        'petal width (cm)': petal_width
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

# Show input data
st.subheader("User Input Parameters")
st.write(input_df)

# Prediction
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# Output
iris = load_iris()
st.subheader("Prediction")
st.write(f"Predicted Species: **{iris.target_names[prediction[0]]}**")

st.subheader("Prediction Probability")
df_proba = pd.DataFrame(prediction_proba, columns=iris.target_names)
st.bar_chart(df_proba.T)

# Footer
st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit")
