import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image, ImageOps
import numpy as np

@st.cache_resource
def load_ml_model():
    model = tf.keras.models.load_model('model.h5', compile=False)
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=False),
                  metrics=['accuracy'])
    return model

st.set_page_config(
    page_title="Weather Classification App",
    page_icon="🌤️",
)

st.title("🌤️ Weather Classification")
st.write("Upload a weather image to classify it as Cloudy, Rain, Sunshine, or Sunrise.")

# Load the model
try:
    model = load_ml_model()
    model_loading_success = True
except Exception as e:
    st.error(f"Error loading model: {e}")
    model_loading_success = False

# File uploader
file = st.file_uploader(
    "Upload a Weather Picture", 
    type=["jpg", "png"],
    help="Choose any picture of weather from your device gallery"
)

def import_and_predict(image_data, model):
    size = (128, 128)
    image = ImageOps.fit(image_data, size, Image.Resampling.LANCZOS)
    img = np.asarray(image)
    
    # Normalize the image data to [0, 1]
    img = img / 255.0

    # Reshape the image to match the model's input shape
    img_reshape = np.expand_dims(img, axis=0)
    prediction = model.predict(img_reshape)
    return prediction

# Display weather classification
if model_loading_success:
    if file is None:
        st.info("Please upload an image file to get a prediction")
    else:
        try:
            col1, col2 = st.columns([2, 1])
            
            with col1:
                image = Image.open(file)
                st.image(image, use_container_width=True, caption="Uploaded Image")  # Updated parameter
            
            with col2:
                with st.spinner("Classifying..."):
                    prediction = import_and_predict(image, model)
                class_names = ["Cloudy", "Rain", "Sunshine", "Sunrise"]
                predicted_class = class_names[np.argmax(prediction)]
                confidence = np.max(prediction) * 100
                
                st.success(f"**Prediction:** {predicted_class}")
                st.info(f"Confidence: {confidence:.2f}%")
            
            # Show all class probabilities
            st.subheader("Prediction Probabilities")
            for i, class_name in enumerate(class_names):
                st.progress(float(prediction[0][i]))
                st.write(f"{class_name}: {prediction[0][i]*100:.2f}%")
                
        except Exception as e:
            st.error(f"Error during prediction: {e}")

st.markdown("---")
st.markdown("### About this app")
st.markdown("""
This app uses a deep learning model built with TensorFlow to classify weather conditions 
from images. The model was trained to identify four types of weather: Cloudy, Rain, 
Sunshine, and Sunrise.
""")
