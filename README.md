# 🌤️ Weather Classification Web App using Streamlit and TensorFlow

This project is a simple yet effective web application that classifies weather conditions from images using a deep learning model built with TensorFlow and Keras. The interface is created with Streamlit, allowing users to interactively upload weather images and receive instant predictions.

---

## 📌 Overview

The **Weather Classification Web App** identifies one of four weather conditions:

- ☁️ **Cloudy**
- 🌧️ **Rain**
- ☀️ **Sunshine**
- 🌅 **Sunrise**

Upload any image, and the app will classify it using a pre-trained deep learning model (`model.h5`). This tool showcases the potential of computer vision in real-world applications such as automated weather categorization and environmental monitoring.

---

## 🚀 Features

- 🧠 **Deep Learning-Powered**: Built with TensorFlow and Keras.
- 🖼️ **Image Upload**: Upload `.jpg` or `.png` weather images through a simple UI.
- ⚙️ **Real-time Prediction**: Instantly returns the weather condition with high accuracy.
- 💾 **Model Caching**: Fast and efficient with Streamlit’s caching mechanism.
- ✅ **User-Friendly Interface**: Clean design with informative outputs.

---

## 🛠️ How It Works

1. **Upload Image** via the file uploader widget.
2. **Preprocessing**: The image is resized to 128x128 pixels and normalized.
3. **Prediction**: The image is fed into a pre-trained CNN model.
4. **Output**: The most probable weather class is displayed on screen.

---

## 🧠 Model Details

The model (`model.h5`) is a Convolutional Neural Network trained on a labeled dataset of weather images.

- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Input Shape**: (128, 128, 3)
- **Output Classes**: Cloudy, Rain, Sunshine, Sunrise

---

## 📦 Dependencies

Install the required packages with:

```bash
pip install streamlit tensorflow pillow numpy
