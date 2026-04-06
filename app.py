import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt

# ------------------------------------------------
# PAGE CONFIG
# ------------------------------------------------
st.set_page_config(
    page_title="AI Oral Disease Detection",
    page_icon="🦷",
    layout="wide"
)

st.title("🦷 AI-Powered Oral Disease Detection System")
st.markdown("Upload an oral cavity image and select a trained model to predict the disease.")

# ------------------------------------------------
# CLASS LABELS (Same Order As Training)
# ------------------------------------------------
classes = [
    "Calculus",
    "Caries",
    "Gingivitis",
    "Mouth Ulcer",
    "Tooth Discoloration",
    "Hypodontia"
]

# ------------------------------------------------
# MODEL INPUT SIZES (Same As Training)
# ------------------------------------------------
model_input_sizes = {
    "DenseNet-169": 224,       # Best Model
    "EfficientNet-B4": 380,
    "EfficientNet-B5": 456
}

# ------------------------------------------------
# LOAD MODELS (Cached)
# ------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "DenseNet-169": tf.keras.models.load_model(
            "densenet169_best.keras", compile=False
        ),
        "EfficientNet-B4": tf.keras.models.load_model(
            "efficientnet_b4_best.keras", compile=False
        ),
        "EfficientNet-B5": tf.keras.models.load_model(
            "efficientnet_b5_best.keras", compile=False
        )
    }
    return models

models = load_models()

# ------------------------------------------------
# PREPROCESS FUNCTION (EXACT SAME AS TRAINING)
# ------------------------------------------------
def preprocess_image(image, size):
    image = image.resize((size, size))
    img_array = np.array(image).astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# ------------------------------------------------
# SIDEBAR SETTINGS
# ------------------------------------------------
st.sidebar.header("⚙ Model Selection")

selected_model = st.sidebar.selectbox(
    "Choose Model",
    list(models.keys()),
    index=0  # Default = DenseNet-169
)



# ------------------------------------------------
# IMAGE UPLOAD
# ------------------------------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Oral Image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        size = model_input_sizes[selected_model]
        model = models[selected_model]

        img_array = preprocess_image(image, size)

        with st.spinner("🔍 Analyzing Image..."):
            predictions = model.predict(img_array)
            probabilities = predictions[0]

        predicted_class = classes[np.argmax(probabilities)]
        confidence = np.max(probabilities)

        st.subheader("🧾 Prediction Result")
        st.success(f"**Predicted Disease:** {predicted_class}")
        st.info(f"**Confidence:** {confidence*100:.2f}%")

        # ------------------------------------------------
        # PROBABILITY CHART
        # ------------------------------------------------
        st.subheader("📊 Class Probability Distribution")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(classes, probabilities)
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        plt.ylabel("Probability")
        plt.tight_layout()

        st.pyplot(fig)

        

else:
    st.info("Please upload an oral cavity image to begin prediction.")

