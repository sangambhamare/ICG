import streamlit as st
import numpy as np
from PIL import Image
import io
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array

# Cache the model so it is loaded only once per session.
@st.cache_resource
def load_model():
    model = MobileNetV2(weights="imagenet")
    return model

def extract_labels(image: Image.Image, model, top=5):
    """
    Given a PIL image and a model, preprocess the image, run prediction,
    and return the top predicted labels.
    """
    # Resize image to the model's expected input size (224x224)
    image = image.resize((224, 224))
    
    # Convert the image to a NumPy array
    x = img_to_array(image)
    
    # Expand dimensions to create a batch of size 1
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image as required by MobileNetV2
    x = preprocess_input(x)
    
    # Run the model prediction
    preds = model.predict(x)
    
    # Decode the predictions into a list of tuples (imagenet_id, label, confidence)
    decoded = decode_predictions(preds, top=top)[0]
    return decoded

def main():
    st.title("Image Label Extractor")
    st.write("Upload an image and get the labels predicted by a pre-trained MobileNetV2 model.")

    # Let the user upload an image file (jpg, jpeg, png)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the uploaded image using PIL
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        st.write("Extracting labels, please wait...")

        # Load the pre-trained model
        model = load_model()

        # Extract labels from the image
        predictions = extract_labels(image, model)
        
        st.write("### Predictions:")
        for pred in predictions:
            # Each prediction tuple is (imagenet_id, label, confidence)
            st.write(f"**Label:** {pred[1]}, **Confidence:** {pred[2]*100:.2f}%")
    else:
        st.write("Please upload an image to get started.")

if __name__ == '__main__':
    main()
