import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import pipeline

# Cache the image classification model so it's loaded only once per session.
@st.cache_resource
def load_image_model():
    model = MobileNetV2(weights="imagenet")
    return model

# Cache the caption generation pipeline from Hugging Face.
@st.cache_resource
def load_caption_generator():
    caption_generator = pipeline("text-generation", model="gpt2")
    return caption_generator

def extract_labels(image: Image.Image, model, top=5):
    """
    Given a PIL image and a model, preprocess the image,
    run prediction, and return the top predicted labels.
    """
    # Resize the image to the expected input size (224x224)
    image = image.resize((224, 224))
    
    # Convert the image to a NumPy array and add a batch dimension
    x = img_to_array(image)
    x = np.expand_dims(x, axis=0)
    
    # Preprocess the image for MobileNetV2
    x = preprocess_input(x)
    
    # Run prediction and decode the results
    preds = model.predict(x)
    decoded = decode_predictions(preds, top=top)[0]
    return decoded

def generate_caption(labels, generator, max_length=50):
    """
    Given a list of labels and a Hugging Face text generator,
    create a prompt and generate a creative social media caption.
    """
    # Filter out labels with very low confidence if desired
    # (here we use all top 5 labels; you can add a threshold if needed)
    label_list = [label for (_, label, confidence) in labels]
    
    # Create a prompt using the extracted labels.
    prompt = f"Write a creative and engaging social media caption for a photo featuring: {', '.join(label_list)}."
    
    # Generate text from the prompt
    result = generator(prompt, max_length=max_length, num_return_sequences=1)
    caption = result[0]['generated_text']
    
    # Optionally, remove the prompt from the generated caption if it's repeated.
    if caption.startswith(prompt):
        caption = caption[len(prompt):].strip()
    
    return caption

def main():
    st.title("Image Label Extractor & Caption Generator")
    st.write("Upload an image to extract labels and generate a creative social media caption.")

    # Allow the user to upload an image file (jpg, jpeg, or png)
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Open the image using PIL and display it
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Extracting labels, please wait...")

        # Load the MobileNetV2 model and extract the labels
        image_model = load_image_model()
        labels = extract_labels(image, image_model, top=5)
        
        st.write("### Extracted Labels:")
        for pred in labels:
            # Each 'pred' tuple is (imagenet_id, label, confidence)
            st.write(f"**{pred[1]}**: {pred[2]*100:.2f}%")
        
        st.write("Generating social media caption, please wait...")
        # Load the caption generator and generate a caption based on the labels
        caption_generator = load_caption_generator()
        caption = generate_caption(labels, caption_generator, max_length=50)
        
        st.write("### Social Media Caption:")
        st.write(caption)
    else:
        st.write("Please upload an image to get started.")

if __name__ == '__main__':
    main()
