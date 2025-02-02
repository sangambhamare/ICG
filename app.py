import os
import json
import requests
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

# Set up Hugging Face API key from st.secrets (if available)
hf_api_key = st.secrets.get("HF_API_KEY")  # optional; remove if you don't have one

# Optionally force CPU usage if GPU issues persist:
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Configure GPU memory growth (if GPUs are available)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception as e:
        st.warning(f"Could not set GPU memory growth: {e}")

# ---------------------
# Custom decode_predictions function
# ---------------------
def custom_decode_predictions(preds, top=5):
    """
    Decode the predictions of an ImageNet model.
    Loads the ImageNet class index from a JSON file and returns the top predicted labels.
    """
    CLASS_INDEX_PATH = tf.keras.utils.get_file(
        'imagenet_class_index.json',
        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    )
    with open(CLASS_INDEX_PATH) as f:
        class_index = json.load(f)
    
    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = []
        for i in top_indices:
            str_i = str(i)
            if str_i in class_index:
                result.append((class_index[str_i][0], class_index[str_i][1], float(pred[i])))
            else:
                result.append(("N/A", "N/A", float(pred[i])))
        results.append(result)
    return results

# ---------------------
# Caching the image model
# ---------------------
@st.cache_resource
def load_image_model():
    model = MobileNetV2(weights="imagenet")
    return model

# ---------------------
# Helper function to extract labels from an image
# ---------------------
def extract_labels(image: Image.Image, model, top=5):
    """
    Resize and preprocess the image, run prediction, and return the top predicted labels.
    """
    try:
        image = image.resize((224, 224))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Debug: Display input shape and type
        st.write("Input shape:", x.shape, "dtype:", x.dtype)
        
        preds = model.predict(x)
        decoded = custom_decode_predictions(preds, top=top)[0]
        return decoded
    except Exception as e:
        st.error("Error during label extraction: " + str(e))
        raise

# ---------------------
# Generate captions using Hugging Face Inference API
# ---------------------
def generate_captions_with_hf(labels, num_captions=10):
    """
    Use the Hugging Face Inference API to generate social media captions based on the extracted labels.
    The API call sends a prompt to a text-generation model (here, using GPT-2).
    """
    # Extract label names from predictions
    label_list = [label for (_, label, confidence) in labels]
    
    # Create a refined prompt for succinct captions
    prompt = (
        f"Generate {num_captions} succinct, creative, and engaging social media captions "
        f"(each between 5 and 10 words) for a photo featuring: {', '.join(label_list)}. "
        f"Output the captions as a numbered list."
    )
    
    # Hugging Face Inference API endpoint for text-generation (using GPT-2)
    API_URL = "https://api-inference.huggingface.co/models/gpt2"
    headers = {}
    if hf_api_key:
        headers = {"Authorization": f"Bearer {hf_api_key}"}
    
    # Set payload parameters (you can adjust these as needed)
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 50,
            "temperature": 0.7,
            "top_p": 0.9,
            "do_sample": True,
            "num_return_sequences": num_captions
        },
        "options": {
            "wait_for_model": True
        }
    }
    
    response = requests.post(API_URL, headers=headers, json=payload)
    if response.status_code != 200:
        st.error("Error with Hugging Face API: " + response.text)
        return []
    
    results = response.json()
    
    # Parse the generated output(s)
    captions = []
    for item in results:
        generated_text = item.get("generated_text", "")
        # Remove the prompt if it appears at the beginning
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        # Split output into lines; expect a numbered list
        lines = generated_text.splitlines()
        for line in lines:
            line = line.strip()
            if line:
                # Remove numbering (e.g., "1. " or "1) ")
                if line[0].isdigit():
                    dot_index = line.find('.')
                    paren_index = line.find(')')
                    idx = -1
                    if dot_index != -1:
                        idx = dot_index
                    elif paren_index != -1:
                        idx = paren_index
                    if idx != -1:
                        line = line[idx+1:].strip()
                # Optionally trim to first 10 words
                words = line.split()
                if len(words) > 10:
                    line = " ".join(words[:10])
                captions.append(line)
                if len(captions) >= num_captions:
                    break
        if len(captions) >= num_captions:
            break

    return captions[:num_captions]

# ---------------------
# Main Streamlit app function
# ---------------------
def main():
    st.title("Image Label Extractor & Hugging Face Caption Generator")
    st.write("Upload an image to extract labels and generate creative social media captions (5-10 words each) using Hugging Face's API.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception as e:
            st.error("Error opening the image: " + str(e))
            return

        st.image(image, caption="Uploaded Image", use_container_width=True)
        st.write("Extracting labels, please wait...")
        image_model = load_image_model()
        labels = extract_labels(image, image_model, top=5)
        
        st.write("### Extracted Labels:")
        for pred in labels:
            st.write(f"**{pred[1]}**: {pred[2]*100:.2f}%")
        
        st.write("Generating social media captions using Hugging Face's Inference API, please wait...")
        captions = generate_captions_with_hf(labels, num_captions=10)
        
        st.write("### Generated Social Media Captions:")
        for idx, caption in enumerate(captions, start=1):
            st.write(f"**Caption {idx}:** {caption}")
    else:
        st.write("Please upload an image to get started.")

if __name__ == '__main__':
    main()
