import os
import json
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
import openai

# Set OpenAI API key from st.secrets
openai.api_key = st.secrets["OPENAI_API_KEY"]

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
        
        # Debug: Display the input shape and data type
        st.write("Input shape:", x.shape, "dtype:", x.dtype)
        
        preds = model.predict(x)
        decoded = custom_decode_predictions(preds, top=top)[0]
        return decoded
    except Exception as e:
        st.error("Error during label extraction: " + str(e))
        raise

# ---------------------
# Generate captions using ChatGPT
# ---------------------
def generate_captions_with_chatgpt(labels, num_captions=10):
    """
    Use ChatGPT (OpenAI's ChatCompletion API) to generate social media captions based on the extracted labels.
    """
    # Extract just the label names (ignoring the ImageNet ID and confidence)
    label_list = [label for (_, label, confidence) in labels]
    
    # Create a prompt asking for succinct captions.
    prompt = (
        f"Generate {num_captions} succinct, creative, and engaging social media captions "
        f"(each between 5 and 10 words) for a photo featuring: {', '.join(label_list)}. "
        f"Output the captions as a numbered list."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a creative caption generator."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=150
    )
    
    text = response["choices"][0]["message"]["content"]
    # Parse the response into separate captions.
    lines = text.strip().splitlines()
    captions = []
    for line in lines:
        line = line.strip()
        if line:
            # Remove numbering if present (e.g., "1. " or "1) ")
            if line[0].isdigit():
                dot_index = line.find('.')
                if dot_index != -1:
                    line = line[dot_index+1:].strip()
                else:
                    paren_index = line.find(')')
                    if paren_index != -1:
                        line = line[paren_index+1:].strip()
            captions.append(line)
    return captions

# ---------------------
# Main Streamlit app function
# ---------------------
def main():
    st.title("Image Label Extractor & ChatGPT Caption Generator")
    st.write("Upload an image to extract labels and generate creative social media captions (5-10 words each) using ChatGPT.")

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
        
        st.write("Generating social media captions using ChatGPT, please wait...")
        captions = generate_captions_with_chatgpt(labels, num_captions=10)
        
        st.write("### Generated Social Media Captions:")
        for idx, caption in enumerate(captions, start=1):
            st.write(f"**Caption {idx}:** {caption}")
    else:
        st.write("Please upload an image to get started.")

if __name__ == '__main__':
    main()
