import os
import json
import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from transformers import pipeline

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
    Decode the prediction of an ImageNet model.
    This function loads the ImageNet class index from a JSON file and
    returns the top predicted labels.
    """
    # Download (or load from cache) the ImageNet class index file.
    CLASS_INDEX_PATH = tf.keras.utils.get_file(
        'imagenet_class_index.json',
        'https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json'
    )
    with open(CLASS_INDEX_PATH) as f:
        class_index = json.load(f)
    
    results = []
    for pred in preds:
        # Get the indices of the top predictions
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
# Caching models/pipelines
# ---------------------
@st.cache_resource
def load_image_model():
    model = MobileNetV2(weights="imagenet")
    return model

@st.cache_resource
def load_caption_generator():
    # Using do_sample=True for varied outputs.
    caption_generator = pipeline("text-generation", model="gpt2")
    return caption_generator

# ---------------------
# Helper functions
# ---------------------
def extract_labels(image: Image.Image, model, top=5):
    """
    Resize, preprocess the image, run prediction,
    and return the top predicted labels using custom_decode_predictions.
    """
    try:
        image = image.resize((224, 224))
        x = img_to_array(image)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)
        
        # Optional: Debug information
        st.write("Input shape:", x.shape, "dtype:", x.dtype)
        
        preds = model.predict(x)
        decoded = custom_decode_predictions(preds, top=top)[0]
        return decoded
    except Exception as e:
        st.error("Error during label extraction: " + str(e))
        raise

def generate_captions(labels, generator, num_captions=10):
    """
    Create a prompt based on the extracted labels and generate multiple captions.
    Each caption is trimmed to be between 5 and 10 words.
    """
    # Extract label names from the predictions
    label_list = [label for (_, label, confidence) in labels]
    prompt = f"Write a creative and engaging social media caption for a photo featuring: {', '.join(label_list)}."
    
    # Estimate prompt word count.
    prompt_word_count = len(prompt.split())
    # We want the generated part (after the prompt) to be between 5 and 10 words.
    min_length = prompt_word_count + 5
    max_length = prompt_word_count + 10
    
    results = generator(
        prompt,
        min_length=min_length,
        max_length=max_length,
        num_return_sequences=num_captions,
        do_sample=True
    )
    
    captions = []
    for result in results:
        caption_full = result['generated_text']
        # Remove the prompt from the generated text if repeated.
        if caption_full.startswith(prompt):
            caption = caption_full[len(prompt):].strip()
        else:
            caption = caption_full.strip()
        
        # Split into words and enforce length between 5 and 10 words.
        words = caption.split()
        if len(words) < 5:
            final_caption = caption  # You could also choose to skip too-short captions.
        elif len(words) > 10:
            final_caption = " ".join(words[:10])
        else:
            final_caption = caption
        captions.append(final_caption)
    return captions

# ---------------------
# Main app function
# ---------------------
def main():
    st.title("Image Label Extractor & Caption Generator")
    st.write("Upload an image to extract labels and generate creative social media captions (5 to 10 words each).")

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
        
        st.write("Generating social media captions, please wait...")
        caption_generator = load_caption_generator()
        captions = generate_captions(labels, caption_generator, num_captions=10)
        
        st.write("### Generated Social Media Captions:")
        for idx, caption in enumerate(captions, start=1):
            st.write(f"**Caption {idx}:** {caption}")
    else:
        st.write("Please upload an image to get started.")

if __name__ == '__main__':
    main()
