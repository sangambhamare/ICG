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
    # Using do_sample=True for more varied outputs.
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

def generate_captions(labels, generator, num_captions=10):
    """
    Given a list of labels and a Hugging Face text generator,
    create a prompt and generate multiple creative social media captions.
    The final caption text (without the prompt) is trimmed to be between 5 to 10 words.
    """
    # Extract label names from the predictions
    label_list = [label for (_, label, confidence) in labels]
    
    # Create a prompt using the extracted labels.
    prompt = f"Write a creative and engaging social media caption for a photo featuring: {', '.join(label_list)}."
    
    # Approximate the number of words in the prompt.
    prompt_word_count = len(prompt.split())
    # We want the generated part (after the prompt) to have between 5 and 10 words.
    # Set the min_length and max_length for the entire output accordingly.
    min_length = prompt_word_count + 5
    max_length = prompt_word_count + 10
    
    # Generate multiple captions using the text-generation pipeline.
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
        # Remove the prompt from the generated text.
        if caption_full.startswith(prompt):
            caption = caption_full[len(prompt):].strip()
        else:
            caption = caption_full.strip()
            
        # Now ensure the caption is between 5 and 10 words.
        words = caption.split()
        if len(words) < 5:
            # If too short, skip or you can choose to pad/ignore; here we leave it as-is.
            final_caption = caption
        elif len(words) > 10:
            # Trim to the first 10 words.
            final_caption = " ".join(words[:10])
        else:
            final_caption = caption
        captions.append(final_caption)
    return captions

def main():
    st.title("Image Label Extractor & Caption Generator")
    st.write("Upload an image to extract labels and generate creative social media captions (5 to 10 words each).")

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
            # Each prediction tuple is (imagenet_id, label, confidence)
            st.write(f"**{pred[1]}**: {pred[2]*100:.2f}%")
        
        st.write("Generating social media captions, please wait...")
        # Load the caption generator and generate multiple captions based on the labels
        caption_generator = load_caption_generator()
        captions = generate_captions(labels, caption_generator, num_captions=10)
        
        st.write("### Generated Social Media Captions:")
        for idx, caption in enumerate(captions, start=1):
            st.write(f"**Caption {idx}:** {caption}")
    else:
        st.write("Please upload an image to get started.")

if __name__ == '__main__':
    main()
