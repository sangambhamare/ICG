import streamlit as st
import logging
import requests
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration,
)

# Suppress extra logging from transformers
logging.getLogger("transformers").setLevel(logging.ERROR)

##############################
# BLIP Large Model for Captioning
##############################
@st.cache_resource
def load_blip_large_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def generate_caption_blip_large(processor, model, image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

##############################
# T5-Based Social Media Caption Generation Function
##############################
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return tokenizer, model

def generate_social_media_captions(initial_caption, num_outputs=3):
    """
    Given the initial caption, generate multiple creative social media captions
    using similar words and including relevant hashtags.
    Returns a list of caption strings.
    """
    prompt = (
        f"Rewrite the following text as creative social media captions with similar words "
        f"and include relevant hashtags: {initial_caption}"
    )
    tokenizer, model = load_t5_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        num_beams=5,
        num_return_sequences=num_outputs,
        early_stopping=True,
    )
    captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

##############################
# Streamlit App
##############################
st.title("Image Captioning & Social Media Caption Generator")
st.write("Upload an image to get creative social media captions!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Captions are on the way...")

    # Generate initial caption using BLIP Large
    proc_large, model_blip_large = load_blip_large_model()
    initial_caption = generate_caption_blip_large(proc_large, model_blip_large, image)
    
    # Generate creative social media captions using T5
    social_captions = generate_social_media_captions(initial_caption, num_outputs=3)
    
    st.markdown("### Generated Social Media Captions:")
    for idx, caption in enumerate(social_captions, start=1):
        st.markdown(f"**{idx}.** {caption}")
