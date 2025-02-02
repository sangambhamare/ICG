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

# Suppress excessive logging from transformers
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
# T5-Based Social Media Caption Generation (flan-t5-large)
##############################
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return tokenizer, model

def generate_social_media_captions(initial_caption, num_outputs=3):
    """
    Given an initial caption string, generate multiple creative social media captions 
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
# Extra Social Media Caption Generation (flan-t5-xxl)
##############################
@st.cache_resource
def load_t5_xxl_model():
    # Load the XXL model for additional caption generation
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-xxl")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-xxl")
    return tokenizer, model

def generate_extra_captions(initial_caption, num_outputs=3):
    """
    Using the flan-t5-xxl model, generate extra creative social media captions
    based on the initial caption.
    Returns a list of caption strings.
    """
    prompt = (
        f"Rewrite the following text as additional creative social media captions with similar words "
        f"and include relevant hashtags: {initial_caption}"
    )
    tokenizer, model = load_t5_xxl_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    outputs = model.generate(
        input_ids,
        max_new_tokens=100,
        num_beams=5,
        num_return_sequences=num_outputs,
        early_stopping=True,
    )
    extra_captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return extra_captions

##############################
# Streamlit App
##############################
st.title("Image Captioning & Enhanced Social Media Caption Generator")
st.write("Upload an image to generate an initial caption with BLIP Large, then generate creative social media captions using T5 models.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    with st.spinner("Generating initial caption using BLIP Large model..."):
        proc_large, model_blip_large = load_blip_large_model()
        initial_caption = generate_caption_blip_large(proc_large, model_blip_large, image)
    
    st.subheader("Initial Caption (BLIP Large):")
    st.write(initial_caption)
    
    with st.spinner("Generating creative social media captions using flan-t5-large..."):
        social_captions = generate_social_media_captions(initial_caption, num_outputs=3)
    
    st.subheader("Creative Social Media Captions (flan-t5-large):")
    for idx, cap in enumerate(social_captions):
        st.write(f"{idx+1}. {cap}")
    
    with st.spinner("Generating extra creative social media captions using flan-t5-xxl..."):
        extra_captions = generate_extra_captions(initial_caption, num_outputs=3)
    
    st.subheader("Extra Creative Social Media Captions (flan-t5-xxl):")
    for idx, cap in enumerate(extra_captions):
        st.write(f"{idx+1}. {cap}")
    
    # Combine all results into one array (optional)
    results = [initial_caption, social_captions, extra_captions]
    st.subheader("Final Results Array:")
    st.write(results)
