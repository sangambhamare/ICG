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
    GPT2Tokenizer,
    GPT2LMHeadModel,
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
# T5-Based Social Media Caption Generation Function (flan-t5-large)
##############################
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return tokenizer, model

def generate_social_media_captions(initial_caption, num_outputs=3):
    """
    Given an initial caption string, generate multiple creative social media captions 
    using T5 (flan-t5-large). Returns a list of caption strings.
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
# GPT-2 Based Extra Social Media Caption Generation Function
##############################
@st.cache_resource
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def generate_extra_captions_gpt2(initial_caption, num_outputs=3):
    """
    Using GPT-2, generate extra creative social media captions based on the initial caption.
    Returns a list of caption strings.
    """
    prompt = f"Rewrite the following text as creative social media captions: {initial_caption}"
    tokenizer, model = load_gpt2_model()
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        input_ids,
        max_length=50,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_return_sequences=num_outputs
    )
    extra_captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return extra_captions

##############################
# Streamlit App
##############################
st.title("Image Captioning & Enhanced Social Media Caption Generator")
st.write("Upload an image to generate an initial caption with BLIP Large, then get creative social media captions using T5 and GPT-2.")

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
        t5_captions = generate_social_media_captions(initial_caption, num_outputs=3)
    
    st.subheader("Creative Social Media Captions (flan-t5-large):")
    for idx, cap in enumerate(t5_captions):
        st.write(f"{idx+1}. {cap}")
    
    with st.spinner("Generating extra creative social media captions using GPT-2..."):
        gpt2_captions = generate_extra_captions_gpt2(initial_caption, num_outputs=3)
    
    st.subheader("Extra Creative Social Media Captions (GPT-2):")
    for idx, cap in enumerate(gpt2_captions):
        st.write(f"{idx+1}. {cap}")
    
    # Combine all results into one array (optional)
    results = [initial_caption, t5_captions, gpt2_captions]
    st.subheader("Final Results Array:")
    st.write(results)
