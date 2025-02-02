import streamlit as st
import logging
import requests
from PIL import Image
import torch
import time
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
# T5-Based Social Media Caption Generation (flan-t5-large)
##############################
@st.cache_resource
def load_t5_model():
    tokenizer = T5Tokenizer.from_pretrained("google/flan-t5-large")
    model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-large")
    return tokenizer, model

def generate_social_media_captions(initial_caption, num_outputs=3):
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
# GPT-2 Based Extra Social Media Caption Generation
##############################
@st.cache_resource
def load_gpt2_model():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    return tokenizer, model

def generate_extra_captions_gpt2(initial_caption, num_outputs=3):
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
st.title("Social Media Caption Generator")
st.write("Upload an image to generate creative social media captions.")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Start timing before generation begins
    start_time = time.time()
    
    proc_large, model_blip_large = load_blip_large_model()
    initial_caption = generate_caption_blip_large(proc_large, model_blip_large, image)
    
    t5_captions = generate_social_media_captions(initial_caption, num_outputs=3)
    gpt2_captions = generate_extra_captions_gpt2(initial_caption, num_outputs=3)
    
    # Combine both sets of captions into one ordered list
    combined_captions = t5_captions + gpt2_captions
    elapsed_time = time.time() - start_time
    
    st.subheader(f"Social Media Captions (generated in {elapsed_time:.1f} seconds):")
    for idx, caption in enumerate(combined_captions, start=1):
        st.write(f"{idx}. {caption}")
