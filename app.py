import streamlit as st
import logging
import requests
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    RobertaTokenizer,
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
# CodeT5-Small Model for Social Media Caption Generation
##############################
@st.cache_resource
def load_codet5_model():
    # Load the CodeT5-small model and its RobertaTokenizer
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-small')
    model = T5ForConditionalGeneration.from_pretrained('Salesforce/codet5-small')
    return tokenizer, model

def generate_social_media_captions(initial_caption, num_outputs=3):
    """
    Given the initial caption, generate multiple creative social media captions using
    the Salesforce CodeT5-small model.
    Returns a list of caption strings.
    """
    prompt = (
        f"Rewrite the following text as creative social media captions with similar words "
        f"and include relevant hashtags: {initial_caption}"
    )
    tokenizer, model = load_codet5_model()
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    # Adjust max_length as needed; here we use 30 as an example.
    outputs = model.generate(
        input_ids,
        max_length=30,
        num_beams=5,
        num_return_sequences=num_outputs,
        early_stopping=True
    )
    captions = [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]
    return captions

##############################
# Streamlit App
##############################
st.title("Image Captioning & Social Media Caption Generator")
st.write("Upload an image to generate creative social media captions!")

uploaded_file = st.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.write("Captions are on the way...")
    
    # Generate the initial caption using BLIP Large model.
    proc_large, model_blip_large = load_blip_large_model()
    initial_caption = generate_caption_blip_large(proc_large, model_blip_large, image)
    
    st.write("**Initial Caption (BLIP Large):**")
    st.write(initial_caption)
    
    # Generate multiple social media captions using CodeT5-small.
    social_captions = generate_social_media_captions(initial_caption, num_outputs=3)
    
    st.markdown("### Generated Social Media Captions:")
    for idx, caption in enumerate(social_captions, start=1):
        st.markdown(f"**{idx}.** {caption}")
    
    # Optionally, combine final results into one list and display.
    results = [initial_caption, social_captions]
    st.write("**Final Results:**")
    st.write(results)
