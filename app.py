import streamlit as st
import requests
from PIL import Image
import torch
from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
    ViTImageProcessor,
    AutoTokenizer,
)

##############################
# BLIP Base Model for Captioning  #
##############################
@st.cache_resource
def load_blip_model():
    """
    Load and cache the BLIP image captioning (base) model and its processor.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption_blip(processor, model, image):
    """
    Generate an unconditional caption using the BLIP (base) model.
    """
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

############################################
# ViT-GPT2 Model for Captioning            #
############################################
@st.cache_resource
def load_vit_gpt2_model():
    """
    Load and cache the ViT-GPT2 image captioning model, its feature extractor, and tokenizer.
    """
    model = VisionEncoderDecoderModel.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    feature_extractor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning", use_fast=True)
    tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return model, feature_extractor, tokenizer, device

# Generation parameters for the ViT-GPT2 model
gen_kwargs = {"max_length": 16, "num_beams": 4}

def generate_caption_vit(image, model, feature_extractor, tokenizer, device):
    """
    Generate a caption using the ViT-GPT2 model.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    pixel_values = feature_extractor(images=[image], return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    output_ids = model.generate(pixel_values, **gen_kwargs)
    preds = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return preds[0].strip()

############################################
# BLIP Large Model for Captioning          #
############################################
@st.cache_resource
def load_blip_large_model():
    """
    Load and cache the BLIP image captioning (large) model and its processor.
    """
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
    return processor, model

def generate_caption_blip_large(processor, model, image):
    """
    Generate an unconditional caption using the BLIP (large) model.
    """
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

#########################
# Streamlit Application #
#########################
def main():
    st.title("Triple Model Image Captioning")
    st.write("Upload an image or enter an image URL to generate captions using three different models:" 
             "\n- **BLIP (Base)** (general captioning)"
             "\n- **ViT-GPT2** (general captioning)"
             "\n- **BLIP (Large)** (general captioning)")

    # Image input options
    input_method = st.radio("Select image input method:", ("Upload Image", "Image URL"))
    image = None

    if input_method == "Upload Image":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
    else:
        img_url = st.text_input("Enter image URL:")
        if img_url:
            try:
                response = requests.get(img_url, stream=True, timeout=10)
                response.raise_for_status()  # Ensure the request was successful
                image = Image.open(response.raw).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # If an image is available, display and process it.
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)

        if st.button("Generate Captions"):
            with st.spinner("Generating captions..."):
                # Generate caption using the BLIP (base) model
                processor_base, model_blip = load_blip_model()
                caption_blip = generate_caption_blip(processor_base, model_blip, image)

                # Generate caption using the ViT-GPT2 model
                model_vit, feature_extractor_vit, tokenizer_vit, device = load_vit_gpt2_model()
                caption_vit = generate_caption_vit(image, model_vit, feature_extractor_vit, tokenizer_vit, device)

                # Generate caption using the BLIP (large) model
                processor_large, model_blip_large = load_blip_large_model()
                caption_blip_large = generate_caption_blip_large(processor_large, model_blip_large, image)

            st.markdown("### BLIP (Base) Caption:")
            st.write(caption_blip)
            st.markdown("### ViT-GPT2 Caption:")
            st.write(caption_vit)
            st.markdown("### BLIP (Large) Caption:")
            st.write(caption_blip_large)
    else:
        st.write("Please upload an image or enter a valid image URL.")

if __name__ == "__main__":
    main()
