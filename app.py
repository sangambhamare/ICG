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
    TrOCRProcessor,
)

##############################
# BLIP Model for Captioning  #
##############################
@st.cache_resource
def load_blip_model():
    """
    Load and cache the BLIP image captioning model and its processor.
    """
    # Use the fast processor if available
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=True)
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption_blip(processor, model, image):
    """
    Generate an unconditional caption using the BLIP model.
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
    # Use the fast image processor if available
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
# TrOCR Model for Handwritten Recognition #
############################################
@st.cache_resource
def load_trocr_model():
    """
    Load and cache the TrOCR model and its processor.
    """
    # Use the fast processor if available
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten', use_fast=True)
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return processor, model, device

def generate_caption_trocr(image, processor, model, device):
    """
    Generate a caption (or text transcription) using the TrOCR model.
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    pixel_values = pixel_values.to(device)
    
    generated_ids = model.generate(pixel_values)
    caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return caption

#########################
# Streamlit Application #
#########################
def main():
    st.title("Triple Model Image Captioning")
    st.write(
        "Upload an image or enter an image URL to generate captions using three different models:\n"
        "- **BLIP** (general captioning)\n"
        "- **ViT-GPT2** (general captioning)\n"
        "- **TrOCR** (handwritten text recognition)"
    )

    # Choose how to input an image
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
                response.raise_for_status()  # Check that the request was successful
                image = Image.open(response.raw).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # If an image is available, display it and generate captions using all three models.
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)

        if st.button("Generate Captions"):
            with st.spinner("Generating captions..."):
                # Generate caption using the BLIP model
                blip_processor, blip_model = load_blip_model()
                caption_blip = generate_caption_blip(blip_processor, blip_model, image)

                # Generate caption using the ViT-GPT2 model
                vit_model, vit_feature_extractor, vit_tokenizer, vit_device = load_vit_gpt2_model()
                caption_vit = generate_caption_vit(image, vit_model, vit_feature_extractor, vit_tokenizer, vit_device)

                # Generate caption (text transcription) using the TrOCR model
                trocr_processor, trocr_model, trocr_device = load_trocr_model()
                caption_trocr = generate_caption_trocr(image, trocr_processor, trocr_model, trocr_device)

            st.markdown("### BLIP Caption:")
            st.write(caption_blip)
            st.markdown("### ViT-GPT2 Caption:")
            st.write(caption_vit)
            st.markdown("### TrOCR Caption:")
            st.write(caption_trocr)
    else:
        st.write("Please upload an image or enter a valid image URL.")

if __name__ == "__main__":
    main()
