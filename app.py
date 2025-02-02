import streamlit as st
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration

# Cache the BLIP model and processor so they're only loaded once.
@st.cache_resource
def load_blip_model():
    processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    return processor, model

def generate_caption(processor, model, image):
    """
    Generate an unconditional caption using BLIP.
    """
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def main():
    st.title("Image Captioning")
    st.write("Upload an image or enter an image URL to generate an unconditional caption using BLIP.")

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
                response.raise_for_status()
                image = Image.open(response.raw).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # If an image is available, display it and allow caption generation
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)

        if st.button("Generate Caption"):
            with st.spinner("Loading model and generating caption..."):
                processor, model = load_blip_model()
                caption = generate_caption(processor, model, image)
            st.write("**Caption:**", caption)
    else:
        st.write("Please upload an image or enter a valid image URL.")

if __name__ == "__main__":
    main()
