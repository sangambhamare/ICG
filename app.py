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

def generate_caption(processor, model, image, prompt=None):
    """
    Generate a caption using BLIP.
    If a prompt is provided, conditional captioning is performed; otherwise, unconditional captioning.
    """
    if prompt:
        inputs = processor(image, prompt, return_tensors="pt")
    else:
        inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    caption = processor.decode(output[0], skip_special_tokens=True)
    return caption

def main():
    st.title("BLIP Image Captioning")
    st.write("Upload an image or enter an image URL to generate captions using BLIP.")

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
                image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")
            except Exception as e:
                st.error(f"Error loading image: {e}")

    # If an image is available, display it and allow caption generation
    if image:
        st.image(image, caption="Selected Image", use_container_width=True)
        
        mode = st.radio("Select captioning mode:", ("Conditional", "Unconditional"))
        
        if mode == "Conditional":
            prompt = st.text_input("Enter a prompt (e.g., 'a photography of'):", value="a photography of")
        else:
            prompt = None
        
        if st.button("Generate Caption"):
            processor, model = load_blip_model()
            caption = generate_caption(processor, model, image, prompt)
            if prompt:
                st.write("**Conditional Caption:**", caption)
            else:
                st.write("**Unconditional Caption:**", caption)
    else:
        st.write("Please upload an image or enter a valid image URL.")

if __name__ == "__main__":
    main()
