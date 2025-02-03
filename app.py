import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import nltk
from nltk.corpus import wordnet
from textblob import TextBlob
import os

# Download necessary NLP datasets
nltk.download('wordnet')

# Load BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Function to generate caption using BLIP
def generate_caption(image):
    inputs = processor(image, return_tensors="pt")
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)

# Function to enhance caption using WordNet (synonyms)
def enhance_caption(caption):
    words = caption.split()
    enhanced_caption = []

    for word in words:
        synonyms = wordnet.synsets(word)
        if synonyms:
            enhanced_caption.append(synonyms[0].lemmas()[0].name())  # Pick first synonym
        else:
            enhanced_caption.append(word)

    return " ".join(enhanced_caption)

# Function to adjust caption based on sentiment
def adjust_caption_tone(caption):
    sentiment = TextBlob(caption).sentiment.polarity

    if sentiment > 0.5:
        return caption + " ✨ A breathtaking moment!"
    elif sentiment < -0.5:
        return caption + " 🌑 A solemn and mysterious view."
    
    return caption + " 🌇 A mesmerizing cityscape."

# Function to generate different caption styles
def generate_styled_captions(caption):
    styles = {
        "Poetic": f"As the sun dips below the skyline, {caption} whispers farewell to the day.",
        "Descriptive": f"The golden hues paint the sky, casting a serene glow over the towering skyline. {caption}",
        "Humorous": f"{caption}. Probably the best skyline selfie moment ever!"
    }
    return styles

# Streamlit UI
st.set_page_config(page_title="AI Caption Generator", layout="centered")
st.title("📸 Intelligent Caption Generator")
st.write("Upload an image, and our AI will generate creative captions!")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating captions..."):
        # Step 1: Generate initial caption
        caption = generate_caption(image)
        
        # Step 2: Enhance caption using synonyms
        enhanced_caption = enhance_caption(caption)

        # Step 3: Adjust based on sentiment
        final_caption = adjust_caption_tone(enhanced_caption)

        # Step 4: Generate different styles
        styled_captions = generate_styled_captions(final_caption)

    st.subheader("Generated Captions")
    st.write(f"🔹 **Original BLIP Caption:** {caption}")
    st.write(f"✨ **Enhanced Caption:** {enhanced_caption}")
    st.write(f"🎭 **Final Adjusted Caption:** {final_caption}")

    st.subheader("✨ Styled Captions")
    for style, text in styled_captions.items():
        st.write(f"**{style}:** {text}")

    # Option to download captions
    caption_text = "\n".join([f"{style}: {text}" for style, text in styled_captions.items()])
    st.download_button("📥 Download Captions", caption_text, file_name="captions.txt")

st.markdown("---")
st.write("🚀 Built with BLIP, NLP, and Streamlit")
