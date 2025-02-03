import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, TableTransformerModel, TableTransformerConfig
from PIL import Image
import torch
import nltk
from nltk.corpus import wordnet
from textblob import TextBlob

# Download necessary NLP datasets
nltk.download('wordnet')

# Load BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load TableTransformer model for object detection
table_config = TableTransformerConfig()
table_model = TableTransformerModel(table_config)

# Function to detect objects in image
def detect_objects(image):
    inputs = processor(image, return_tensors="pt")
    outputs = table_model(**inputs)

    # Extract detected objects
    detected_objects = []
    for i in range(outputs.logits.shape[1]):
        detected_objects.append(f"Object {i+1}")

    return detected_objects if detected_objects else ["Unknown Scene"]

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

# Function to generate dynamic styles based on detected objects
def generate_styled_captions(caption, objects):
    if "sky" in objects or "sunset" in objects:
        styles = {
            "Poetic": f"As the sun dips below the skyline, {caption} whispers farewell to the day.",
            "Descriptive": f"The golden hues paint the sky, casting a serene glow over the towering skyline. {caption}",
            "Humorous": f"{caption}. Probably the best skyline selfie moment ever!"
        }
    elif "car" in objects or "street" in objects:
        styles = {
            "Poetic": f"The city lights flicker as {caption} captures the urban heartbeat.",
            "Descriptive": f"A bustling street, where neon lights reflect on the asphalt. {caption}",
            "Humorous": f"{caption}. Someone honked, and a pigeon took off in style!"
        }
    elif "tree" in objects or "nature" in objects:
        styles = {
            "Poetic": f"Where the sky kisses the earth, {caption} tells a story of nature’s beauty.",
            "Descriptive": f"A peaceful landscape, where every leaf dances in harmony. {caption}",
            "Humorous": f"{caption}. Probably the best place to take a nap!"
        }
    else:
        styles = {
            "Poetic": f"{caption}. A moment frozen in time.",
            "Descriptive": f"{caption}. A scene full of hidden stories.",
            "Humorous": f"{caption}. If only pictures could talk!"
        }

    return styles

# Streamlit UI
st.set_page_config(page_title="AI Caption Generator", layout="centered")
st.title("📸 Intelligent Caption Generator")
st.write("Upload an image, and our AI will generate creative captions based on detected objects!")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image and generating captions..."):
        # Step 1: Detect objects in image
        detected_objects = detect_objects(image)
        st.write(f"🔍 **Detected Objects:** {', '.join(detected_objects)}")

        # Step 2: Generate initial caption
        caption = generate_caption(image)
        
        # Step 3: Enhance caption using synonyms
        enhanced_caption = enhance_caption(caption)

        # Step 4: Adjust based on sentiment
        final_caption = adjust_caption_tone(enhanced_caption)

        # Step 5: Generate different styles dynamically
        styled_captions = generate_styled_captions(final_caption, detected_objects)

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
st.write("🚀 Built with BLIP, Object Detection, NLP, and Streamlit")
