import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import nltk
from nltk.corpus import wordnet
from textblob import TextBlob
import boto3
import io

# Download necessary NLP datasets
nltk.download('wordnet')

# Load BLIP model for captioning
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Initialize AWS Rekognition (or replace with OpenAI Vision API)
aws_client = boto3.client("rekognition", region_name="us-east-1")

# Function to detect objects and scenes using AWS Rekognition
def detect_scene(image):
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    response = aws_client.detect_labels(Image={'Bytes': img_bytes}, MaxLabels=10)
    detected_labels = [label['Name'].lower() for label in response['Labels']]

    return detected_labels

# Function to determine image category based on detected objects
def classify_scene(labels):
    categories = {
        "sunset": ["sunset", "sky", "dusk"],
        "cityscape": ["city", "buildings", "skyscraper"],
        "nature": ["forest", "tree", "mountain"],
        "portrait": ["person", "face", "human"],
        "night": ["night", "stars", "moon"],
        "beach": ["beach", "ocean", "sand"]
    }

    for category, keywords in categories.items():
        if any(label in keywords for label in labels):
            return category

    return "unknown"  # Default if no match is found

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
    
    return caption + " 🌇 A mesmerizing scene."

# Function to generate different caption styles based on detected scene
def generate_styled_captions(caption, scene):
    scene_styles = {
        "sunset": f"As the sun dips below the horizon, {caption} paints a golden memory.",
        "cityscape": f"The skyline whispers stories of ambition. {caption}",
        "nature": f"Nature’s embrace in full glory. {caption} sings with the wind.",
        "portrait": f"A soul captured in time. {caption} speaks volumes.",
        "night": f"The night unveils a world of mystery. {caption} shines in the dark.",
        "beach": f"Waves kiss the shore as {caption} tells a tale of tranquility.",
        "unknown": f"A scene full of wonders. {caption} holds untold stories."
    }
    return scene_styles.get(scene, caption)

# Streamlit UI
st.set_page_config(page_title="AI Caption Generator", layout="centered")
st.title("📸 Intelligent Caption Generator")
st.write("Upload an image, and our AI will generate creative captions based on the image content!")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Analyzing image and generating captions..."):
        # Step 1: Detect objects and scene
        detected_labels = detect_scene(image)
        scene_category = classify_scene(detected_labels)
        
        # Step 2: Generate initial caption
        caption = generate_caption(image)
        
        # Step 3: Enhance caption using synonyms
        enhanced_caption = enhance_caption(caption)

        # Step 4: Adjust based on sentiment
        final_caption = adjust_caption_tone(enhanced_caption)

        # Step 5: Generate scene-based styled captions
        styled_caption = generate_styled_captions(final_caption, scene_category)

    st.subheader("Generated Captions")
    st.write(f"🔹 **Detected Scene:** {scene_category.capitalize()}")
    st.write(f"🔹 **Original BLIP Caption:** {caption}")
    st.write(f"✨ **Enhanced Caption:** {enhanced_caption}")
    st.write(f"🎭 **Final Adjusted Caption:** {final_caption}")
    st.write(f"🌍 **Styled Caption Based on Scene:** {styled_caption}")

    # Option to download captions
    caption_text = f"Scene: {scene_category}\n{styled_caption}"
    st.download_button("📥 Download Caption", caption_text, file_name="caption.txt")

st.markdown("---")
st.write("🚀 Built with BLIP, NLP, AWS Rekognition, and Streamlit")
