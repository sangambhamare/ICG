import streamlit as st
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoImageProcessor, TableTransformerForObjectDetection, ViTForImageClassification
from PIL import Image
import torch

# Load BLIP model for captioning
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

# Load TableTransformer for Object Detection
image_processor = AutoImageProcessor.from_pretrained("microsoft/table-transformer-detection")
table_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

# Load ViT model for Facial Expression Recognition
vit_processor = AutoImageProcessor.from_pretrained("nateraw/vit-facial-expression-recognition")
vit_model = ViTForImageClassification.from_pretrained("nateraw/vit-facial-expression-recognition")

# Function to detect objects in the image
def detect_objects(image):
    inputs = image_processor(images=image, return_tensors="pt")
    outputs = table_model(**inputs)

    # Convert outputs to Pascal VOC format (bounding boxes and labels)
    target_sizes = torch.tensor([image.size[::-1]])
    results = image_processor.post_process_object_detection(outputs, threshold=0.7, target_sizes=target_sizes)[0]

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        detected_objects.append(table_model.config.id2label[label.item()])

    return list(set(detected_objects)) if detected_objects else ["scene"]

# Function to detect facial expressions
def detect_facial_expression(image):
    inputs = vit_processor(images=image, return_tensors="pt")
    outputs = vit_model(**inputs)
    
    # Get the predicted emotion
    predicted_label = vit_model.config.id2label[outputs.logits.argmax().item()]
    return predicted_label

# Function to generate a caption using BLIP
def generate_caption(image):
    inputs = blip_processor(image, return_tensors="pt")
    output = blip_model.generate(**inputs)
    return blip_processor.decode(output[0], skip_special_tokens=True)

# Function to make final captions precise and attractive
def generate_final_caption(caption, objects, emotion):
    object_list = ", ".join(objects)
    
    if emotion:
        final_caption = f"{caption}. Featuring {object_list}. Expressing {emotion.lower()}."
    else:
        final_caption = f"{caption}. Featuring {object_list}."

    return final_caption

# Streamlit UI
st.set_page_config(page_title="AI Caption Generator", layout="centered")
st.title("📸 AI Caption Generator")
st.write("Upload an image, and get a short, AI-generated caption with detected expressions!")

uploaded_image = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

if uploaded_image:
    image = Image.open(uploaded_image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Generating caption..."):
        # Step 1: Detect objects in image
        detected_objects = detect_objects(image)

        # Step 2: Detect facial expression (if a face is present)
        facial_expression = detect_facial_expression(image)

        # Step 3: Generate initial caption
        caption = generate_caption(image)

        # Step 4: Generate final refined caption
        final_caption = generate_final_caption(caption, detected_objects, facial_expression)

    st.subheader("📝 Generated Caption")
    st.markdown(f"**{final_caption}**")

    # Option to copy caption
    st.code(final_caption, language="text")

st.markdown("---")
st.write("🚀 Built with BLIP, TableTransformer Object Detection, Facial Expression Recognition, and Streamlit")
