import streamlit as st
from PIL import Image
import requests
from io import BytesIO
import cv2
import numpy as np
import os
from roboflow import Roboflow
import supervision as sv
from main_app import main as segmentation_app
from diffusers import AutoPipelineForText2Image


pipe = AutoPipelineForText2Image.from_pretrained("stabilityai/sdxl-turbo", torch_dtype=torch.float16, variant="fp16")
pipe.to("cpu")

def generate_image_with_diffuser(prompt, negative_prompt, aspect_ratio, seed, output_format, lighting_condition, background, angle):
    st.title("Text to Image Generation")

    # Generate image from prompt using diffuser
    try:
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            num_inference_steps=1,
            guidance_scale=0.0,
            aspect_ratio=aspect_ratio,
            seed=seed,
            output_format=output_format,
            lighting_condition=lighting_condition,
            background=background,
            angle=angle
        ).images[0]

        image = Image.fromarray(np.uint8(image))

        st.image(image, caption="Generated Image")
    
    except Exception as e:
        st.error(f"Error generating image: {e}")

def generate_image():
    st.title("Text to Image Generation with Diffuser")

    prompt = st.text_area("Prompt", "A cinematic shot of a baby racoon wearing an intricate Italian priest robe.")
    negative_prompt = st.text_input("Negative Prompt", "")
    aspect_ratio = st.selectbox("Aspect Ratio", ['1:1', '16:9', '21:9'])
    seed = st.number_input("Seed", min_value=0, max_value=9999, value=25, step=1)
    output_format = st.selectbox("Output Format", ['png', 'jpg'])
    lighting_condition = st.selectbox("Lighting Condition", ['Natural', 'Studio', 'Dark', 'Bright'])
    background = st.selectbox("Background", ['Plain', 'Patterned', 'Outdoor', 'Indoor'])
    angle = st.selectbox("Angle", ['Front', 'Side', 'Top', 'Perspective'])

    if st.button("Generate Image"):
        generate_image_with_diffuser(prompt, negative_prompt, aspect_ratio, seed, output_format, lighting_condition, background, angle)

def load_image(uploaded_file):
    image = Image.open(uploaded_file)
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

def product_recognition_filter():
    st.title("Product Recognition Filter")

    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg"])

    if uploaded_file:
        image = load_image(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        if st.button("Run Detection"):
            api_key = "W9VikDr39oibIVgg5UOS"
            project_name = "coco-dataset-vdnr1"
            version_number = 11
            confidence = 40

            rf = Roboflow(api_key=api_key)
            project = rf.workspace().project(project_name)
            model = project.version(version_number).model

            with open("temp_image.png", "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            result = model.predict("temp_image.png", confidence=confidence).json()
            labels = [item["class"] for item in result["predictions"]]
            detections = sv.Detections.from_roboflow(result)

            label_annotator = sv.LabelAnnotator()
            mask_annotator = sv.MaskAnnotator()
            annotated_image = mask_annotator.annotate(scene=image.copy(), detections=detections)
            annotated_image = label_annotator.annotate(scene=annotated_image, detections=detections, labels=labels)

            st.image(annotated_image, caption="Annotated Image", use_column_width=True)

            # Show detected labels with larger font size
            st.subheader("Detected Labels")
            detected_labels = ", ".join(set(labels))
            st.write(detected_labels, font="Helvetica 30")

# Function for image segmentation and background change
def image_segmentation_and_background_change():
    st.title("Exclusion of Non-Relevant Images")

    uploaded_file = st.file_uploader("Choose an Image...", type=["jpg", "png", "jpeg"])
    prompt = st.text_input("Enter Your Prompt:", "")
    label = st.text_input("Enter The Label To Segment (e.g., 'Person', 'Cat'):", "")
    seed = 1000

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        img_path = "./uploaded_image.png"
        image.save(img_path)

        st.write("Segmenting and Changing Background...")

        # Replace with your segmentation function call
        orig_img, mask, diffused_img = segmentation_app(img_path=img_path, label=label, prompt=prompt, seed=seed)

        # Example placeholder outputs
        # orig_img = np.array(image)  # Placeholder
        # mask = np.zeros_like(orig_img)  # Placeholder
        # diffused_img = np.ones_like(orig_img) * 255  # Placeholder

        st.image(orig_img, caption='Original Image with Segmentation', use_column_width=True)
        st.image(mask, caption='Segmentation Mask', use_column_width=True)
        st.image(diffused_img, caption='Processed Image', use_column_width=True)

# Main Streamlit application
def main():
    st.title("Integrated Image Processing Application")

    st.sidebar.title("Navigation")
    app_choice = st.sidebar.selectbox(
        "Select an Application",
        ("AI Image Generation", "Product Recognition Filter", "Exclusion of Non-Relevant Images")
    )

    if app_choice == "AI Image Generation":
        generate_image_with_diffuser()
    elif app_choice == "Product Recognition Filter":
        product_recognition_filter()
    elif app_choice == "Exclusion of Non-Relevant Images":
        image_segmentation_and_background_change()

if __name__ == "__main__":
    main()
