import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tempfile
import os

# Set up page configuration
st.set_page_config(
    page_title="Helmet Detection Web App",
    page_icon="👷",
    layout="wide"
)

st.title("Safety Helmet Detection & Segmentation App")
st.info("Did you know? In a motorcycle crash, a helmet reduces your risk of a fatal head injury by 42% and brain injury by 69%. It takes less than 0.03 seconds for an impact to alter a life—wear your helmet.")
st.write("Upload an image or video to detect whether a head/face has a helmet or not.")

# Load the custom YOLO model (cached so it doesn't reload on every button click)
@st.cache_resource
def load_model():
    # Looks for 'best.pt' in the same folder
    if os.path.exists('best.pt'):
        return YOLO('best.pt')
    else:
        st.error("Error: 'best.pt' model file not found in the root directory!")
        return None

model = load_model()

if model is not None:
    # Sidebar selection for choosing input medium
    option = st.sidebar.selectbox("Select Input Type", ("Image", "Video"))
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)

    if option == "Image":
        st.header("Image Object Detection")
        uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_image is not None:
            # Convert uploaded file bytes into a PIL Image object
            image = Image.open(uploaded_image)
            
            # Create side-by-side columns for comparison
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original Image")
                st.image(image, use_container_width=True)
                
            with col2:
                st.subheader("Detection Results")
                with st.spinner("Running inference..."):
                    # Run YOLO prediction on the image
                    results = model.predict(source=image, conf=confidence_threshold)
                    
                    # YOLO's result.plot() returns a BGR numpy array
                    annotated_img_bgr = results[0].plot()
                    
                    # Convert BGR back to RGB for Streamlit display
                    annotated_img_rgb = cv2.cvtColor(annotated_img_bgr, cv2.COLOR_BGR2RGB)
                    
                    st.image(annotated_img_rgb, use_container_width=True)
                    
                    # Display count summaries
                    boxes = results[0].boxes
                    classes = boxes.cls.tolist() if boxes is not None else []
                    class_names = model.names
                    
                    counts = {"helmet": 0, "nohelmet": 0}
                    for c in classes:
                        name = class_names[int(c)]
                        if name in counts:
                            counts[name] += 1
                            
                    st.success(f"Detections complete! Found {counts['helmet']} Helmet(s) and {counts['nohelmet']} Bare Head(s).")

    elif option == "Video":
        st.header("Video Object Detection")
        uploaded_video = st.file_uploader("Choose a video file...", type=["mp4", "avi", "mov", "mkv"])
        
        if uploaded_video is not None:
            # OpenCV needs a concrete system path to open video streams.
            # We must stream uploaded file bytes into a temporary local file.
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            tfile.close()
            
            # Setup video capturing interface
            cap = cv2.VideoCapture(tfile.name)
            
            # Setup placeholder element where frames will continuously update
            st.subheader("Live Processing Feed")
            video_frame_placeholder = st.empty()
            
            # Stop button to interrupt long video sequences manually
            stop_clicked = st.button("Stop Processing")
            
            while cap.isOpened():
                if stop_clicked:
                    break
                    
                ret, frame = cap.read()
                if not ret:
                    break  # Video is finished
                
                # Predict on individual frame. Using stream=True is highly recommended 
                # for videos to manage RAM efficiently.
                results = model.predict(source=frame, conf=confidence_threshold, stream=True)
                
                # Extract the frame array with plotted boxes
                annotated_frame = next(results).plot()
                
                # Convert frame from BGR to RGB
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # Render the current frame matrix directly into the placeholder
                video_frame_placeholder.image(annotated_frame_rgb, use_container_width=True)
                
            cap.release()
            os.unlink(tfile.name)  # Delete the temporary video file to clean up disk storage
            st.success("Video processing completed!")