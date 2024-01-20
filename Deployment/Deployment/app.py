import streamlit as st
import streamlit.components.v1 as components 
import cv2 
import numpy as np 
from ultralytics import YOLO
import streamlit_option_menu as option_menu
from PIL import Image, ImageDraw
import io
import tempfile
import imageio.v2 as imageio
from moviepy.editor import ImageSequenceClip
import os
import shutil
from ultralytics.yolo.utils.plotting import Annotator
from cv2 import cvtColor
import os

#Importing the model

model = YOLO('best.pt')
def bgr2rgb(image):
    return image[:, :, ::-1]


    
def process_video(video_path):
    # Load the video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0 or fps is None:
        fps = 30  # Set a default value for fps if it is 0 or None

    # Create a list to store the processed frames
    processed_frames = []

    # Process each frame in the video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform the prediction on the frame
        prediction = model.predict(frame)
        frame_with_bbox = prediction[0].plot()

        # Convert the frame to PIL Image and store in the list
        processed_frames.append(Image.fromarray(frame_with_bbox))

    cap.release()

    # Create the output video file path
    video_path_output = "output.mp4"

    # Save the processed frames as individual images
    with tempfile.TemporaryDirectory() as temp_dir:
        for i, frame in enumerate(processed_frames):
            frame.save(f"{temp_dir}/frame_{i}.png")

        # Create a video clip from the processed frames
        video_clip_path = f"{temp_dir}/clip.mp4"
        os.system(f"ffmpeg -framerate {fps} -i {temp_dir}/frame_%d.png -c:v libx264 -pix_fmt yuv420p {video_clip_path}")

        # Rename the video clip with the desired output path
        shutil.copy2(video_clip_path, video_path_output)

    return video_path_output



        
def main():

    with open("styles.css", "r") as source_style:
        st.markdown(f"<style>{source_style.read()}</style>", 
             unsafe_allow_html = True)
        
    st.title("Vegetation Image Classification")
    Header = st.container()
    js_code = """
        const elements = window.parent.document.getElementsByTagName('footer');
        elements[0].innerHTML = "Vegtable Image Classification " + new Date().getFullYear();
        """
    st.markdown(f"<script>{js_code}</script>", unsafe_allow_html=True)
            
    #st.image("logo.png")
    
    ##MainMenu
    
    with st.sidebar:
        selected = option_menu.option_menu(
            "Main Menu",
            options=[
                "Project Information",
                "Model Information",           
                "Classify Vegetable",
                "Contributors"
            ],
        )
    
    st.sidebar.markdown('---')
        
    ##HOME page 
    
    if selected == "Project Information":
        
        st.subheader("Problem Statement")
        problem_statement = """
        Our project focuses on automating vegetable classification using deep learning techniques.
        By leveraging convolutional neural networks and a diverse dataset, we aim to develop an accurate and efficient model. 
        This innovation holds the promise of revolutionizing agriculture and food processing, improving productivity, and reducing food waste.
        """
        
        st.write(problem_statement)
        
        with st.expander("Read More"): 
            text = """
        Our project is dedicated to automating vegetable classification through the utilization of cutting-edge
        deep learning techniques, specifically Convolutional Neural Networks (CNNs), and a comprehensive dataset. 
        This initiative has the potential to bring about a profound transformation in the agricultural and food processing sectors.
         By accurately and efficiently classifying vegetables, we aim to enhance productivity, reduce food waste, and ultimately contribute to
        global food security. This innovative approach streamlines the sorting and processing of vegetables, minimizing labor costs and
        increasing production throughput, while simultaneously reducing food wastage through precise quality assessment and enabling better
        supply chain management. In an era of growing global population, our project stands as a crucial step toward a more
        efficient, sustainable, and secure food production and distribution system."""
            
            st.write(text)
        
        st.subheader("Our Solution")
        Project_goal = """
        Our Team developed a Machine Learning ( ML ) model based on the YOLOv8 Architecture, which was trained on a comprehensive
        dataset of vegetable images and manually annotated them to highlight the various types of classification. Once the model was trained,
        we proceeded to test its performance on new and unseen data. This testing phase was vital to ensure that our model could
        generalize well and accurately identify road defects in real-world scenarios. In addition to the model,
        we developed a web application using the Streamlit API which serves as a user friendly interface for others to test the 
        trained model on their own videos and images
        """
        st.write(Project_goal)
        
    elif selected == "Classify Vegetable": 
        
        text1="Make the settings in the left Panel and see the classifcation"
        st.write(text1)
        st.sidebar.subheader('Settings')
        
        options = st.sidebar.radio(
            'Options:', ('Image', 'Video'), index=1)
        
        st.sidebar.markdown("---")
         # Image
        if options == 'Image':
            upload_img_file = st.sidebar.file_uploader('Upload Image', type=['jpg', 'jpeg', 'png'])
            if upload_img_file is not None:
                file_bytes = np.asarray(bytearray(upload_img_file.read()), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, 1)
                img_rgb = cvtColor(img, cv2.COLOR_BGR2RGB)
                prediction = model.predict(img_rgb)
                res_plotted = prediction[0].plot()
                image_pil = Image.fromarray(res_plotted)
                image_bytes = io.BytesIO()
                image_pil.save(image_bytes, format='PNG')

                # Create a container for the two images side by side
                col1, col2 = st.columns(2)

                # Display the uploaded image in the first column
                col1.image(img_rgb, caption='Uploaded Image', use_column_width=True)

                # Display the predicted image in the second column
                col2.image(image_bytes, caption='Predicted Image', use_column_width=True)


                
        if options == 'Video':
            upload_vid_file = st.sidebar.file_uploader(
                'Upload Video', type=['mp4', 'avi', 'mkv']
                )
            if upload_vid_file is not None:
            # Save the uploaded video file temporarily
                temp_file = tempfile.NamedTemporaryFile(delete=False)
                temp_file.write(upload_vid_file.read())

                # Process the video frames and get the output video file path
                video_path_output = process_video(temp_file.name)

                # Display the processed video using the st.video function
                st.video(video_path_output)
                
                

                # Remove the temporary files
                temp_file.close()
                os.remove(video_path_output)
           
                 
            
    elif selected == "Contributors":
        st.subheader("Contributors")
        st.markdown("<b><u>Project Contributors :</u></b> \n  ", unsafe_allow_html=True)
        st.write("""1.  Panistha Gupta \n 2.  Varun Venkat Sarvanan \n   """)

    elif selected == "Model Information":
        st.subheader('Introduction')
        Introduction = """
        In traditional agriculture and food processing,
        the task of classifying vegetables has been labor-intensive, error-prone, and often dependent
        on human expertise. This process is susceptible to subjectivity and can be time-consuming, leading to inefficiencies
        in the supply chain. Additionally, the accurate and timely classification of vegetables is vital for sorting, packaging,
        and distribution. 
        """
        st.write(Introduction)
        st.subheader('Architecture')
        Architecture = """
        The architecture of YOLO consists of a convolutional neural network i.e CNN which is inspired by GoogleNet 
        and is composed of several convolutional layers followed by fully connected layers: This means that the YOLO 
        architecture is made up of two types of layers - convolutional and fully connected layers. Convolutional
        layers are used to extract features from the input image, while fully connected layers are used to predict the
        class probabilities and bounding boxes for each object detected in the image.YOLO also uses various other techniques
        like anchor boxes, class prediction objectness score etc

        """
        st.write(Architecture)
        st.image('architecture.jpg')
        st.subheader('Training')
        Training = """
        The YOLOv8 model used in the Vegetable Image Classification System is trained on a large dataset of road images which were
        annotated with bounding boxes and class labels on Roboflow.Roboflow offers a range of datasets and annotation 
        tools specifically designed for computer vision and also provides a user-friendly interface and annotation 
        capabilities that stremline the process of labeling and preparing datasets for training machine learning models. 
        The training data includes diverse road conditions, different types of objects, and various environmental factors
        to ensure the model's generalization capability.
        
        """
        st.write(Training)
        st.subheader("Conclusion")
        conclusion = """
        In conclusion, our project, empowered by the YOLOv8 model, represents a significant advancement in the automation
        of vegetable classification. This state-of-the-art deep learning model, trained on a diverse and extensive dataset,
        offers a promising solution to the challenges faced by the agricultural and food processing industries. By harnessing
        the capabilities of YOLOv8, we have not only streamlined the classification process but also improved its accuracy and efficiency.
        The impact of this innovation is far-reaching, with the potential to enhance productivity, reduce food waste, and bolster global
        food security. As we move forward, we envision a future where the integration of YOLOv8 and vegetable classification will play a
        pivotal role in shaping a more sustainable and resilient food production and distribution system.

        """
        st.write(conclusion)   
if __name__ == '__main__':
    try:
        main()
    except SystemExit:
        pass
    
