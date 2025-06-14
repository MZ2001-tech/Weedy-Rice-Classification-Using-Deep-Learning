import streamlit as st
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image

# Streamlit configuration
st.set_page_config(page_title="WeedNet  ", layout="wide")


# Load AlexNet
def load_alexnet_model(alexnet_model_path, num_classes=2):
    model = models.alexnet(weights="DEFAULT")
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
    model.load_state_dict(torch.load(alexnet_model_path), strict=False)
    model.eval()
    return model

# Load DenseNet
def load_densenet_model(densenet_model_path, num_classes=2):
    model = models.densenet121(weights="DEFAULT")
    model.classifier = nn.Linear(in_features=1024, out_features=num_classes)
    model.load_state_dict(torch.load(densenet_model_path), strict=False)
    model.eval()
    return model

# File paths to models
MODEL_PATH_ALEXNET = "C:\Desktop\CDCS6A\CSP650\Testing and development\Experiment Model\AlexE2V1.pth"
MODEL_PATH_DENSENET = "C:\Desktop\CDCS6A\CSP650\Testing and development\Experiment Model\DensenetE1v3.pth"

# Load the models
alexnet_model = load_alexnet_model(MODEL_PATH_ALEXNET)
densenet_model = load_densenet_model(MODEL_PATH_DENSENET)


# Preprocess image
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image_tensor = transform(image).unsqueeze(0)
    return image_tensor

# Classify image with the model
def classify_image(model, image):
    input_tensor = preprocess_image(image)
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = nn.Softmax(dim=1)(output).numpy().flatten()
    return probabilities


# Sidebar: File uploader
st.sidebar.title("Upload Image")
uploaded_file = st.sidebar.file_uploader("Choose an image file", type=["jpg", "jpeg", "png"])

# Main content
if uploaded_file is not None:
    # Read and preprocess the image
    image = Image.open(uploaded_file)
    #   st.image(image, caption="Uploaded Image", use_container_width=True)

    # Get confidence scores from both models
    alexnet_scores = classify_image(alexnet_model, image)
    densenet_scores = classify_image(densenet_model, image)

    # Calculate confidence percentages
    alexnet_confidence = max(alexnet_scores) * 100
    densenet_confidence = max(densenet_scores) * 100

    # Get predicted class
    if alexnet_confidence > densenet_confidence:
        predicted_class = "Cultivated Rice" if alexnet_scores[0] > alexnet_scores[1] else "Weedy Rice"
    else:
        predicted_class = "Cultivated Rice" if densenet_scores[0] > densenet_scores[1] else "Weedy Rice"

    # Description of predicted class    
    if predicted_class == "Cultivated Rice":
        characteristics = """
                                   Cultivated rice chracteristics:
                                   - Height : Tall
                                   - Grainsize: Small and inconsistant
                                   - Seed Shatering: Low tendency for seed shatering
                                   - Thick of Awness: Have thin to no awnwess
                                    """

    else:
        characteristics = """
                                    weedy rice  chracteristics:
                                   - Height : Shorter
                                   - Grainsize: Small and inconsistant
                                   - Seed Shatering: High tendency for seed shatering
                                   - Thick of Awness: Have thick Awness 
                                    """

    # Layout for displaying results
    col1, col2 = st.columns([1, 2])  # Adjust width as needed

    # Left column: Display input image and predicted class
    with col1:
        st.subheader("Input Image")
        st.image(image, caption="Uploaded Image", use_container_width=True)

        st.subheader("Predicted Class")
        st.write(f"**{predicted_class}**")

        st.subheader("Class Chracteristics")
        st.write(characteristics)

    # Right column: Horizontal bar chart for confidence scores
    with col2:
        st.subheader("Confidence Scores Comparison")

        # Horizontal bar chart
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=[alexnet_confidence, densenet_confidence],
            y=["AlexNet", "DenseNet-121"],
            orientation='h',
            marker=dict(color=["blue", "green"])
        ))

        fig.update_layout(
            title="Model Confidence Scores",
            xaxis_title="Confidence (%)",
            yaxis_title="Model",
            height=400
        )

        st.plotly_chart(fig)
else:
    st.write("")
