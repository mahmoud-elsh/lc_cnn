import streamlit as st
from PIL import Image
from torchvision import transforms
from model import LungCancerCNN
from predict import load_model, predict
from torchcam.methods import SmoothGradCAMpp
from torchvision.transforms.functional import to_pil_image
from torchcam.utils import overlay_mask
import torch

if __name__ == "__main__":
    st.set_page_config(layout="wide")

    st.markdown("<h1 style='text-align: center;'> Lung Cancer Histopathological Image Detector</h1>", unsafe_allow_html=True)
    st.divider()

    left_column, middle, right_column = st.columns(3)

    with middle:
        uploaded_file = st.file_uploader("Choose An Image", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        with left_column:
            st.image(image, caption="Histopathological Image")

        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image_tensor = transform(image)
        image_tensor = image_tensor.unsqueeze(0)

        model = load_model("./model/model_state_dict3.pth")
        cam_extractor = SmoothGradCAMpp(model, target_layer=model.conv_block_3[4])
        
        pred = predict(model, image_tensor)
        pred_labels = (torch.sigmoid(pred) > 0.5).int().squeeze(1).item()
        if pred_labels == 0:
            label = "Cancerous"
        else:
            label = "Non-cancerous"

        activation_map = cam_extractor(0, pred)
        result = overlay_mask(image, to_pil_image(activation_map[0].squeeze(0), mode='F'), alpha=0.75)

        middle.markdown(f"<h3 style='text-align: center;'> Predicted Class: {label} </h3>", unsafe_allow_html=True)

        with right_column:
            st.image(result, caption="Grad-CAM Heatmap")
            
        st.divider()
        st.markdown("<p style='text-align: center;'> Made with <3 by Mahmoud</p>", unsafe_allow_html=True)
