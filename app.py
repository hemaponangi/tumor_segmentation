import json
from pathlib import Path

import numpy as np
import streamlit as st
import torch
from PIL import Image

from model import UNet, prepare_input, normalize, make_overlay, IMG_SIZE

st.set_page_config(
    page_title="Brain MRI Tumor Segmentation",
    page_icon="🧠",
    layout="wide",
)

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "unet_best.pt"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"


@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None, None, None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    metadata = {}
    if METADATA_PATH.exists():
        with open(METADATA_PATH, "r") as file:
            metadata = json.load(file)

    size = int(metadata.get("image_size", IMG_SIZE))
    base_filters = int(metadata.get("base_filters", 32))
    threshold = float(metadata.get("threshold", 0.5))

    model = UNet(base=base_filters).to(device)
    state = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model, device, (threshold, size, metadata)


@torch.inference_mode()
def predict(model, device, image, threshold):
    x = torch.from_numpy(normalize(image)).unsqueeze(0).float().to(device)
    probability = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    mask = (probability >= threshold).astype(np.uint8)
    return probability, mask


st.title("🧠 Brain MRI Tumor Annotation Assistant")
st.write(
    "U-Net based brain MRI image segmentation. Upload an MRI slice to view "
    "the predicted tumor mask and annotation overlay."
)
st.warning(
    "Research and educational prototype only. This application is not a "
    "medical diagnosis tool. Results must not be used for treatment decisions."
)

model, device, model_info = load_model()

if model is None:
    st.error("Trained model checkpoint not found.")
    st.markdown(
        """
        **To prepare the model:**
        1. Run `python train_model.py` in an environment with sufficient RAM/GPU.
        2. Wait until training finishes.
        3. Confirm that `artifacts/unet_best.pt` and `artifacts/metadata.json` exist.
        4. Restart this Streamlit app.
        """
    )
    st.stop()

default_threshold, image_size, metadata = model_info
st.sidebar.header("Model settings")
threshold = st.sidebar.slider(
    "Tumor detection threshold",
    min_value=0.10,
    max_value=0.90,
    value=float(default_threshold),
    step=0.05,
)
st.sidebar.caption(f"Running on: {device}")
st.sidebar.caption(f"Model input size: {image_size} × {image_size}")

uploaded_file = st.file_uploader(
    "Upload a brain MRI slice",
    type=["png", "jpg", "jpeg", "tif", "tiff"],
)

if uploaded_file is not None:
    try:
        pil_image = Image.open(uploaded_file)
        image = prepare_input(np.asarray(pil_image), size=image_size)
        probability, mask = predict(model, device, image, threshold)
        overlay = make_overlay(image, mask)

        tumor_pixels = int(mask.sum())
        total_pixels = int(mask.size)
        tumor_percentage = 100.0 * tumor_pixels / total_pixels
        mean_confidence = (
            float(probability[mask > 0].mean()) if tumor_pixels > 0 else 0.0
        )

        st.subheader("Segmentation results")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(image, caption="Processed MRI image", use_container_width=True)
        with col2:
            st.image(mask * 255, caption="Predicted tumor mask", clamp=True, use_container_width=True)
        with col3:
            st.image(overlay, caption="Prediction overlay", use_container_width=True)

        metric1, metric2, metric3 = st.columns(3)
        metric1.metric("Predicted tumor pixels", f"{tumor_pixels:,}")
        metric2.metric("Predicted area", f"{tumor_percentage:.2f}%")
        metric3.metric("Mean probability in predicted region", f"{mean_confidence:.3f}")

        import io
        buffer = io.BytesIO()
        Image.fromarray(mask * 255).save(buffer, format="PNG")
        st.download_button(
            "Download mask as PNG",
            data=buffer.getvalue(),
            file_name="predicted_tumor_mask.png",
            mime="image/png",
        )

        st.caption(
            "The red/yellow overlay shows the model prediction. The model does not "
            "confirm whether a tumor is present."
        )
    except Exception as error:
        st.error(f"Could not process this image: {error}")
else:
    st.info("Upload a brain MRI image to begin.")

with st.expander("Model information"):
    st.write("Architecture: U-Net with skip connections")
    st.write(f"Dataset: {metadata.get('dataset', 'LGG MRI Segmentation')}")
    if "best_validation_dice" in metadata:
        st.write(f"Best validation Dice during training: {metadata['best_validation_dice']}")
    st.write("Input channels: 3")
