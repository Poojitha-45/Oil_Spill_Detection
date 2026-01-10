import os
import cv2
import torch
import numpy as np
import gdown
from model import UNet

# -------------------------------------------------
# Device
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -------------------------------------------------
# Load Model (STREAMLIT SAFE)
# -------------------------------------------------
MODEL_PATH = "unet_oilspill_aug_final.pth"

# Google Drive direct download URL
MODEL_URL = "https://drive.google.com/uc?id=1leD8XL-mqN-BNh7Pwa-_NdizQIdRcIXk"

# Download model if not present (Streamlit Cloud)
if not os.path.exists(MODEL_PATH):
    gdown.download(MODEL_URL, MODEL_PATH, quiet=False)

model = UNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------------------------------------------------
# Image Preprocessing (FIXED)
# -------------------------------------------------
def preprocess_image(img, img_size=(256, 256)):
    """
    img: GRAYSCALE SAR image (H, W)
    """
    # DO NOT convert color – SAR is already grayscale
    img = cv2.resize(img, img_size)
    img = img.astype(np.float32) / 255.0

    tensor = torch.from_numpy(img).unsqueeze(0).unsqueeze(0)
    return tensor.to(DEVICE)

# -------------------------------------------------
# Post-processing
# -------------------------------------------------
def post_process(prob_map, threshold=0.3):
    """
    threshold lowered to match SAR oil contrast
    """
    binary = (prob_map > threshold).astype(np.uint8)

    kernel = np.ones((3, 3), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)

    return binary

# -------------------------------------------------
# Prediction
# -------------------------------------------------
def predict(img):
    """
    img: GRAYSCALE SAR image
    """
    with torch.no_grad():
        input_tensor = preprocess_image(img)
        prob_map = model(input_tensor)[0, 0].cpu().numpy()

    binary_mask = post_process(prob_map)
    return prob_map, binary_mask

# -------------------------------------------------
# Overlay Creation
# -------------------------------------------------
def create_overlay(original_bgr, binary_mask, alpha=0.5):
    """
    White mask = oil spill (red overlay)
    """
    overlay = cv2.resize(
        original_bgr,
        (binary_mask.shape[1], binary_mask.shape[0])
    )

    red_mask = np.zeros_like(overlay)
    red_mask[:, :, 2] = binary_mask * 255  # RED = oil

    blended = cv2.addWeighted(red_mask, alpha, overlay, 1 - alpha, 0)
    return blended

# -------------------------------------------------
# Severity Calculation
# -------------------------------------------------
def compute_oil_severity(binary_mask):
    total_pixels = binary_mask.size
    oil_pixels = binary_mask.sum()
    oil_percentage = (oil_pixels / total_pixels) * 100

    if oil_percentage == 0:
        severity = "None"
        risk = "No oil spill detected"
        color = "🟢"
    elif oil_percentage < 5:
        severity = "Low"
        risk = "Minimal environmental impact"
        color = "🟢"
    elif oil_percentage < 20:
        severity = "Medium"
        risk = "Moderate environmental risk"
        color = "🟡"
    else:
        severity = "High"
        risk = "Severe environmental threat"
        color = "🔴"

    return {
        "oil_percentage": round(oil_percentage, 2),
        "severity": severity,
        "risk": risk,
        "color": color
    }
