import streamlit as st
import cv2
import numpy as np
import base64
from pathlib import Path
from io import BytesIO

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4

from utils import predict, compute_oil_severity, create_overlay

# -------------------------------------------------
# Page configuration
# -------------------------------------------------
st.set_page_config(page_title="Oil Spill Detection", layout="wide")

# -------------------------------------------------
# Background
# -------------------------------------------------
def set_background(path):
    path = Path(path)
    if not path.exists():
        return
    with open(path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_background("background.jpg")

# -------------------------------------------------
# Title
# -------------------------------------------------
st.markdown(
    """
    <h1 style="text-align:center; font-size:44px;">
        🌊 Oil Spill Detection
    </h1>
    <p style="text-align:center; font-size:18px;">
        SAR Satellite Image Analysis for Oil Spill Detection
    </p>
    <hr>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Upload
# -------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload SAR Satellite Image (PNG / JPG)",
    type=["png", "jpg", "jpeg"]
)

run = st.button("🚀 Run SAR Analysis")

# -------------------------------------------------
# ANALYSIS (LOGIC UNCHANGED)
# -------------------------------------------------
if uploaded_file is not None and run:

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image_gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    if image_gray is None:
        st.error("❌ Invalid SAR image")
        st.stop()

    st.subheader("🖼️ Uploaded SAR Image")
    st.image(image_gray, width=340)

    with st.spinner("🔍 Detecting oil spills..."):
        prob_map, binary_mask = predict(image_gray)

    image_bgr = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2BGR)
    overlay = create_overlay(image_bgr, binary_mask)
    severity = compute_oil_severity(binary_mask)

    # -------------------------------------------------
    # Output
    # -------------------------------------------------
    st.subheader("🧠 Model Output")

    c1, c2, c3, c4 = st.columns([1.3, 1.3, 1.3, 1.5])

    with c1:
        st.markdown("<b style='font-size:18px;'>Probability Map</b>", unsafe_allow_html=True)
        st.image((prob_map * 255).astype(np.uint8), width=360)

    with c2:
        st.markdown("<b style='font-size:18px;'>Binary Oil Mask</b>", unsafe_allow_html=True)
        st.image(binary_mask * 255, width=360)

    with c3:
        st.markdown("<b style='font-size:18px;'>Oil Spill Overlay</b>", unsafe_allow_html=True)
        st.image(overlay, channels="BGR", width=360)

    with c4:
        st.markdown("<b style='font-size:18px;'>Severity Analysis</b>", unsafe_allow_html=True)
        st.markdown(
            f"""
            <div style="
                background: rgba(0,0,0,0.55);
                padding:16px;
                border-radius:12px;
                color:white;
            ">
                <div style="font-size:18px;">Oil Coverage</div>
                <div style="font-size:38px; font-weight:700;">
                    {severity["oil_percentage"]}%
                </div>
                <div style="font-size:22px;">
                    {severity["color"]} <b>{severity["severity"]}</b>
                </div>
                <div style="font-size:16px;">
                    {severity["risk"]}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # Detection message
    # -------------------------------------------------
    detected = binary_mask.sum() > 0
    st.markdown("<br>", unsafe_allow_html=True)

    if detected:
        st.markdown(
            """
            <div style="background:#8b0000;padding:14px;
            border-radius:10px;color:white;font-size:18px;text-align:center;">
            🛢️ Oil Spill Detected in the Image
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            """
            <div style="background:#006633;padding:14px;
            border-radius:10px;color:white;font-size:18px;text-align:center;">
            ✅ No Oil Spill Detected
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------------------------------
    # PDF REPORT GENERATION (WHAT YOU ASKED)
    # -------------------------------------------------
    def generate_pdf():
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        styles = getSampleStyleSheet()
        elements = []

        elements.append(Paragraph("<b>Oil Spill Analysis Report</b>", styles["Title"]))
        elements.append(Spacer(1, 12))

        elements.append(Paragraph(
            f"""
            Oil Coverage: <b>{severity['oil_percentage']}%</b><br/>
            Severity Level: <b>{severity['severity']}</b><br/>
            Risk Assessment: {severity['risk']}<br/>
            Oil Detected: {"Yes" if detected else "No"}
            """,
            styles["Normal"]
        ))

        elements.append(Spacer(1, 16))

        def add_img(arr, title):
            _, img_bytes = cv2.imencode(".png", arr)
            img_buf = BytesIO(img_bytes.tobytes())
            elements.append(Paragraph(title, styles["Heading3"]))
            elements.append(RLImage(img_buf, width=250, height=250))
            elements.append(Spacer(1, 12))

        add_img((prob_map * 255).astype(np.uint8), "Probability Map")
        add_img(binary_mask * 255, "Binary Oil Mask")
        add_img(overlay, "Oil Spill Overlay")

        doc.build(elements)
        buffer.seek(0)
        return buffer

    pdf_data = generate_pdf()

    st.markdown("<br>", unsafe_allow_html=True)

    st.download_button(
        label="📄 Download Analysis Report (PDF)",
        data=pdf_data,
        file_name="oil_spill_analysis_report.pdf",
        mime="application/pdf"
    )
