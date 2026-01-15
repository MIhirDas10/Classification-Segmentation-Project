import streamlit as st
import torch
import os
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt

from huggingface_hub import hf_hub_download  # ✅ add this
from model import UNetMultiTask

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

CLASS_NAMES = ["No Tumor", "Glioma", "Meningioma", "Pituitary"]
N_CLASSES = len(CLASS_NAMES)

# ✅ Hugging Face model location
HF_REPO_ID = "MihirDas/brisc-unet"
HF_FILENAME = "unet_multitask_4cls_best.pth"   # must match the file name in the repo

st.set_page_config(page_title="BriscApp — Brain Tumor Detection", layout="wide")

# ---------- Styling ----------
st.markdown(
    """
    <style>
      .block-container{
        padding-top: 2.2rem !important;
        padding-bottom: 0.8rem !important;
        max-width: 1200px;
      }
      [data-testid="stVerticalBlock"] {gap: 0.55rem;}
      [data-testid="stHorizontalBlock"] {gap: 0.55rem;}
      .stCaption {margin-top: -6px;}
      h1 {margin: 0.25rem 0 0.55rem 0 !important;}
      h2, h3 {margin-top: 0.55rem; margin-bottom: 0.25rem;}
      img {max-height: 240px; object-fit: contain;}
      [data-testid="stPyplotFigure"] {margin-top: -10px;}

      section[data-testid="stSidebar"]{
        overflow: hidden !important;
      }
      section[data-testid="stSidebar"] > div{
        overflow: hidden !important;
      }
      section[data-testid="stSidebar"] [data-testid="stSidebarContent"]{
        overflow: hidden !important;
      }

      div.stDownloadButton > button {
        padding: 0.25rem 0.6rem !important;
        font-size: 0.85rem !important;
        border-radius: 10px !important;
      }
      div.stDownloadButton {margin: 0.18rem 0 !important;}
      div.stDownloadButton > button {margin: 0 !important;}

      .brisc-logo{
        font-size: 26px;
        font-weight: 900;
        letter-spacing: 0.6px;
        padding: 10px 12px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.14);
        background: linear-gradient(180deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        box-shadow:
          inset 0 1px 0 rgba(255,255,255,0.10),
          inset 0 -1px 0 rgba(0,0,0,0.25),
          0 12px 20px rgba(0,0,0,0.28);
        text-align: center;
        margin-bottom: 10px;
      }
      .brisc-tag{
        text-align:center;
        font-size: 12px;
        opacity: 0.85;
        margin-top: -6px;
        margin-bottom: 12px;
      }

      .cls-title{
        text-align:left;
        font-size: 26px;
        font-weight: 800;
        margin: 0.2rem 0 0.7rem 0;
        letter-spacing: 0.2px;
      }

      .pred-wrap{display:flex; justify-content:flex-start; margin-bottom: 0.6rem;}
      .pred-chip{
        margin-left: 300px;
        padding: 10px 14px;
        border-radius: 14px;
        border: 1px solid rgba(255,255,255,0.18);
        background: rgba(255,255,255,0.04);
        box-shadow: 0 10px 22px rgba(0,0,0,0.30);
      }
      .pred-chip .pred-label{font-size: 13px; opacity: 0.9; margin: 0;}
      .pred-chip .pred-class{font-size: 22px; font-weight: 900; margin: 2px 0 0 0; line-height: 1.1;}
      .pred-chip .pred-conf{font-size: 13px; opacity: 0.9; margin: 6px 0 0 0;}

      .sidebar-hr{
        margin: 0.35rem 0 !important;
        border: none;
        border-top: 1px solid rgba(255,255,255,0.12);
      }
      .sidebar-title{
        margin: 0.25rem 0 0.15rem 0 !important;
        padding: 0 !important;
        font-weight: 800;
        font-size: 1.05rem;
      }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------- Preprocess ----------
def load_any_image_for_model(pil_img, image_size=256, expect_channels=1):
    img = pil_img.convert("RGB").resize((image_size, image_size))
    arr = np.array(img).astype(np.float32)

    if expect_channels == 1:
        arr = arr.mean(axis=2, keepdims=True)

    arr = arr / 255.0
    arr = (arr - 0.5) / 0.5  # same as training Normalize(mean=0.5,std=0.5)
    arr = np.transpose(arr, (2, 0, 1))
    x = torch.from_numpy(arr).unsqueeze(0)
    return x, img


@torch.no_grad()
def predict_pil(pil_img, model, threshold=0.5, expect_channels=1):
    model.eval()
    x, img_vis = load_any_image_for_model(pil_img, IMAGE_SIZE, expect_channels)
    x = x.to(DEVICE)

    seg_logits, cls_logits = model(x)

    seg_prob = torch.sigmoid(seg_logits)[0, 0].detach().cpu().numpy()
    seg_mask = (seg_prob >= threshold).astype(np.uint8)

    probs = torch.softmax(cls_logits, dim=1)[0].detach().cpu().numpy()
    pred_cls = int(np.argmax(probs))
    pred_conf = float(probs[pred_cls])

    return img_vis, seg_prob, seg_mask, probs, pred_cls, pred_conf


def overlay_mask(img_vis, seg_mask):
    base = np.array(img_vis.convert("RGB"))
    overlay = base.copy()
    overlay[seg_mask == 1] = (255, 0, 0)
    blended = (0.65 * base + 0.35 * overlay).astype(np.uint8)
    return blended


def mask_to_png_bytes(mask_u8):
    pil = Image.fromarray((mask_u8 * 255).astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


def rgb_to_png_bytes(arr_rgb_u8):
    pil = Image.fromarray(arr_rgb_u8.astype(np.uint8))
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue()


# ---------- Load model (HF download) ----------
@st.cache_resource
def load_model():
    """
    Downloads checkpoint from Hugging Face once, caches it, then loads weights.
    Falls back to local file if present (nice for local dev).
    """
    # 1) Prefer local file if it exists (optional)
    local_path = HF_FILENAME
    if os.path.exists(local_path):
        ckpt_path = local_path
    else:
        # 2) Download from HF
        ckpt_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="model"
        )

    model = UNetMultiTask(n_channels=1, seg_classes=1, cls_classes=4).to(DEVICE)

    sd = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(sd)
    model.eval()
    return model


model = load_model()

# ---------- Sidebar ----------
with st.sidebar:
    st.markdown('<div class="brisc-logo">BriscApp</div>', unsafe_allow_html=True)
    st.markdown('<div class="brisc-tag">Segmentation + 4-Class Classifier</div>', unsafe_allow_html=True)

    st.header("Controls")
    threshold = st.slider("Segmentation threshold", 0.05, 0.95, 0.50, 0.01)
    show_prob = st.checkbox("Show probability map", True)
    show_mask = st.checkbox("Show binary mask", False)
    show_overlay = st.checkbox("Show overlay", True)
    low_conf_thr = st.slider("Low-confidence warning threshold", 0.30, 0.90, 0.55, 0.01)


# ---------- Main UI ----------
st.title("Brain Tumor Detection")

uploaded = st.file_uploader("Upload an image (.png/.jpg)", type=["png", "jpg", "jpeg"])
if uploaded is None:
    st.info("Upload an image to get started.")
    st.stop()

pil = Image.open(uploaded)

img_vis, seg_prob, seg_mask, probs, pred_cls, pred_conf = predict_pil(
    pil, model, threshold=threshold, expect_channels=1
)
pred_name = CLASS_NAMES[pred_cls]

# Sidebar downloads + disclaimer at end
with st.sidebar:
    st.markdown('<hr class="sidebar-hr">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Downloads</div>', unsafe_allow_html=True)

    st.download_button(
        "⬇ Mask PNG",
        data=mask_to_png_bytes(seg_mask),
        file_name="pred_mask.png",
        mime="image/png",
        use_container_width=True,
    )

    if show_overlay:
        over = overlay_mask(img_vis, seg_mask)
        st.download_button(
            "⬇ Overlay PNG",
            data=rgb_to_png_bytes(over),
            file_name="overlay.png",
            mime="image/png",
            use_container_width=True,
        )

    st.markdown('<hr class="sidebar-hr">', unsafe_allow_html=True)
    st.warning("⚠️ **Medical Disclaimer**\nConsult with a doctor first as it can make error")


# ----- classification -----
st.markdown('<div class="cls-title">Classification</div>', unsafe_allow_html=True)

c1, c2 = st.columns([1.15, 1.0], gap="large")

with c1:
    st.markdown(
        f"""
        <div class="pred-wrap">
          <div class="pred-chip">
            <div class="pred-label">Predicted</div>
            <div class="pred-class">{pred_name}</div>
            <div class="pred-conf">Confidence: {pred_conf:.3f}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    if pred_conf < low_conf_thr:
        st.warning("Low confidence prediction. Consider checking image quality / orientation.")

    st.write("Top probabilities:\n")
    topk = np.argsort(probs)[::-1]
    for idx in topk:
        st.write(f"{CLASS_NAMES[idx]}: {probs[idx]:.3f}")

with c2:
    fig = plt.figure(figsize=(4.2, 2.4))
    plt.bar(CLASS_NAMES, probs)
    plt.ylim(0, 1)
    plt.xticks(rotation=15)
    plt.title("Class probability distribution")
    st.pyplot(fig, use_container_width=True)


# ----- segmentation -----
st.subheader("Segmentation")

tumor_pixels = int(seg_mask.sum())
total_pixels = int(seg_mask.size)
tumor_pct = 100.0 * tumor_pixels / max(1, total_pixels)
st.write(f"**Tumor area:** {tumor_pixels} / {total_pixels} pixels (**{tumor_pct:.2f}%**)")

col1, col2, col3 = st.columns(3, gap="small")

with col1:
    st.image(img_vis, caption="Resized input", use_container_width=False, width=260)

with col2:
    if show_prob:
        st.image(seg_prob, caption="Seg prob map", use_container_width=False, width=260, clamp=True)
    else:
        st.info("Probability map hidden.")

with col3:
    if show_overlay:
        over = overlay_mask(img_vis, seg_mask)
        st.image(over, caption="Overlay", use_container_width=False, width=260)
    else:
        st.info("Overlay hidden.")

if show_mask:
    st.image(seg_mask * 255, caption="Binary mask", use_container_width=False, width=320)
