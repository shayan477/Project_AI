# app.py

import streamlit as st
import cv2, tempfile, os, time
import torch.nn.functional as F
from human_action_predictor import HumanActionPredictor
from PIL import Image

# ─── PAGE CONFIG & THEME ────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🔥 Human Action Recognition 🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
      body { background-color: #0e1117; color: #fafafa; }
      .sidebar .sidebar-content { background-color: #1f2937; }
      .stButton>button { background-color: #2563eb; color: white; }
      .stSlider>div>div>div>div { background-color: #2563eb; }
    </style>
    """,
    unsafe_allow_html=True
)

# ─── SIDEBAR CONTROLS ──────────────────────────────────────────────────────────
st.sidebar.title("⚙️ Settings")

model_path     = st.sidebar.text_input("🗂️ Model checkpoint", "vit_model4.pth")
device_choice  = st.sidebar.radio("🔌 Device", ["auto","cpu","cuda"])
process_every  = st.sidebar.slider("🔢 Process every Nth frame", 1, 30, 5)
conf_thresh    = st.sidebar.slider("🎯 Confidence threshold", 0.0, 1.0, 0.5, 0.05)
uploaded_file = st.sidebar.file_uploader(
    "📥 Upload video",
    type=["mp4","avi","mov","mkv","mpg","mpeg"]
)


# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("## 🔥 Real‑Time Fighting Detection with ViT")

# ─── LOAD MODEL ────────────────────────────────────────────────────────────────
try:
    predictor = HumanActionPredictor(model_path, device=None if device_choice=="auto" else device_choice)
    st.sidebar.success("✅ Model loaded")
except Exception as e:
    st.sidebar.error(f"Model load failed: {e}")
    st.stop()

# ─── MAIN LAYOUT ───────────────────────────────────────────────────────────────
if uploaded_file:
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tmp.write(uploaded_file.read()); tmp.close()
    cap = cv2.VideoCapture(tmp.name)

    fps    = cap.get(cv2.CAP_PROP_FPS) or 25
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    col1, col2 = st.columns([3,1])

    # placeholders
    frame_ph = col1.empty()
    stats_ph = col2.empty()
    prog_ph  = st.progress(0)

    # counters
    frame_idx      = 0
    processed      = 0
    fighting_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # progress bar
        prog_ph.progress(min(frame_idx/total, 1.0))

        if frame_idx % process_every == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            cls, conf = predictor.predict(Image.fromarray(rgb))
            processed += 1

            if cls=="fighting" and conf>=conf_thresh:
                fighting_count +=1
                color = (220,38,38)   # red
                cv2.rectangle(frame, (0,0),(width,height), color, 8)
            else:
                color = (52,211,153)  # green

            label = f"{cls.upper()} ({conf:.2f})"
            cv2.putText(frame, label, (10,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            stats_ph.markdown(
                f"<div style='font-size:18px; text-align:center;'>"
                f"<span style='background-color: {('#dc2626' if cls=='fighting' else '#34d399')}; "
                f"padding:6px 12px; border-radius:8px; color:#ffffff;'>"
                f"{label}</span></div>",
                unsafe_allow_html=True
            )

        frame_ph.image(frame, use_container_width=True, channels="BGR")
        frame_idx += 1
        time.sleep(1/fps)

    cap.release()

    # final summary
    fight_pct = (fighting_count/processed*100) if processed else 0
    col2.markdown("### 📊 Summary")
    col2.metric("Processed frames", processed)
    col2.metric("Fighting frames", f"{fighting_count} ({fight_pct:.1f}%)")

    os.unlink(tmp.name)
