import streamlit as st
from alert_system import AlertSystem
from analytics_dashboard import render_analytics_dashboard
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import os
import tempfile
import time
import random
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
load_dotenv(override=True)
alert_system = AlertSystem()

# Load YOLO model with error handling
@st.cache_resource
def load_model():
    try:
        model = YOLO('best.pt')
        # Test model with a dummy image to verify it works
        dummy_img = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = model(dummy_img, verbose=False)
        st.sidebar.success("✅ Model loaded successfully!")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ Failed to load model: {e}")
        st.error("""
        ### Model Loading Error!
        
        Please ensure:
        1. `best.pt` file is in the current directory
        2. It's a valid YOLOv8 model file
        3. Required dependencies are installed:
        ```
        pip install ultralytics opencv-python-headless pillow streamlit pandas
        ```
        """)
        return None

model = load_model()

# Define class names
CLASS_NAMES = {
    0: "Boots",
    1: "Gloves",
    2: "Goggles",
    3: "Helmet",
    4: "NO Boots",
    5: "NO Gloves",
    6: "NO Goggles",
    7: "NO Helmet",
    8: "NO Vest",
    9: "Vest"
}

# Dataset paths
DATASET_PATH = 'PPE-detection-1'
TEST_IMAGES_DIR = os.path.join(DATASET_PATH, 'test', 'images')

st.set_page_config(
    page_title="Site Safety AI", 
    layout="wide",
    page_icon="🚧"
)

# ========== ENHANCED CUSTOM CSS WITH COLORS ==========
st.markdown("""
<style>
    /* Main background with light gradient */
    .stApp {
        background: linear-gradient(135deg, #e8f5e9 0%, #c8e6c9 100%);
    }
    
    /* Sidebar styling with light color */
    section[data-testid="stSidebar"] {
        background-color: #e8eff9;
        border-right: 1px solid #d1d9e6;
    }
    
    /* Main content area cards */
    .main .block-container {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
        font-weight: 600;
    }
    
    /* Tabs styling - Light colors for different modes */
    div[data-testid="stTabs"] {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 5px;
    }
    
    button[data-testid="stTab"] {
        background-color: #f0f4f8;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        color: #4a5568;
        margin-right: 5px;
    }
    
    button[data-testid="stTab"]:hover {
        background-color: #e1e8f0;
        transform: translateY(-2px);
    }
    
    /* Active tab styling */
    button[data-testid="stTab"][aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    /* Different colors for different modes */
    /* Image Upload Tab */
    button[data-testid="stTab"]:nth-child(1) {
        border-left: 4px solid #4299e1;
    }
    
    /* Video Analysis Tab */
    button[data-testid="stTab"]:nth-child(2) {
        border-left: 4px solid #48bb78;
    }
    
    /* Live Webcam Tab */
    button[data-testid="stTab"]:nth-child(3) {
        border-left: 4px solid #ed8936;
    }
    
    /* Dataset Batch Test Tab */
    button[data-testid="stTab"]:nth-child(4) {
        border-left: 4px solid #9f7aea;
    }
    
    /* Buttons styling */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(50, 50, 93, 0.11), 0 1px 3px rgba(0, 0, 0, 0.08);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 7px 14px rgba(50, 50, 93, 0.1), 0 3px 6px rgba(0, 0, 0, 0.08);
    }
    
    /* Primary button */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        font-size: 1.1em;
    }
    
    /* Secondary button */
    .stButton > button[kind="secondary"] {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Metrics cards */
    div[data-testid="stMetric"] {
        background-color: #ffffff;
        padding: 15px;
        border-radius: 10px;
        border-left: 4px solid #4299e1;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        margin-bottom: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f7fafc;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        font-weight: 600;
        color: #2d3748;
    }
    
    .streamlit-expanderContent {
        background-color: #ffffff;
        border-radius: 0 0 8px 8px;
        border: 1px solid #e2e8f0;
        border-top: none;
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        background-color: #f8fafc;
        border: 2px dashed #cbd5e0;
        border-radius: 10px;
        padding: 20px;
    }
    
    /* Webcam container */
    .webcam-container {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.1);
    }
    
    /* Alert boxes */
    .violation-alert {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #dc2626;
        margin: 10px 0;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .safe-status {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #059669;
        margin: 10px 0;
        font-weight: 600;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #4facfe 0%, #00f2fe 100%);
    }
    
    /* Sidebar widgets */
    .stSidebar .stSlider > div > div {
        background-color: #ffffff;
    }
    
    /* Radio buttons */
    .stRadio > div {
        background-color: #f8fafc;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Checkbox styling */
    .stCheckbox > label {
        color: #2d3748;
        font-weight: 500;
    }
    
    /* Selectbox styling */
    .stSelectbox > div > div {
        background-color: #ffffff;
        border: 2px solid #e2e8f0;
        border-radius: 8px;
    }
    
    /* Success/Error/Info/Warning messages */
    .stAlert {
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background-color: #ffffff;
        border-radius: 10px;
        border: 1px solid #e2e8f0;
    }
    
    /* Video player styling */
    video {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    
    /* Image containers */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        border: 2px solid #e2e8f0;
    }
    
    /* Horizontal rule styling */
    hr {
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #667eea, transparent);
        margin: 2rem 0;
    }
    
    /* Caption styling */
    .stCaption {
        color: #4a5568;
        font-style: italic;
    }
    
    /* Spinner styling */
    .stSpinner > div {
        border-color: #667eea transparent transparent transparent;
    }
    
    /* Tooltip styling */
    .tooltip {
        background-color: #2d3748 !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px !important;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Rajdhani:wght@700&family=Share+Tech+Mono&display=swap" rel="stylesheet">
<div style="display:flex; align-items:center; gap:16px; margin-bottom:16px; padding:16px 20px;
            background:linear-gradient(135deg, #1a3c2e 0%, #2d5a3d 100%);
            border-radius:12px; border-left:5px solid #f5c518;
            box-shadow:0 4px 15px rgba(0,0,0,0.15);">
    <div style="background:linear-gradient(135deg,#f5c518,#ff6b35); padding:10px 14px;
                border-radius:8px; font-size:2em;">🦺</div>
    <div>
        <div style="font-family:'Rajdhani',sans-serif; font-weight:700; font-size:2.2em;
                    color:#f5c518; letter-spacing:3px; text-transform:uppercase; line-height:1;">
            SafeGuard AI
        </div>
        <div style="font-family:'Share Tech Mono',monospace; color:#a8d5b5; font-size:0.78em;
                    letter-spacing:2px; text-transform:uppercase; margin-top:3px;">
            Construction Site Safety Monitor — Powered by YOLOv8
        </div>
    </div>
    <div style="margin-left:auto; font-family:'Share Tech Mono',monospace; font-size:0.75em;
                color:#f5c518; background:rgba(245,197,24,0.15); padding:8px 14px;
                border-radius:6px; border:1px solid rgba(245,197,24,0.4);">
        ● SYSTEM ONLINE
    </div>
</div>
""", unsafe_allow_html=True)

# Sidebar for Navigation with enhanced styling
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            padding: 20px; 
            border-radius: 10px; 
            color: white;
            margin-bottom: 20px;">
    <h2 style="color: white; margin: 0;">Control Panel</h2>
    <p style="color: #e2e8f0; font-size: 0.9em; margin: 5px 0 0 0;">Monitor Safety Compliance</p>
</div>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "Choose Mode", 
    ["Image Upload", "Video Analysis", "Live Webcam", "Dataset Batch Test", "Analytics Dashboard"],
    key="mode_select"
)

# ========== ENHANCED DETECTION SETTINGS ==========
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
            color: white; 
            padding: 15px; 
            border-radius: 10px;
            margin-bottom: 20px;">
    <h4 style="color: white; margin: 0;">🔧 Advanced Detection Settings</h4>
</div>
""", unsafe_allow_html=True)

# Add advanced tuning parameters
confidence_threshold = st.sidebar.slider(
    "Confidence Threshold", 
    0.05, 1.0, 0.15, 0.05,
    help="Lower values (0.10-0.15) help detect small objects like shoes"
)

iou_threshold = st.sidebar.slider(
    "IOU Threshold", 
    0.1, 1.0, 0.45, 0.05,
    help="Higher values (0.45-0.50) prevent person boxes from hiding shoes"
)

inference_size = st.sidebar.selectbox(
    "Inference Resolution",
    [640, 1280, 1536],
    index=1,
    help="1280+ recommended for detecting small objects like shoes"
)

use_augmentation = st.sidebar.checkbox(
    "Use Test-Time Augmentation",
    value=True,
    help="Check multiple orientations for better detection"
)

agnostic_nms = st.sidebar.checkbox(
    "Use Agnostic NMS",
    value=True,
    help="Prevents different classes from suppressing each other"
)

# Location settings
location_name = st.sidebar.text_input("Site Location", "Construction Site 1", 
                                     help="Name of construction site/location")
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background: linear-gradient(135deg, #f5576c 0%, #f093fb 100%);
            color: white; padding: 15px; border-radius: 10px; margin-bottom: 20px;">
    <h4 style="color: white; margin: 0;">🔔 Alert Settings</h4>
</div>
""", unsafe_allow_html=True)

with st.sidebar.expander("⚙️ Configure Alerts", expanded=False):
    enable_email = st.checkbox("Enable Email Alerts", value=False)
    enable_sms   = st.checkbox("Enable SMS Alerts",   value=False)
    sender_email    = st.text_input("Sender Email")
    sender_password = st.text_input("App Password", type="password")
    recipient_emails_raw = st.text_input("Recipient Emails (comma-separated)")
    recipient_emails = [e.strip() for e in recipient_emails_raw.split(",") if e.strip()]
    twilio_sid   = st.text_input("Twilio Account SID")
    twilio_token = st.text_input("Twilio Auth Token", type="password")
    twilio_from  = st.text_input("Twilio From Number")
    sms_to_raw   = st.text_input("SMS Recipients (comma-separated)")
    sms_to_numbers = [n.strip() for n in sms_to_raw.split(",") if n.strip()]
    alert_cooldown = st.slider("Alert Cooldown (seconds)", 10, 300, 60)
    min_frames = st.slider("Min Consecutive Frames", 1, 10, 3)

alert_config = {
    "email": {
        "enabled": enable_email,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": sender_email,
        "sender_password": sender_password,
        "recipient_emails": recipient_emails or ["placeholder@example.com"],
        "send_image": True,
    },
    "sms": {
        "enabled": enable_sms,
        "account_sid": twilio_sid,
        "auth_token": twilio_token,
        "from_number": twilio_from,
        "to_numbers": sms_to_numbers or ["+10000000000"],
    },
    "cooldown_seconds": alert_cooldown,
    "min_consecutive_frames": min_frames,
    "alert_on_violations": ["NO Helmet", "NO Vest", "NO Gloves", "NO Boots", "NO Goggles"],
    "log_file": "violation_alerts.json",
}

# Force rebuild alert system every time sidebar changes
alert_system = AlertSystem({
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "krinkughosh3112@gmail.com",
        "sender_password": "PASTE_YOUR_APP_PASSWORD_HERE",
        "recipient_emails": ["rinku8143kumari@gmail.com"],
        "send_image": True,
    },
    "sms": {
        "enabled": False,
        "account_sid": "",
        "auth_token": "",
        "from_number": "",
        "to_numbers": [],
    },
    "cooldown_seconds": 10,
    "min_consecutive_frames": 1,
    "alert_on_violations": ["NO Helmet", "NO Vest", "NO Gloves", "NO Boots", "NO Goggles"],
    "log_file": "violation_alerts.json",
})

# Add some color to sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style="background-color: #f7fafc; padding: 15px; border-radius: 10px; border-left: 4px solid #4299e1;">
    <h4 style="color: #2c3e50; margin: 0;">📊 Quick Stats</h4>
    <p style="color: #4a5568; font-size: 0.9em; margin: 5px 0;">Monitor PPE compliance in real-time</p>
</div>
""", unsafe_allow_html=True)

# ========== FUNCTION DEFINITIONS ==========

def enhanced_detection(image, conf=0.15, iou=0.45, imgsz=1280, augment=True, agnostic=True):
    """
    Enhanced detection function with better parameters for small objects
    """
    if model is None:
        return None
    
    try:
        results = model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=imgsz,
            augment=augment,
            agnostic_nms=agnostic,
            verbose=False
        )
        return results
    except Exception as e:
        st.error(f"Detection error: {e}")
        return None

def analyze_detections(results):
    """
    Analyze detection results and return statistics
    """
    if results is None or len(results) == 0 or results[0].boxes is None:
        return {
            'person_count': 0,
            'shoe_count': 0,
            'no_shoe_count': 0,
            'helmet_count': 0,
            'no_helmet_count': 0,
            'vest_count': 0,
            'no_vest_count': 0,
            'gloves_count': 0,
            'no_gloves_count': 0,
            'total_detections': 0,
            'violations': [],
            'safety_items': []
        }
    
    stats = {
        'person_count': 0,
        'shoe_count': 0,
        'no_shoe_count': 0,
        'helmet_count': 0,
        'no_helmet_count': 0,
        'vest_count': 0,
        'no_vest_count': 0,
        'gloves_count': 0,
        'no_gloves_count': 0,
        'total_detections': 0,
        'violations': [],
        'safety_items': []
    }
    
    detections = results[0].boxes
    if detections is not None and detections.cls is not None:
        for cls_id in detections.cls.cpu().numpy():
            cls_id_int = int(cls_id)
            stats['total_detections'] += 1

            if cls_id_int == 3:    # Helmet
                stats['helmet_count'] += 1
                stats['safety_items'].append('Helmet')
            elif cls_id_int == 7:  # NO Helmet
                stats['no_helmet_count'] += 1
                stats['violations'].append('NO Helmet')
            elif cls_id_int == 9:  # Vest
                stats['vest_count'] += 1
                stats['safety_items'].append('Vest')
            elif cls_id_int == 8:  # NO Vest
                stats['no_vest_count'] += 1
                stats['violations'].append('NO Vest')
            elif cls_id_int == 1:  # Gloves
                stats['gloves_count'] += 1
                stats['safety_items'].append('Gloves')
            elif cls_id_int == 5:  # NO Gloves
                stats['no_gloves_count'] += 1
                stats['violations'].append('NO Gloves')
            elif cls_id_int == 0:  # Boots
                stats['shoe_count'] += 1
                stats['safety_items'].append('Boots')
            elif cls_id_int == 4:  # NO Boots
                stats['no_shoe_count'] += 1
                stats['violations'].append('NO Boots')
            elif cls_id_int == 2:  # Goggles
                stats['safety_items'].append('Goggles')
            elif cls_id_int == 6:  # NO Goggles
                stats['violations'].append('NO Goggles')
                stats['violations'].append('NO Gloves')
    
    return stats

def process_video_file(video_path, location="site_1"):
    """Process video and display results"""
    st.subheader("🔍 Analyzing Video for Safety Compliance...")
    
    # Create columns for video and stats
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Video display placeholder
        video_placeholder = st.empty()
        
    with col2:
        # Stats placeholder
        stats_placeholder = st.empty()
    
    # Progress bar
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error(f"Failed to open video: {video_path}")
        return
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Info box with styling
    st.markdown(f"""
    <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                color: white; 
                padding: 15px; 
                border-radius: 10px;
                margin-bottom: 20px;">
        <strong>📹 Video Analysis</strong><br>
        File: {os.path.basename(video_path)} | Size: {width}x{height} | FPS: {fps} | Frames: {total_frames}
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize counters
    frame_count = 0
    safety_stats = {name: 0 for name in CLASS_NAMES.values()}
    violation_frames = 0
    
    # Process video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection with enhanced parameters
        results = enhanced_detection(
            frame,
            conf=confidence_threshold,
            iou=iou_threshold,
            imgsz=inference_size,
            augment=use_augmentation,
            agnostic=agnostic_nms
        )
        
        if results is not None:
            processed_frame = results[0].plot()
            
            # Update stats
            detections = results[0].boxes
            frame_violations = []
            
            if detections is not None and detections.cls is not None:
                for idx, cls_id in enumerate(detections.cls.cpu().numpy()):
                    cls_name = CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}")
                    safety_stats[cls_name] = safety_stats.get(cls_name, 0) + 1
                    
                    # Check for violations
                    if int(cls_id) in [4, 5, 6, 7, 8]:  # NO- classes
                        frame_violations.append(cls_name)
            
            # Check if any violations were detected in this frame
            violation_detected = len(frame_violations) > 0
            if violation_detected:
                violation_frames += 1
            
            # Highlight violation
            if violation_detected:
                # Draw violation warning on frame
                cv2.putText(processed_frame, "⚠️ SAFETY VIOLATION", 
                           (width//2 - 200, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                           1.2, (0, 0, 255), 3, cv2.LINE_AA)
                cv2.rectangle(processed_frame, (width//2 - 220, 20), 
                             (width//2 + 220, 80), (0, 0, 255), 3)
                
                # List violations on frame
                y_offset = 120
                for i, violation in enumerate(set(frame_violations)):
                    cv2.putText(processed_frame, f"• {violation}", 
                               (width//2 - 150, y_offset + i*35), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            # Add frame counter
            cv2.putText(processed_frame, f"Frame: {frame_count}/{total_frames}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Resize for better display (max width 800px)
            display_width = min(800, width)
            display_height = int(display_width * height / width)
            processed_frame = cv2.resize(processed_frame, (display_width, display_height))
            
            # Convert to RGB for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display frame
            video_placeholder.image(processed_frame_rgb, 
                                  caption=f"Frame {frame_count}/{total_frames} - {'⚠️ VIOLATION' if violation_detected else '✅ SAFE'}")
            
            # Update stats in sidebar
            with stats_placeholder.container():
                st.markdown("""
                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 4px solid #48bb78;">
                    <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📈 Live Stats</h4>
                </div>
                """, unsafe_allow_html=True)
                st.metric("Frames Processed", frame_count)
                st.metric("Violation Frames", violation_frames)
                
                # Show top detections
                top_detections = sorted([(k, v) for k, v in safety_stats.items() if v > 0], 
                                      key=lambda x: x[1], reverse=True)[:5]
                if top_detections:
                    st.markdown("**Top Detections:**")
                    for item, count in top_detections:
                        if "NO" in item:
                            st.markdown(f"<span style='color: #e53e3e;'>⚠️ **{item}:** {count}</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"<span style='color: #38a169;'>✅ **{item}:** {count}</span>", unsafe_allow_html=True)
        
        # Update progress
        progress = frame_count / total_frames
        progress_bar.progress(min(progress, 1.0))
        status_text.text(f"Processing: {frame_count}/{total_frames} frames ({progress*100:.1f}%)")
        
        # Control speed based on original FPS
        if fps > 0:
            time.sleep(1.0 / fps)
        else:
            time.sleep(0.03)  # Default to ~30 FPS
    
    cap.release()
    
    st.success("✅ Video analysis complete!")
    
    # Results summary with colored cards
    st.markdown("""
    <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 15px;
                margin: 20px 0;">
        <h2 style="color: white; text-align: center;">📊 Analysis Results</h2>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 4px solid #4299e1;">
            <h4 style="color: #2c3e50; margin: 0 0 10px 0;">Frame Analysis</h4>
        </div>
        """, unsafe_allow_html=True)
        st.metric("Total Frames", frame_count)
        violation_percent = (violation_frames/frame_count*100) if frame_count > 0 else 0
        st.metric("Violation %", f"{violation_percent:.1f}%", 
                 delta=f"{violation_frames} frames" if violation_frames > 0 else None,
                 delta_color="inverse")
    
    with col2:
        st.markdown("""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 4px solid #48bb78;">
            <h4 style="color: #2c3e50; margin: 0 0 10px 0;">Detection Summary</h4>
        </div>
        """, unsafe_allow_html=True)
        total_detections = sum(safety_stats.values())
        st.metric("Total Detections", total_detections)
        if total_detections > 0:
            st.metric("Detections/Frame", f"{(total_detections/frame_count):.1f}")
    
    with col3:
        st.markdown("""
        <div style="background-color: #ffffff; padding: 20px; border-radius: 10px; border-left: 4px solid #ed8936;">
            <h4 style="color: #2c3e50; margin: 0 0 10px 0;">Violations</h4>
        </div>
        """, unsafe_allow_html=True)
        violation_count = sum(v for k, v in safety_stats.items() if "NO" in k)
        st.metric("Total Violations", violation_count)
    
    # Detailed breakdown
    with st.expander("📈 Detailed Statistics", expanded=False):
        df = pd.DataFrame(list(safety_stats.items()), columns=['Equipment', 'Count'])
        df = df[df['Count'] > 0].sort_values('Count', ascending=False)
        
        if not df.empty:
            # Add category column
            df['Category'] = df['Equipment'].apply(lambda x: 'Violation' if 'NO' in x else 'Safety Equipment')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.dataframe(df, use_container_width=True)
            
            with col2:
                # Create separate charts for violations and safety equipment
                violations_df = df[df['Category'] == 'Violation']
                safety_df = df[df['Category'] == 'Safety Equipment']
                
                if not violations_df.empty:
                    st.subheader("🚨 Violations")
                    st.bar_chart(violations_df.set_index('Equipment')['Count'])
                
                if not safety_df.empty:
                    st.subheader("✅ Safety Equipment")
                    st.bar_chart(safety_df.set_index('Equipment')['Count'])
        else:
            st.info("No safety equipment detected in this video.")

# ========== MAIN APP CODE ==========

if model is None:
    st.stop()  # Stop execution if model failed to load

# --- MODE 1: IMAGE UPLOAD ---
if app_mode == "Image Upload":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #4299e1 0%, #3182ce 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white;">📷 Upload Construction Site Image</h2>
        <p style="color: #e2e8f0;">Upload an image to detect safety equipment compliance</p>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file:
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display original image
            img = Image.open(uploaded_file)
            st.image(img, caption="Original Image", use_column_width=True)
        
        with col2:
            st.markdown("""
            <div style="background-color: #f7fafc; padding: 20px; border-radius: 10px; border: 1px solid #e2e8f0;">
                <h4 style="color: #2c3e50; margin: 0 0 15px 0;">⚙️ Detection Settings</h4>
            </div>
            """, unsafe_allow_html=True)
            st.info(f"Using: Conf={confidence_threshold}, IOU={iou_threshold}, Size={inference_size}")
        
        # Run detection with enhanced parameters
        with st.spinner("🔍 Detecting safety equipment..."):
            results = enhanced_detection(
                img,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=inference_size,
                augment=use_augmentation,
                agnostic=agnostic_nms
            )
            
            if results is not None:
                # Display results
                res_plotted = results[0].plot()
                st.image(res_plotted, caption="Safety Equipment Detection", use_column_width=True)
                
                # Analyze detections
                stats = analyze_detections(results)
                print(f"VIOLATIONS FOUND: {stats['violations']}")
                print(f"EMAIL ENABLED: {enable_email}")
                print(f"SENDER: {sender_email}")
                frame_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                triggered = alert_system.process_frame_detections(stats['violations'], frame_bgr, location_name)
                print(f"TRIGGERED: {triggered}")
                if triggered:
                   st.toast(f"🚨 Alert sent: {', '.join(triggered)}", icon="🚨")
                
                # Display statistics
                st.markdown("""
                <div style="background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%); 
                            color: white; 
                            padding: 20px; 
                            border-radius: 15px;
                            margin: 20px 0;">
                    <h2 style="color: white;">📊 Detection Statistics</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display in columns
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Workers Detected", stats['person_count'])
                    st.metric("Safety Shoes", stats['shoe_count'])
                
                with col2:
                    st.metric("NO Shoes", stats['no_shoe_count'])
                    st.metric("Helmets", stats['helmet_count'])
                
                with col3:
                    # Status indicator
                    if stats['violations']:
                        st.markdown('<div class="violation-alert">⚠️ SAFETY VIOLATION DETECTED!</div>', unsafe_allow_html=True)
                        st.markdown(f"**Violations:** {', '.join(set(stats['violations']))}")
                    else:
                        st.markdown('<div class="safe-status">✅ ALL SAFETY EQUIPMENT PROPERLY WORN</div>', unsafe_allow_html=True)
                
                # Detailed breakdown
                with st.expander("View Detailed Breakdown", expanded=False):
                    # Show person-to-shoe ratio if applicable
                    if stats['person_count'] > 0:
                        shoe_coverage = (stats['shoe_count'] / stats['person_count']) * 100
                        st.progress(min(shoe_coverage/100, 1.0))
                        st.caption(f"Shoe coverage: {shoe_coverage:.1f}% of workers")
                    
                    detections = results[0].boxes
                    if detections is not None and detections.cls is not None:
                        for cls_id in detections.cls.cpu().numpy():
                            cls_name = CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}")
                            if "NO" in cls_name:
                                st.markdown(f"<span style='color: #e53e3e;'>🔴 **{cls_name}**</span>", unsafe_allow_html=True)
                            else:
                                st.markdown(f"<span style='color: #38a169;'>🟢 **{cls_name}**</span>", unsafe_allow_html=True)
                    else:
                        st.warning("No safety equipment detected in the image.")

# --- MODE 2: VIDEO ANALYSIS ---
elif app_mode == "Video Analysis":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #48bb78 0%, #38a169 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white;">📹 Construction Site Video Analysis</h2>
        <p style="color: #e2e8f0;">Upload or select a video to analyze safety compliance</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Create tabs for different video sources
    tab1, tab2 = st.tabs(["📤 Upload Your Video", "📁 Sample Videos"])
    
    video_to_process = None
    
    with tab1:
        st.markdown("""
        <div style="background-color: #f0fff4; padding: 20px; border-radius: 10px; border: 2px dashed #c6f6d5; margin-bottom: 20px;">
            <h4 style="color: #22543d; margin: 0 0 10px 0;">Upload Video File</h4>
            <p style="color: #4a5568;">Supported formats: MP4, MOV, AVI, M4V</p>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_video = st.file_uploader("Choose a video file", 
                                         type=["mp4", "mov", "avi", "m4v"])
        if uploaded_video:
            # Save to temp file
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
            tfile.write(uploaded_video.read())
            video_to_process = tfile.name
            
            # Show preview
            st.video(uploaded_video)
    
    with tab2:
        st.markdown("""
        <div style="background-color: #f0fff4; padding: 20px; border-radius: 10px; border: 2px dashed #c6f6d5; margin-bottom: 20px;">
            <h4 style="color: #22543d; margin: 0 0 10px 0;">Sample Videos</h4>
            <p style="color: #4a5568;">Select from pre-loaded sample videos</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("Select from sample videos:")
        
        # Check if sample_videos folder exists
        if not os.path.exists('sample_videos'):
            st.warning("'sample_videos' folder not found. Creating it...")
            os.makedirs('sample_videos')
            st.info("Add your MP4 videos to the 'sample_videos' folder and refresh.")
        else:
            # List sample videos
            sample_files = [f for f in os.listdir('sample_videos') 
                          if f.lower().endswith(('.mp4', '.mov', '.avi', '.m4v'))]
            
            if sample_files:
                # Display videos in a grid
                cols = st.columns(2)
                for i, video_file in enumerate(sample_files):
                    video_path = os.path.join('sample_videos', video_file)
                    
                    with cols[i % 2]:
                        # Show video thumbnail and info
                        cap = cv2.VideoCapture(video_path)
                        if cap.isOpened():
                            ret, frame = cap.read()
                            if ret:
                                # Resize thumbnail
                                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                                thumbnail = cv2.resize(frame_rgb, (320, 180))
                                
                                # Display thumbnail with styled container
                                st.markdown(f"""
                                <div style="background-color: #ffffff; padding: 10px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 10px;">
                                    <h5 style="color: #2d3748; margin: 0 0 10px 0;">{video_file}</h5>
                                </div>
                                """, unsafe_allow_html=True)
                                st.image(thumbnail, use_column_width=True)
                                
                                # Video info
                                fps = int(cap.get(cv2.CAP_PROP_FPS))
                                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                                
                                st.caption(f"📏 {width}x{height} ⏱️ {fps}fps")
                                
                                # Process button
                                if st.button(f"🔍 Analyze", key=f"btn_{i}"):
                                    video_to_process = video_path
                            cap.release()
            else:
                st.info("No videos found in 'sample_videos' folder.")
    
    # Process video if selected
    if video_to_process:
        if st.button("▶️ Start Safety Analysis", type="primary", use_container_width=True):
            process_video_file(video_to_process, location_name)

# --- MODE 3: LIVE WEBCAM ---
elif app_mode == "Live Webcam":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white;">🎥 Live Safety Feed</h2>
        <p style="color: #e2e8f0;">Monitor construction site safety in real-time</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Camera settings
    with st.sidebar:
        st.markdown("""
        <div style="background: linear-gradient(135deg, #ed8936 0%, #dd6b20 100%); 
                    color: white; 
                    padding: 15px; 
                    border-radius: 10px;
                    margin-bottom: 20px;">
            <h4 style="color: white; margin: 0;">⚙️ Live Detection Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        show_detections = st.checkbox("Show Bounding Boxes", True)
        show_stats_overlay = st.checkbox("Show Statistics Overlay", True)
        alert_on_violation = st.checkbox("Alert on Violation", True)
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                    color: white; 
                    padding: 15px; 
                    border-radius: 10px;
                    margin-bottom: 20px;">
            <h4 style="color: white; margin: 0;">📷 Camera Settings</h4>
        </div>
        """, unsafe_allow_html=True)
        
        # Camera settings
        camera_source = st.selectbox("Camera Source", ["Default (0)", "External (1)", "External (2)"])
        camera_index = int(camera_source.split("(")[1].replace(")", ""))
        
        # Resolution settings
        resolution = st.selectbox("Resolution", ["640x480", "800x600", "1024x768", "1280x720"])
        width, height = map(int, resolution.split('x'))
    
    # Main webcam area - LARGER DISPLAY
    st.markdown("""
    <div class="webcam-container">
        <h3 style="color: white; text-align: center; margin: 0 0 15px 0;">Live Construction Site Feed</h3>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Webcam feed placeholder with larger size
        FRAME_WINDOW = st.empty()
        st.caption("🎯 Safety Monitoring Active - Real-time Detection")
    
    with col2:
        # Real-time stats
        stats_placeholder = st.empty()
        violation_placeholder = st.empty()
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Webcam controls with styled buttons
    st.markdown("""
    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; margin: 20px 0; border: 1px solid #e2e8f0;">
        <h4 style="color: #2c3e50; margin: 0 0 15px 0;">🎮 Camera Controls</h4>
    </div>
    """, unsafe_allow_html=True)
    
    col_controls = st.columns([1, 1, 2])
    with col_controls[0]:
        run = st.button('▶️ Start Webcam', type='primary', use_container_width=True)
    with col_controls[1]:
        stop = st.button('⏹️ Stop Webcam', use_container_width=True)
    
    if run:
        # Initialize webcam
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            st.error(f"❌ Cannot access camera {camera_index}. Please check connection.")
            st.stop()
        
        # Set camera resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        # Initialize statistics
        frame_counter = 0
        safety_stats = {name: 0 for name in CLASS_NAMES.values()}
        violation_history = []
        
        while run:
            ret, frame = cap.read()
            if not ret: 
                st.error("Failed to capture frame")
                break
            
            frame_counter += 1
            
            # Flip horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Run detection with enhanced parameters
            results = enhanced_detection(
                frame,
                conf=confidence_threshold,
                iou=iou_threshold,
                imgsz=inference_size,
                augment=use_augmentation,
                agnostic=agnostic_nms
            )
            
            if results is not None:
                # Process results
                violation_detected = False
                current_violations = []
                current_safety = []
                
                detections = results[0].boxes
                if detections is not None and detections.cls is not None:
                    for cls_id in detections.cls.cpu().numpy():
                        cls_name = CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}")
                        safety_stats[cls_name] = safety_stats.get(cls_name, 0) + 1
                        if int(cls_id) in [4, 5, 6, 7, 8]:
                            violation_detected = True
                            current_violations.append(cls_name)
                        else:
                            current_safety.append(cls_name)

                    if violation_detected and alert_on_violation:
                        violation_history.append(frame_counter)
                        if len(violation_history) > 10:
                            violation_history.pop(0)

                # ✅ STEP 1: Define display_frame FIRST
                if show_detections:
                    display_frame = results[0].plot()
                else:
                    display_frame = frame.copy()

                # ✅ STEP 2: Send alert using display_frame
                if current_violations:
                    print(f"🚨 Webcam violations: {current_violations}")
                    triggered = alert_system.process_frame_detections(
                        current_violations, display_frame, location_name)
                    print(f"📧 Email triggered: {triggered}")
                    if triggered:
                        for v in triggered:
                            st.toast(f"🚨 Alert sent: {v}", icon="🚨")
                
                # Add overlays
                y_offset = 40
                
                # Frame counter
                cv2.putText(display_frame, f"Frame: {frame_counter}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                y_offset += 30
                
                # Safety status
                status_text = "⚠️ VIOLATION" if violation_detected else "✅ SAFE"
                status_color = (0, 0, 255) if violation_detected else (0, 255, 0)
                cv2.putText(display_frame, f"Status: {status_text}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
                
                # Show statistics overlay
                if show_stats_overlay:
                    y_offset += 30
                    cv2.putText(display_frame, "Detections:", (10, y_offset), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    y_offset += 25
                    
                    # Show safety equipment (green)
                    for i, item in enumerate(set(current_safety[:3])):  # Show first 3 unique
                        cv2.putText(display_frame, f"  ✅ {item}", 
                                   (10, y_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (0, 255, 0), 2)
                    
                    # Show violations (red)
                    violation_offset = y_offset + len(set(current_safety[:3])) * 25
                    for i, item in enumerate(set(current_violations[:3])):  # Show first 3 unique
                        cv2.putText(display_frame, f"  ❌ {item}", 
                                   (10, violation_offset + i*25), cv2.FONT_HERSHEY_SIMPLEX, 
                                   0.5, (0, 0, 255), 2)
                
                # Highlight if violation detected
                if violation_detected:
                    cv2.rectangle(display_frame, (0, 0), (width, height), (0, 0, 255), 10)
                    cv2.putText(display_frame, "SAFETY VIOLATION!", 
                               (width//2 - 150, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                               1.2, (0, 0, 255), 3, cv2.LINE_AA)
                
                # Display in Streamlit - FULL COLUMN WIDTH
                FRAME_WINDOW.image(display_frame, channels="BGR", use_column_width=True)
                
                # Update sidebar statistics
                with stats_placeholder.container():
                    st.markdown("""
                    <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border-left: 4px solid #ed8936;">
                        <h4 style="color: #2c3e50; margin: 0 0 10px 0;">📈 Live Statistics</h4>
                    </div>
                    """, unsafe_allow_html=True)
                    st.metric("Frame", frame_counter)
                    st.metric("Status", "VIOLATION" if violation_detected else "SAFE")
                    
                    # Current detections
                    if current_safety or current_violations:
                        st.write("**Current Frame:**")
                        
                        # Safety equipment
                        if current_safety:
                            unique_safe = set(current_safety)
                            for item in unique_safe:
                                count = current_safety.count(item)
                                st.success(f"✅ {item}: {count}")
                        
                        # Violations
                        if current_violations:
                            unique_violations = set(current_violations)
                            for item in unique_violations:
                                count = current_violations.count(item)
                                st.error(f"❌ {item}: {count}")
                
                # Show alert if violation detected
                if violation_detected:
                    with violation_placeholder.container():
                        violation_types = ', '.join(set(current_violations))
                        st.markdown(f"""
                        <div class="violation-alert">
                            <h4>⚠️ SAFETY VIOLATION DETECTED!</h4>
                            <p><strong>Frame:</strong> {frame_counter}</p>
                            <p><strong>Violations:</strong> {violation_types}</p>
                            <p><strong>Total Violations:</strong> {len(violation_history)}</p>
                        </div>
                        """, unsafe_allow_html=True)
            
            # Check for stop signal
            if stop:
                run = False
                break
        
        cap.release()
        st.info("Webcam stopped.")
    
    else:
        # Show placeholder when webcam is not running
        placeholder_img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Create a gradient background
        for i in range(480):
            color = int(150 + (i / 480) * 105)
            cv2.line(placeholder_img, (0, i), (640, i), (color, color, color), 1)
        
        cv2.putText(placeholder_img, "CONSTRUCTION SITE MONITOR", (50, 200), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(placeholder_img, "Click 'Start Webcam' to begin", (100, 250), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
        cv2.putText(placeholder_img, f"Camera: {camera_source}", (150, 300), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        cv2.putText(placeholder_img, f"Resolution: {resolution}", (150, 330), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (150, 150, 150), 1)
        
        FRAME_WINDOW.image(placeholder_img, channels="BGR", use_column_width=True)

# --- MODE 4: DATASET BATCH TEST ---
elif app_mode == "Dataset Batch Test":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white;">📊 Automated Testing from Roboflow Dataset</h2>
        <p style="color: #e2e8f0;">Batch test multiple images for comprehensive analysis</p>
    </div>
    """, unsafe_allow_html=True)
    
    if os.path.exists(TEST_IMAGES_DIR):
        # Get all available images
        all_images = [f for f in os.listdir(TEST_IMAGES_DIR) 
                     if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        if not all_images:
            st.warning(f"No images found in {TEST_IMAGES_DIR}")
        else:
            st.markdown(f"""
            <div style="background-color: #faf5ff; padding: 15px; border-radius: 10px; border-left: 4px solid #9f7aea; margin-bottom: 20px;">
                <h4 style="color: #44337a; margin: 0;">📁 Found {len(all_images)} images in the dataset folder</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Let user choose how many images to test
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0;">
                    <h5 style="color: #2c3e50; margin: 0 0 10px 0;">Test Configuration</h5>
                </div>
                """, unsafe_allow_html=True)
                # Show a slider to select number of images (1 to all, default 10)
                num_images = st.slider(
                    "Number of images to test:",
                    min_value=1,
                    max_value=min(50, len(all_images)),  # Limit to 50 max for performance
                    value=min(10, len(all_images)),
                    help="Select how many images to test from your dataset"
                )
            
            with col2:
                st.markdown("""
                <div style="background-color: #ffffff; padding: 15px; border-radius: 10px; border: 1px solid #e2e8f0;">
                    <h5 style="color: #2c3e50; margin: 0 0 10px 0;">Test Mode</h5>
                </div>
                """, unsafe_allow_html=True)
                # Option to test random images or sequential
                test_mode = st.radio(
                    "Test mode:",
                    ["First N images", "Random N images"],
                    help="Test first N images or random N images from dataset"
                )
            
            # Select images based on mode
            if test_mode == "First N images":
                test_files = all_images[:num_images]
            else:
                test_files = random.sample(all_images, min(num_images, len(all_images)))
            
            # Add a progress bar for batch processing
            if st.button(f"🚀 Run Batch Test ({len(test_files)} Images)", type="primary", use_container_width=True):
                
                # Initialize progress
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Process images
                all_results = []
                all_violations = []
                batch_safety_stats = {name: 0 for name in CLASS_NAMES.values()}
                
                for idx, image_name in enumerate(test_files):
                    img_path = os.path.join(TEST_IMAGES_DIR, image_name)
                    
                    # Use enhanced detection
                    results = enhanced_detection(
                        img_path,
                        conf=confidence_threshold,
                        iou=iou_threshold,
                        imgsz=inference_size,
                        augment=use_augmentation,
                        agnostic=agnostic_nms
                    )
                    
                    if results is not None:
                        all_results.append((image_name, results))
                        
                        # Process detections
                        detections = results[0].boxes
                        image_violations = []
                        
                        if detections is not None and detections.cls is not None:
                            for cls_id in detections.cls.cpu().numpy():
                                cls_name = CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}")
                                batch_safety_stats[cls_name] = batch_safety_stats.get(cls_name, 0) + 1
                                
                                # Check for violations
                                if int(cls_id) in [4, 5, 6, 7, 8]:
                                    image_violations.append(cls_name)
                        
                        if image_violations:
                            all_violations.extend(image_violations)
                    
                    # Update progress
                    progress = (idx + 1) / len(test_files)
                    progress_bar.progress(min(progress, 1.0))
                    status_text.text(f"Processing image {idx + 1}/{len(test_files)}: {image_name}")
                
                # Clear progress indicators
                progress_bar.empty()
                status_text.empty()
                
                st.success(f"✅ Successfully processed {len(all_results)} images!")
                
                # Display summary statistics
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%); 
                            color: white; 
                            padding: 20px; 
                            border-radius: 15px;
                            margin: 20px 0;">
                    <h2 style="color: white; text-align: center;">📈 Batch Test Summary</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Calculate overall statistics
                total_detections = sum(batch_safety_stats.values())
                violation_count = len(all_violations)
                images_with_violations = len([1 for _, results in all_results if any(int(cls_id) in [4, 5, 6, 7, 8] for cls_id in (results[0].boxes.cls.cpu().numpy() if results[0].boxes is not None and results[0].boxes.cls is not None else []))])
                
                # Display summary metrics
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Images", len(all_results))
                
                with col2:
                    st.metric("Total Detections", total_detections)
                
                with col3:
                    st.metric("Violations Found", violation_count)
                
                with col4:
                    avg_per_image = total_detections / len(all_results) if all_results else 0
                    st.metric("Avg/Image", f"{avg_per_image:.1f}")
                
                # Show detection distribution
                detection_summary = {k: v for k, v in batch_safety_stats.items() if v > 0}
                if detection_summary:
                    df_summary = pd.DataFrame(
                        list(detection_summary.items()), 
                        columns=['Equipment', 'Count']
                    ).sort_values('Count', ascending=False)
                    
                    st.markdown("#### 📊 Detection Distribution")
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        st.dataframe(df_summary, use_container_width=True)
                    
                    with col2:
                        st.bar_chart(df_summary.set_index('Equipment'))
                
                # Display all processed images in a grid
                st.markdown("---")
                st.markdown("""
                <div style="background: linear-gradient(135deg, #9f7aea 0%, #805ad5 100%); 
                            color: white; 
                            padding: 20px; 
                            border-radius: 15px;
                            margin: 20px 0;">
                    <h2 style="color: white; text-align: center;">🖼️ Processed Images ({len(all_results)} total)</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Display in grid with pagination
                images_per_page = 6  # Show 6 images per page
                total_pages = (len(all_results) + images_per_page - 1) // images_per_page
                
                if total_pages > 1:
                    # Add pagination controls
                    page_num = st.number_input(
                        "Page",
                        min_value=1,
                        max_value=total_pages,
                        value=1,
                        step=1
                    )
                    start_idx = (page_num - 1) * images_per_page
                    end_idx = min(start_idx + images_per_page, len(all_results))
                    
                    st.caption(f"Showing images {start_idx + 1}-{end_idx} of {len(all_results)} (Page {page_num}/{total_pages})")
                else:
                    start_idx = 0
                    end_idx = len(all_results)
                
                # Display current page of images
                current_images = all_results[start_idx:end_idx]
                
                # Create grid display (2 columns)
                for i in range(0, len(current_images), 2):
                    cols = st.columns(2)
                    
                    for col_idx in range(2):
                        if i + col_idx < len(current_images):
                            image_name, results = current_images[i + col_idx]
                            if results is not None:
                                res_plotted = results[0].plot()
                                
                                with cols[col_idx]:
                                    # Display image with detection count
                                    st.markdown(f"""
                                    <div style="background-color: #ffffff; padding: 10px; border-radius: 10px; border: 1px solid #e2e8f0; margin-bottom: 15px;">
                                        <h5 style="color: #2d3748; margin: 0 0 10px 0;">{image_name}</h5>
                                    </div>
                                    """, unsafe_allow_html=True)
                                    st.image(res_plotted, use_column_width=True)
                                    
                                    # Show detection counts
                                    detections = results[0].boxes
                                    if detections is not None and detections.cls is not None:
                                        unique, counts = np.unique(detections.cls.cpu().numpy(), return_counts=True)
                                        
                                        # Create a compact summary
                                        safe_count = 0
                                        violation_count_img = 0
                                        
                                        for cls_id, count in zip(unique, counts):
                                            cls_name = CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}")
                                            
                                            if int(cls_id) in [4, 5, 6, 7, 8]:
                                                violation_count_img += count
                                                st.markdown(f"<span style='color: #e53e3e;'>⚠️ **{cls_name}:** {count}</span>", unsafe_allow_html=True)
                                            else:
                                                safe_count += count
                                                st.markdown(f"<span style='color: #38a169;'>✅ **{cls_name}:** {count}</span>", unsafe_allow_html=True)
                                        
                                        # Summary line
                                        if violation_count_img > 0:
                                            st.error(f"**Violations:** {violation_count_img}")
                                        else:
                                            st.success("**No violations detected**")
                
                # Add option to download results
                st.markdown("---")
                st.markdown("""
                <div style="background-color: #faf5ff; padding: 20px; border-radius: 10px; border-left: 4px solid #9f7aea; margin-bottom: 20px;">
                    <h4 style="color: #44337a; margin: 0 0 10px 0;">💾 Export Results</h4>
                    <p style="color: #4a5568;">Generate and download a comprehensive report</p>
                </div>
                """, unsafe_allow_html=True)
                
                if st.button("📥 Generate Results Report", type="primary", use_container_width=True):
                    # Create a summary report
                    report_lines = []
                    report_lines.append("=" * 60)
                    report_lines.append("PPE Detection Batch Test Report")
                    report_lines.append("=" * 60)
                    report_lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
                    report_lines.append(f"Location: {location_name}")
                    report_lines.append(f"Total Images Tested: {len(all_results)}")
                    report_lines.append(f"Total Detections: {total_detections}")
                    report_lines.append(f"Violations Found: {violation_count}")
                    report_lines.append(f"Images with Violations: {images_with_violations}")
                    report_lines.append("")
                    report_lines.append("Detection Summary:")
                    report_lines.append("-" * 30)
                    
                    for equipment, count in batch_safety_stats.items():
                        if count > 0:
                            report_lines.append(f"{equipment}: {count}")
                    
                    report_lines.append("")
                    report_lines.append("Image-wise Results:")
                    report_lines.append("-" * 30)
                    
                    for image_name, results in all_results:
                        if results is not None:
                            detections = results[0].boxes
                            violation_count_img = 0
                            safe_count = 0
                            violations_in_image = []
                            
                            if detections is not None and detections.cls is not None:
                                for cls_id in detections.cls.cpu().numpy():
                                    if int(cls_id) in [4, 5, 6, 7, 8]:
                                        violation_count_img += 1
                                        violations_in_image.append(CLASS_NAMES.get(int(cls_id), f"Class {int(cls_id)}"))
                                    else:
                                        safe_count += 1
                            
                            status = "VIOLATION" if violation_count_img > 0 else "SAFE"
                            violation_text = f" ({', '.join(set(violations_in_image))})" if violations_in_image else ""
                            report_lines.append(f"{image_name}: {safe_count} safe, {violation_count_img} violations{violation_text} - {status}")
                    
                    report_text = "\n".join(report_lines)
                    
                    # Provide download button
                    st.download_button(
                        label="📄 Download Report (TXT)",
                        data=report_text,
                        file_name=f"ppe_detection_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                        mime="text/plain"
                    )
                    
                    st.success("Report generated! Click download button above.")
    else:
        st.error(f"Folder not found! Make sure '{DATASET_PATH}' is in the project folder.")

# --- MODE 5: ANALYTICS DASHBOARD ---
elif app_mode == "Analytics Dashboard":
    st.markdown("""
    <div style="background: linear-gradient(135deg, #2d6a4f 0%, #1a472a 100%); 
                color: white; 
                padding: 20px; 
                border-radius: 15px;
                margin-bottom: 20px;">
        <h2 style="color: white;">📊 Analytics Dashboard</h2>
        <p style="color: #a8d5b5;">View violation trends, statistics and export reports</p>
    </div>
    """, unsafe_allow_html=True)
    render_analytics_dashboard()

# Handle invalid mode
else:
    st.warning("Please select a valid mode from the sidebar.")

# Add footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #718096; font-size: 0.9em; padding: 20px;">
    <p>🚧 <strong>Construction Site Safety Monitor</strong> | Powered by YOLOv8 | v2.0 - Enhanced Detection</p>
    <p>Real-time safety monitoring for construction sites with improved small-object detection</p>
</div>
""", unsafe_allow_html=True)