# 🦺 SafeGuard AI — Construction Site Safety Monitor

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **AI-powered real-time PPE (Personal Protective Equipment) 
> detection system for construction sites using YOLOv8 
> and Streamlit.**

---

## Screenshots

###  Main Dashboard
![Dashboard](screenshots/dashboard.png)

###  PPE Detection in Action
![Detection](screenshots/detection.png)

###  Analytics Dashboard
![Analytics](screenshots/analytics.png)

###  Email Alert Sample
![Email Alert](screenshots/email_alert.png)

> *Add your own screenshots in the `screenshots/` folder 
> after running the app.*

---

##  Project Overview

SafeGuard AI is a **computer vision-based safety monitoring system**
built for construction sites. It uses the **YOLOv8 object detection 
model** to detect whether workers are wearing the required 
Personal Protective Equipment (PPE) in real time.

When a violation is detected, the system **automatically sends 
an email alert** with a snapshot of the violation — helping site 
supervisors act immediately.

This project demonstrates skills in:
- Deep Learning & Computer Vision (YOLOv8)
- Real-time video processing (OpenCV)
- Web app development (Streamlit)
- Automated alert systems (Gmail SMTP)
- Data analytics and reporting (Pandas)

---

##  Features

-  **Real-time PPE Detection** — Detects helmets, vests, 
  gloves, boots, goggles and violations
-  **4 Detection Modes** — Image Upload, Video Analysis, 
  Live Webcam, Dataset Batch Test
-  **Automatic Email Alerts** — Gmail alerts with violation 
  snapshot image attached
-  **Analytics Dashboard** — Charts, KPIs, trends and 
  CSV export
-  **Secure Credentials** — Passwords stored in `.env` file
-  **Professional UI** — SafeGuard AI branding with 
  light green theme

---

##  PPE Classes Detected

| Class | Type | Description |
|-------|------|-------------|
| ✅ Helmet | Safe | Safety helmet detected |
| ✅ Vest | Safe | Safety vest detected |
| ✅ Gloves | Safe | Safety gloves detected |
| ✅ Boots | Safe | Safety boots detected |
| ✅ Goggles | Safe | Safety goggles detected |
| 🚨 NO Helmet | Violation | Worker without helmet |
| 🚨 NO Vest | Violation | Worker without vest |
| 🚨 NO Gloves | Violation | Worker without gloves |
| 🚨 NO Boots | Violation | Worker without boots |
| 🚨 NO Goggles | Violation | Worker without goggles |

---

##  Tech Stack

| Technology | Version | Purpose |
|-----------|---------|---------|
| Python | 3.13 | Core language |
| YOLOv8 (Ultralytics) | Latest | PPE object detection |
| Streamlit | 1.x | Web UI framework |
| OpenCV | 4.x | Image/video processing |
| Gmail SMTP | — | Email alert delivery |
| python-dotenv | Latest | Secure credential management |
| Pandas | Latest | Analytics & data export |
| NumPy | Latest | Numerical operations |

---

##  Project Structure
```
SafeGuard-AI/
├── app.py                    # Main Streamlit application
├── alert_system.py           # Email alert system
├── analytics_dashboard.py    # Analytics dashboard
├── video_detection.py        # Video detection module
├── test_sample_videos.py     # Batch testing module
├── sample_videos/            # Sample test videos
├── screenshots/              # App screenshots for README
├── Requirements.txt          # Python dependencies
├── .gitignore                # Git ignore file
└── README.md                 # Project documentation
```

>  `best.pt` (YOLOv8 model) and `.env` (credentials) are 
> **not included** in the repository for security and size reasons.

---

##  Installation & Setup

### Step 1 — Clone the Repository
```bash
git clone https://github.com/krinkughosh3112-wq/SafeGuard-AI.git
cd SafeGuard-AI
```

### Step 2 — Install Dependencies
```bash
pip install -r Requirements.txt
```

### Step 3 — Add YOLOv8 Model
Download or train your `best.pt` model and place it 
in the project root folder.

### Step 4 — Create `.env` File
Create a `.env` file in the project root:
```
EMAIL_PASSWORD=your_gmail_app_password_here
```

>  Use a **Gmail App Password** — not your regular Gmail password!  
> Get it from:  
> Google Account → Security → 2-Step Verification → App Passwords

---

##  How to Run
```bash
streamlit run app.py
```

Open your browser at: **http://localhost:8501**

---

##  Detection Modes

### 1️⃣ Image Upload
Upload any JPG/PNG construction site image for 
instant PPE analysis with bounding box visualisation.

### 2️⃣ Video Analysis
Upload MP4 videos for frame-by-frame detection 
with automatic email alerts on violation detection.

### 3️⃣ Live Webcam
Real-time webcam feed with continuous PPE monitoring 
and instant alerts when violations are detected.

### 4️⃣ Dataset Batch Test
Bulk test multiple images at once with paginated 
results and full report export to CSV.

### 5️⃣ Analytics Dashboard
View violation trends, KPI cards, peak hours chart 
and export detailed CSV reports for reporting.

---

##  Email Alert Setup

1. Enable **2-Step Verification** on your Gmail account
2. Go to **Google Account → Security → App Passwords**
3. Create a new App Password for "Mail"
4. Copy the generated password
5. Add it to your `.env` file as shown above

---

## ⚙️ Detection Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Confidence Threshold | 0.15 | Low value catches small PPE items |
| IOU Threshold | 0.45 | Prevents duplicate bounding boxes |
| Inference Resolution | 1280px | High resolution for better accuracy |
| Test-Time Augmentation | Enabled | Catches items in multiple orientations |

---

##  Use Cases

- Construction site safety monitoring
- Factory floor PPE compliance checking
- Security camera violation detection
- Safety audit reporting and analytics
- Real-time supervisor alert system

---

##  Key Learnings

- Training and deploying YOLOv8 for custom object detection
- Real-time video stream processing using OpenCV
- Building multi-page Streamlit web applications
- Integrating automated email alerts using Gmail SMTP
- Secure credential management using `.env` files
- Data analytics and visualisation using Pandas

---

## Future Improvements

- [ ] Add support for RTSP camera streams
- [ ] Deploy on cloud (AWS / Azure / Streamlit Cloud)
- [ ] Add WhatsApp alerts using Twilio API
- [ ] Train model on larger custom dataset
- [ ] Add worker ID tracking using face recognition
- [ ] Generate automated PDF safety reports

---

## 👩‍💻 Developer

**Rinku Ghosh**
-  GitHub: [@krinkughosh3112-wq](https://github.com/krinkughosh3112-wq)
-  Email: krinkughosh3112@gmail.com
-  LinkedIn: [Rinku Ghosh K](https://www.linkedin.com/in/k-rinku-ghosh3112/)

---

## 📄 License

This project is licensed under the **MIT License** — 
feel free to use and modify with attribution.

---

⭐ **If you found this project useful, please give it a star!** ⭐
