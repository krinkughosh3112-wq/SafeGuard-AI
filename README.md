# 🦺 SafeGuard AI — Construction Site Safety Monitor

![Python](https://img.shields.io/badge/Python-3.13-blue)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red)
![License](https://img.shields.io/badge/License-MIT-yellow)

> **AI-powered real-time PPE (Personal Protective Equipment) detection system for construction sites using YOLOv8 and Streamlit.**

---

## 🚀 Features

- 🤖 **Real-time PPE Detection** — Detects helmets, vests, gloves, boots, goggles and violations
- 📷 **4 Detection Modes** — Image Upload, Video Analysis, Live Webcam, Dataset Batch Test
- 📧 **Automatic Email Alerts** — Gmail alerts with violation snapshot image attached
- 📊 **Analytics Dashboard** — Charts, KPIs, trends and CSV export
- 🔐 **Secure Credentials** — Passwords stored in `.env` file
- 🎨 **Professional UI** — SafeGuard AI branding with light green theme

---

## 🛡️ PPE Classes Detected

| Class | Description |
|-------|-------------|
| ✅ Helmet | Safety helmet detected |
| ✅ Vest | Safety vest detected |
| ✅ Gloves | Safety gloves detected |
| ✅ Boots | Safety boots detected |
| ✅ Goggles | Safety goggles detected |
| 🚨 NO Helmet | Violation — no helmet |
| 🚨 NO Vest | Violation — no vest |
| 🚨 NO Gloves | Violation — no gloves |
| 🚨 NO Boots | Violation — no boots |
| 🚨 NO Goggles | Violation — no goggles |

---

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| Python 3.13 | Core language |
| YOLOv8 (Ultralytics) | PPE object detection |
| Streamlit | Web UI framework |
| OpenCV | Image/video processing |
| Gmail SMTP | Email alert delivery |
| python-dotenv | Secure credential management |
| Pandas | Analytics & data export |

---

## ⚙️ Installation

### 1. Clone the repository
```bash
git clone https://github.com/krinkughosh3112-wq/SafeGuard-AI.git
cd SafeGuard-AI
```

### 2. Install dependencies
```bash
pip install -r Requirements.txt
```

### 3. Add YOLOv8 model
Download or train your `best.pt` model and place it in the project root folder.

### 4. Create `.env` file
Create a `.env` file in the project root:
```
EMAIL_PASSWORD=your_gmail_app_password_here
```

> ⚠️ Use a **Gmail App Password** — not your regular Gmail password!
> Get it from: Google Account → Security → 2-Step Verification → App Passwords

---

## 🚀 How to Run

```bash
streamlit run app.py
```

Open browser at: **http://localhost:8501**

---

## 📱 Detection Modes

### 1. 📷 Image Upload
Upload any JPG/PNG construction site image for instant PPE analysis.

### 2. 🎬 Video Analysis
Upload MP4 videos for frame-by-frame detection with automatic email alerts.

### 3. 📹 Live Webcam
Real-time webcam feed with continuous PPE monitoring and instant alerts.

### 4. 📦 Dataset Batch Test
Bulk test multiple images with paginated results and report export.

### 5. 📊 Analytics Dashboard
View violation trends, KPI cards, peak hours chart and export CSV reports.

---

## 📧 Email Alert Setup

1. Enable **2-Step Verification** on your Gmail account
2. Go to **Google Account → Security → App Passwords**
3. Create a new App Password
4. Add it to your `.env` file as `EMAIL_PASSWORD`

---

## 📁 Project Structure

```
SafeGuard-AI/
├── app.py                  # Main Streamlit application
├── alert_system.py         # Email alert system
├── analytics_dashboard.py  # Analytics dashboard
├── Requirements.txt        # Python dependencies
├── .gitignore             # Git ignore file
└── README.md              # This file
```

> **Note:** `best.pt` (YOLOv8 model) and `.env` (credentials) are not included in the repository for security and size reasons.

---

## 🔧 Detection Settings

| Setting | Value | Reason |
|---------|-------|--------|
| Confidence Threshold | 0.15 | Low value catches small PPE items |
| IOU Threshold | 0.45 | Prevents duplicate bounding boxes |
| Inference Resolution | 1280px | High resolution for better accuracy |
| Test-Time Augmentation | Enabled | Catches items in multiple orientations |

---

## 👩‍💻 Developer

**Rinku Ghosh**
- GitHub: [@krinkughosh3112-wq](https://github.com/krinkughosh3112-wq)
- Email: krinkughosh3112@gmail.com

---

## 📄 License

This project is licensed under the MIT License.

---

⭐ **If you found this project useful, please give it a star!** ⭐
