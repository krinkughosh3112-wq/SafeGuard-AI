"""
==============================================================
  Real-time Alert System for Construction Site Safety Monitor
==============================================================
"""

import smtplib
import os
import json
import logging
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.image import MIMEImage
from datetime import datetime, timedelta
from typing import Optional
import cv2
import numpy as np
from dotenv import load_dotenv
load_dotenv(override=True)

# Optional: Twilio for SMS
try:
    from twilio.rest import Client as TwilioClient
    TWILIO_AVAILABLE = True
except ImportError:
    TWILIO_AVAILABLE = False

# ── Default Config ────────────────────────────
DEFAULT_CONFIG = {
    "email": {
        "enabled": True,
        "smtp_server": "smtp.gmail.com",
        "smtp_port": 587,
        "sender_email": "krinkughosh3112@gmail.com",
        "sender_password": "wsao ugoi ibwg vqdz",
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
}


class AlertSystem:

    def __init__(self, config: dict = None):
        self.config = config or DEFAULT_CONFIG
        self.logger = logging.getLogger("AlertSystem")
        logging.basicConfig(level=logging.INFO)
        self._last_alert_time: dict = {}
        self._consecutive_counts: dict = {}
        self.alert_history: list = []
        self._load_log()

    def process_frame_detections(self, violations, frame=None, location="Construction Site"):
        triggered = []

        if not violations:
            self._consecutive_counts.clear()
            return triggered

        for violation in set(violations):
            if violation not in self.config["alert_on_violations"]:
                continue

            self._consecutive_counts[violation] = self._consecutive_counts.get(violation, 0) + 1

            if self._consecutive_counts[violation] < self.config["min_consecutive_frames"]:
                continue

            if not self._is_cooldown_expired(violation):
                continue

            self._last_alert_time[violation] = datetime.now()
            self._log_alert(violation, location)
            triggered.append(violation)

            if self.config["email"]["enabled"]:
                print(f"📧 Attempting to send email for: {violation}")
                result = self._send_email_alert(violation, location, frame)
                print(f"📧 Email result: {result}")

            if self.config["sms"]["enabled"] and TWILIO_AVAILABLE:
                self._send_sms_alert(violation, location)

        for v in list(self._consecutive_counts):
            if v not in violations:
                self._consecutive_counts[v] = 0

        return triggered

    def _send_email_alert(self, violation, location, frame=None):
        cfg = self.config["email"]
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build message
        msg = MIMEMultipart("mixed")
        msg["Subject"] = f"Safety Violation: {violation} at {location}"
        msg["From"] = cfg["sender_email"]
        msg["To"] = ", ".join(cfg["recipient_emails"])

        # HTML body
        html_body = f"""
        <html><body>
        <div style="font-family:Arial,sans-serif;max-width:600px;margin:auto;">
          <div style="background:#dc2626;color:white;padding:20px;border-radius:8px 8px 0 0;">
            <h2 style="margin:0;">Safety Violation Detected</h2>
          </div>
          <div style="background:#fff8f8;border:2px solid #dc2626;padding:20px;border-radius:0 0 8px 8px;">
            <p><strong>Violation:</strong> {violation}</p>
            <p><strong>Location:</strong> {location}</p>
            <p><strong>Time:</strong> {timestamp}</p>
            <p>Please take immediate corrective action.</p>
          </div>
        </div>
        </body></html>
        """

        msg.attach(MIMEText(html_body, "html"))

        # Attach image if available
        if frame is not None and cfg.get("send_image", True):
            try:
                _, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                img_bytes = buffer.tobytes()
                img_attachment = MIMEImage(img_bytes, name="violation_snapshot.jpg")
                img_attachment.add_header("Content-Disposition", "attachment", filename="violation_snapshot.jpg")
                msg.attach(img_attachment)
                print("📎 Image attached successfully")
            except Exception as e:
                print(f"⚠️ Could not attach image: {e}")

        # Send email
        try:
            # Always reload .env fresh to get latest password
            load_dotenv(override=True)
            fresh_password = os.getenv("EMAIL_PASSWORD") or cfg["sender_password"]
            print(f"📧 Connecting to {cfg['smtp_server']}:{cfg['smtp_port']}...")
            server = smtplib.SMTP(cfg["smtp_server"], cfg["smtp_port"], timeout=30)
            server.ehlo()
            server.starttls()
            server.ehlo()
            print(f"📧 Logging in as {cfg['sender_email']}...")
            server.login(cfg["sender_email"], fresh_password)
            print(f"📧 Sending to {cfg['recipient_emails']}...")
            server.sendmail(cfg["sender_email"], cfg["recipient_emails"], msg.as_string())
            server.quit()
            print(f"✅ EMAIL SENT SUCCESSFULLY for: {violation}")
            return True
        except smtplib.SMTPAuthenticationError as e:
            print(f"❌ AUTH ERROR - wrong email/password: {e}")
        except smtplib.SMTPConnectError as e:
            print(f"❌ CONNECTION ERROR - cannot reach Gmail: {e}")
        except smtplib.SMTPException as e:
            print(f"❌ SMTP ERROR: {e}")
        except Exception as e:
            import traceback
            print(f"❌ GENERAL ERROR: {e}")
            traceback.print_exc()
        return False

    def _send_sms_alert(self, violation, location):
        if not TWILIO_AVAILABLE:
            return False
        cfg = self.config["sms"]
        timestamp = datetime.now().strftime("%H:%M:%S")
        body = f"SAFETY ALERT\nViolation: {violation}\nLocation: {location}\nTime: {timestamp}"
        try:
            client = TwilioClient(cfg["account_sid"], cfg["auth_token"])
            for number in cfg["to_numbers"]:
                client.messages.create(body=body, from_=cfg["from_number"], to=number)
            return True
        except Exception as e:
            print(f"❌ SMS ERROR: {e}")
            return False

    def _is_cooldown_expired(self, violation):
        last = self._last_alert_time.get(violation)
        if last is None:
            return True
        return (datetime.now() - last).total_seconds() >= self.config["cooldown_seconds"]

    def _log_alert(self, violation, location):
        entry = {"timestamp": datetime.now().isoformat(), "violation": violation, "location": location}
        self.alert_history.append(entry)
        self._save_log()

    def _save_log(self):
        try:
            with open(self.config["log_file"], "w") as f:
                json.dump(self.alert_history, f, indent=2)
        except Exception as e:
            print(f"⚠️ Could not save log: {e}")

    def _load_log(self):
        self.alert_history = []
        self._last_alert_time = {}

    def get_alert_history(self, hours=24):
        cutoff = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if datetime.fromisoformat(a["timestamp"]) > cutoff]

    def get_stats(self):
        if not self.alert_history:
            return {"total": 0}
        counts = {}
        for alert in self.alert_history:
            counts[alert["violation"]] = counts.get(alert["violation"], 0) + 1
        return {"total": len(self.alert_history), "by_type": counts}