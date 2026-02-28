"""
==============================================================
  Analytics Dashboard for Construction Site Safety Monitor
==============================================================
Add this to your app.py in the MODE selection section.

STEP 1: Add "Analytics Dashboard" to your app_mode selectbox:
    app_mode = st.sidebar.selectbox(
        "Choose Mode",
        ["Image Upload", "Video Analysis", "Live Webcam", "Dataset Batch Test", "Analytics Dashboard"],
    )

STEP 2: Add this import at the top of app.py:
    from analytics_dashboard import render_analytics_dashboard

STEP 3: Add this at the bottom of app.py (before the footer):
    elif app_mode == "Analytics Dashboard":
        render_analytics_dashboard()
==============================================================
"""

import streamlit as st
import json
import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import Counter


def load_alert_history(log_file="violation_alerts.json"):
    """Load alert history from JSON log file."""
    if not os.path.exists(log_file):
        return []
    try:
        with open(log_file) as f:
            return json.load(f)
    except Exception:
        return []


def generate_demo_data():
    """Generate demo data if no real data exists yet."""
    import random
    violations = ["NO Helmet", "NO Vest", "NO Gloves", "NO Shoes"]
    locations = ["Construction Site 1", "Zone A", "Zone B", "Entry Gate"]
    data = []
    now = datetime.now()
    for i in range(60):
        ts = now - timedelta(hours=random.randint(0, 168))
        data.append({
            "timestamp": ts.isoformat(),
            "violation": random.choice(violations),
            "location": random.choice(locations),
        })
    return sorted(data, key=lambda x: x["timestamp"])


def render_analytics_dashboard():
    """Render the full analytics dashboard."""

    st.markdown("""
    <div style="background: linear-gradient(135deg, #1a1a2e 0%, #16213e 50%, #0f3460 100%);
                color: white; padding: 30px; border-radius: 20px; margin-bottom: 30px;
                box-shadow: 0 20px 60px rgba(0,0,0,0.3);">
        <h1 style="color: white; margin: 0; font-size: 2.2em;">📊 Safety Analytics Dashboard</h1>
        <p style="color: #a0aec0; margin: 8px 0 0 0; font-size: 1.1em;">
            Real-time violation trends and compliance insights
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Load Data ─────────────────────────────────
    history = load_alert_history()

    # Show demo data option if no real data
    use_demo = False
    if not history:
        st.info("📭 No alert history found yet. Showing **demo data** so you can see what the dashboard looks like!")
        use_demo = True
        history = generate_demo_data()
    else:
        col_info, col_demo = st.columns([3, 1])
        with col_info:
            st.success(f"✅ Loaded **{len(history)}** real alerts from your system")
        with col_demo:
            use_demo = st.checkbox("Show demo data instead", value=False)
            if use_demo:
                history = generate_demo_data()

    # ── Convert to DataFrame ──────────────────────
    df = pd.DataFrame(history)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    df["day_name"] = df["timestamp"].dt.strftime("%A")
    df["week"] = df["timestamp"].dt.isocalendar().week

    # ── Time Filter ───────────────────────────────
    st.markdown("### 🗓️ Time Range")
    time_filter = st.select_slider(
        "Select time range",
        options=["Last 24 Hours", "Last 7 Days", "Last 30 Days", "All Time"],
        value="Last 7 Days"
    )

    now = datetime.now()
    if time_filter == "Last 24 Hours":
        df = df[df["timestamp"] >= now - timedelta(hours=24)]
    elif time_filter == "Last 7 Days":
        df = df[df["timestamp"] >= now - timedelta(days=7)]
    elif time_filter == "Last 30 Days":
        df = df[df["timestamp"] >= now - timedelta(days=30)]

    if df.empty:
        st.warning("No data for the selected time range. Try 'All Time'.")
        return

    # ── KPI Cards ─────────────────────────────────
    st.markdown("---")
    st.markdown("### 📈 Key Metrics")

    total = len(df)
    unique_violations = df["violation"].nunique()
    most_common = df["violation"].value_counts().index[0] if total > 0 else "N/A"
    today_count = len(df[df["date"] == datetime.now().date()])

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #667eea, #764ba2);
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(102,126,234,0.4);">
            <div style="font-size: 2.5em; font-weight: bold;">{total}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">Total Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #f093fb, #f5576c);
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(245,87,108,0.4);">
            <div style="font-size: 2.5em; font-weight: bold;">{today_count}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">Today's Alerts</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #4facfe, #00f2fe);
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(79,172,254,0.4);">
            <div style="font-size: 2.5em; font-weight: bold;">{unique_violations}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">Violation Types</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #43e97b, #38f9d7);
                    padding: 20px; border-radius: 15px; text-align: center; color: white;
                    box-shadow: 0 4px 15px rgba(67,233,123,0.4);">
            <div style="font-size: 1.3em; font-weight: bold;">{most_common.replace('NO ', '❌ ')}</div>
            <div style="font-size: 0.9em; opacity: 0.9;">Most Common</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts Row 1 ──────────────────────────────
    st.markdown("### 📊 Violation Breakdown")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div style="background:#ffffff; padding:15px; border-radius:12px;
                    border-left:4px solid #667eea; margin-bottom:10px;">
            <h4 style="color:#2c3e50; margin:0;">🔴 Violations by Type</h4>
        </div>
        """, unsafe_allow_html=True)
        violation_counts = df["violation"].value_counts().reset_index()
        violation_counts.columns = ["Violation", "Count"]
        st.bar_chart(violation_counts.set_index("Violation"), color="#667eea")

    with col2:
        st.markdown("""
        <div style="background:#ffffff; padding:15px; border-radius:12px;
                    border-left:4px solid #f5576c; margin-bottom:10px;">
            <h4 style="color:#2c3e50; margin:0;">📍 Violations by Location</h4>
        </div>
        """, unsafe_allow_html=True)
        if "location" in df.columns:
            location_counts = df["location"].value_counts().reset_index()
            location_counts.columns = ["Location", "Count"]
            st.bar_chart(location_counts.set_index("Location"), color="#f5576c")

    # ── Charts Row 2 ──────────────────────────────
    st.markdown("### 📅 Violation Trends Over Time")

    daily = df.groupby("date").size().reset_index(name="Alerts")
    daily["date"] = pd.to_datetime(daily["date"])
    daily = daily.sort_values("date")

    st.markdown("""
    <div style="background:#ffffff; padding:15px; border-radius:12px;
                border-left:4px solid #4facfe; margin-bottom:10px;">
        <h4 style="color:#2c3e50; margin:0;">📈 Daily Alert Trend</h4>
    </div>
    """, unsafe_allow_html=True)
    st.line_chart(daily.set_index("date")["Alerts"], color="#4facfe")

    # ── Hourly Heatmap ────────────────────────────
    st.markdown("### ⏰ Peak Violation Hours")

    hourly = df.groupby("hour").size().reset_index(name="Count")
    all_hours = pd.DataFrame({"hour": range(24)})
    hourly = all_hours.merge(hourly, on="hour", how="left").fillna(0)
    hourly["Count"] = hourly["Count"].astype(int)
    hourly["Time"] = hourly["hour"].apply(lambda h: f"{h:02d}:00")

    st.markdown("""
    <div style="background:#ffffff; padding:15px; border-radius:12px;
                border-left:4px solid #43e97b; margin-bottom:10px;">
        <h4 style="color:#2c3e50; margin:0;">🕐 Alerts by Hour of Day</h4>
    </div>
    """, unsafe_allow_html=True)
    st.bar_chart(hourly.set_index("Time")["Count"], color="#43e97b")

    # ── Violation Type Over Time ──────────────────
    st.markdown("### 🔍 Violation Type Trends")

    pivot = df.groupby(["date", "violation"]).size().unstack(fill_value=0)
    pivot.index = pd.to_datetime(pivot.index)

    st.markdown("""
    <div style="background:#ffffff; padding:15px; border-radius:12px;
                border-left:4px solid #f093fb; margin-bottom:10px;">
        <h4 style="color:#2c3e50; margin:0;">📊 Each Violation Type Over Time</h4>
    </div>
    """, unsafe_allow_html=True)
    st.line_chart(pivot)

    # ── Recent Alerts Table ───────────────────────
    st.markdown("### 📋 Recent Alerts Log")

    recent = df.sort_values("timestamp", ascending=False).head(20).copy()
    recent["timestamp"] = recent["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    recent["Status"] = recent["violation"].apply(
        lambda x: "🔴 VIOLATION" if "NO" in x else "✅ OK"
    )

    display_cols = ["timestamp", "violation", "location", "Status"] if "location" in recent.columns else ["timestamp", "violation", "Status"]
    recent_display = recent[display_cols].rename(columns={
        "timestamp": "⏰ Time",
        "violation": "⚠️ Violation",
        "location": "📍 Location",
    })

    st.dataframe(recent_display, use_container_width=True, height=400)

    # ── Export ────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💾 Export Data")

    col1, col2 = st.columns(2)

    with col1:
        csv = df[["timestamp", "violation", "location"]].to_csv(index=False) if "location" in df.columns else df[["timestamp", "violation"]].to_csv(index=False)
        st.download_button(
            label="📥 Download CSV Report",
            data=csv,
            file_name=f"safety_alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with col2:
        summary_lines = [
            "=" * 50,
            "SAFETY ANALYTICS SUMMARY REPORT",
            "=" * 50,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Time Range: {time_filter}",
            f"Total Alerts: {total}",
            f"Today's Alerts: {today_count}",
            f"Most Common Violation: {most_common}",
            "",
            "BREAKDOWN BY VIOLATION TYPE:",
            "-" * 30,
        ]
        for v, c in df["violation"].value_counts().items():
            summary_lines.append(f"  {v}: {c} alerts")

        if "location" in df.columns:
            summary_lines += ["", "BREAKDOWN BY LOCATION:", "-" * 30]
            for loc, c in df["location"].value_counts().items():
                summary_lines.append(f"  {loc}: {c} alerts")

        st.download_button(
            label="📄 Download Text Summary",
            data="\n".join(summary_lines),
            file_name=f"safety_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            mime="text/plain",
            use_container_width=True,
        )

    # ── Footer ────────────────────────────────────
    st.markdown("---")
    st.markdown(f"""
    <div style="text-align:center; color:#718096; font-size:0.85em; padding:10px;">
        📊 Analytics Dashboard | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
        {"| ⚠️ Showing demo data" if use_demo else "| ✅ Showing real alert data"}
    </div>
    """, unsafe_allow_html=True)