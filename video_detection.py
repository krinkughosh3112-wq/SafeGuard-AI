# video_detection.py
import cv2
from ultralytics import YOLO
import os

def process_video_for_app(video_path, model):
    """
    Process a single video frame by frame for Streamlit app
    Returns: Processed frames as a generator
    """
    cap = cv2.VideoCapture(video_path)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Run detection
        results = model(frame)
        processed_frame = results[0].plot()
        
        yield processed_frame
    
    cap.release()

def save_processed_video(input_path, output_path="output_safety.mp4"):
    """
    Process and save a video with safety detections
    """
    model = YOLO('best.pt')
    
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Process frame
        results = model(frame)
        processed_frame = results[0].plot()
        
        # Add frame counter
        cv2.putText(processed_frame, f"Frame: {frame_count}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        out.write(processed_frame)
    
    cap.release()
    out.release()
    
    print(f"✅ Video saved: {output_path}")
    return output_path