# test_sample_videos.py
import cv2
import os

print("🔍 Checking sample videos...")
print("=" * 50)

video_dir = "sample_videos"
if os.path.exists(video_dir):
    videos = [f for f in os.listdir(video_dir) if f.endswith(('.mp4', '.mov', '.avi'))]
    
    if videos:
        print(f"Found {len(videos)} video(s):")
        print("-" * 50)
        
        for video in videos:
            video_path = os.path.join(video_dir, video)
            cap = cv2.VideoCapture(video_path)
            
            if cap.isOpened():
                fps = int(cap.get(cv2.CAP_PROP_FPS))
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                duration = frames / fps if fps > 0 else 0
                
                print(f"✓ {video}")
                print(f"  Size: {width}x{height}")
                print(f"  Duration: {duration:.1f}s ({frames} frames @ {fps}fps)")
                
                # Try to read first frame
                ret, frame = cap.read()
                if ret:
                    print(f"  ✓ Can read frames")
                else:
                    print(f"  ✗ Cannot read frames")
                
                cap.release()
            else:
                print(f"✗ {video} - Cannot open")
            
            print("-" * 50)
    else:
        print(f"No videos found in '{video_dir}'")
else:
    print(f"Directory '{video_dir}' not found")
    print("Creating it...")
    os.makedirs(video_dir)
    print(f"Add MP4 videos to '{video_dir}' folder")

print("\n✅ Test complete!")
print("Run: streamlit run app.py")