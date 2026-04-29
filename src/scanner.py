import cv2
import json
import time
from pathlib import Path

def test_rtsp_robust(url, target_frames=5):
    """
    Tries to read multiple frames to ensure the stream is decodable and stable.
    """
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        return False
    
    count = 0
    timeout_start = time.time()
    
    while count < target_frames:
        if time.time() - timeout_start > 5: # 5 second timeout per stream path
            break
            
        ret, frame = cap.read()
        if ret and frame is not None and frame.size > 0:
            count += 1
        else:
            # Small delay on failure to let the buffer stabilize
            time.sleep(0.1)
            
    cap.release()
    return count >= target_frames

def scan_network():
    print("Starting Robust RTSP Network Scan (10.6.7.50 - 61)...")
    user = "admin"
    pw = "Admin12345-"
    ips = range(50, 62)
    
    # Expanded Pattern List (Hikvision, Dahua, Generic)
    patterns = [
        # Pattern, Alias
        ("Streaming/Channels/1", "Hik-General"),
        ("Streaming/Channels/101", "Hik-Main"),
        ("Streaming/Channels/102", "Hik-Sub1"),
        ("Streaming/Channels/103", "Hik-Sub2"),
        ("cam/realmonitor?channel=1&subtype=0", "Dahua-Main"),
        ("cam/realmonitor?channel=1&subtype=1", "Dahua-Sub"),
        ("live/ch0", "Legacy-CH0"),
        ("media/video1", "Media-V1"),
        ("1", "Short-1")
    ]
    
    valid_cameras = []
    
    for i in ips:
        base_ip = f"10.6.7.{i}"
        print(f"[{base_ip:12}] Testing Patterns: ", end="", flush=True)
        found_any = False
        
        for p_str, p_name in patterns:
            url = f"rtsp://{user}:{pw}@{base_ip}:554/{p_str}"
            if test_rtsp_robust(url):
                print(f"[{p_name}] ", end="", flush=True)
                valid_cameras.append({
                    "ip": base_ip,
                    "alias": p_name,
                    "url": url
                })
                found_any = True
                # Often multiple patterns work on the same cam (Main vs Sub).
                # We collect ALL that pass the 5-frame test.
        
        if not found_any:
            print("No Signal", end="")
        print()

    # Save
    output_file = Path(__file__).parent / "cameras.json"
    with open(output_file, 'w') as f:
        json.dump(valid_cameras, f, indent=4)
    
    print(f"\nFinal Result: {len(valid_cameras)} verified streams found.")
    print(f"Config saved to: {output_file}")

if __name__ == "__main__":
    scan_network()
