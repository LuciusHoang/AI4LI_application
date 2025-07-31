import os
import cv2
import json
import mediapipe as mp

# Đường dẫn thư mục
CLIPS_DIR = "data/clips"
OUT_DIR = "data/keypoints"
os.makedirs(OUT_DIR, exist_ok=True)

# Khởi tạo MediaPipe Hands
mp_hands = mp.solutions.hands
Hands = mp_hands.Hands

def extract_keypoints_from_clip(clip_path, output_path):
    cap = cv2.VideoCapture(clip_path)
    if not cap.isOpened():
        print(f"[❌] Không mở được video: {clip_path}")
        return

    all_keypoints = []

    with Hands(
        static_image_mode=False,
        max_num_hands=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:

        while True:
            success, frame = cap.read()
            if not success:
                break

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(image)

            frame_keypoints = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = [{
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    } for lm in hand_landmarks.landmark]
                    frame_keypoints.append(keypoints)

            all_keypoints.append(frame_keypoints)

    cap.release()

    with open(output_path, "w") as f:
        json.dump(all_keypoints, f)

    print(f"[✓] Đã lưu keypoints: {output_path}")

def run_batch_extraction():
    print("🚀 Bắt đầu trích xuất keypoints...")

    for filename in os.listdir(CLIPS_DIR):
        if not filename.endswith(".mp4"):
            continue

        input_path = os.path.join(CLIPS_DIR, filename)
        output_filename = os.path.splitext(filename)[0] + ".json"
        output_path = os.path.join(OUT_DIR, output_filename)

        print(f"📌 Đang xử lý: {filename}")
        extract_keypoints_from_clip(input_path, output_path)

if __name__ == "__main__":
    run_batch_extraction()
