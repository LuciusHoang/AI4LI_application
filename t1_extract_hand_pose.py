import cv2
import mediapipe as mp
import json
import os
import numpy as np
import argparse

DEFAULT_DATA_DIR = "data"
DEFAULT_VIDEO_PATH = os.path.join(DEFAULT_DATA_DIR, "input_video.mp4")
DEFAULT_OUTPUT_JSON = os.path.join(DEFAULT_DATA_DIR, "hand_keypoints.json")
WINDOW_SIZE = 500  # Kích thước khung hiển thị mô phỏng

def extract_hand_keypoints(video_path, output_json):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False,
                           max_num_hands=2,
                           min_detection_confidence=0.5,
                           min_tracking_confidence=0.5)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[❌] Không mở được video: {video_path}")
        return

    frame_data = {}
    frame_idx = 0

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        joints = []
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for lm in hand_landmarks.landmark:
                    joints.append({
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z
                    })

        frame_data[f"frame_{frame_idx}"] = joints
        frame_idx += 1

    cap.release()
    hands.close()

    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(frame_data, f, indent=2)

    print(f"[✓] Đã lưu tọa độ vào {output_json}")

def replay_hand_keypoints(json_path, window_size=500):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for frame_id in sorted(data.keys(), key=lambda x: int(x.split("_")[1])):
        frame = np.ones((window_size, window_size, 3), dtype=np.uint8) * 255

        for point in data[frame_id]:
            x = int(point["x"] * window_size)
            y = int(point["y"] * window_size)
            cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        cv2.imshow("Replay Hand Movement", frame)
        if cv2.waitKey(50) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract or replay hand keypoints from video.")
    parser.add_argument("--input", type=str, help="Path to input video file")
    parser.add_argument("--output", type=str, help="Path to output JSON file")
    parser.add_argument("--replay", action="store_true", help="Replay extracted keypoints")

    args = parser.parse_args()

    video_path = args.input if args.input else DEFAULT_VIDEO_PATH
    output_json = args.output if args.output else DEFAULT_OUTPUT_JSON

    print(f"🔍 Trích xuất keypoints từ: {video_path}")
    extract_hand_keypoints(video_path, output_json)

    if args.replay:
        print(f"🎞️ Mô phỏng lại từ: {output_json}")
        replay_hand_keypoints(output_json, window_size=WINDOW_SIZE)
