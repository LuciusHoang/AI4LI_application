import json
import cv2
import numpy as np
import os
import argparse

# Mặc định
DATA_DIR = "data"
DEFAULT_JSON = os.path.join(DATA_DIR, "hand_keypoints.json")
OUTPUT_VIDEO = os.path.join(DATA_DIR, "hand_replay.mp4")
WINDOW_SIZE = 500  # Kích thước hiển thị khung tay
FPS = 20

def replay_hand_keypoints(json_path, window_size=500, save_video=True):
    if not os.path.exists(json_path):
        print(f"[❌] Không tìm thấy file: {json_path}")
        return

    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict):
        frame_keys = sorted(data.keys(), key=lambda x: int(x.split("_")[1]))
    else:
        # dạng list từ MediaPipe
        frame_keys = range(len(data))

    print(f"[▶] Phát chuyển động từ {json_path}... (nhấn 'q' để thoát)")

    # Khởi tạo video writer nếu cần lưu
    writer = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, FPS, (window_size, window_size))

    for idx, key in enumerate(frame_keys):
        frame = np.ones((window_size, window_size, 3), dtype=np.uint8) * 255

        keypoints = data[key] if isinstance(data, dict) else data[key]
        if keypoints:
            hand = keypoints[0] if isinstance(keypoints[0], list) else keypoints
            for point in hand:
                x = int(point["x"] * window_size)
                y = int(point["y"] * window_size)
                cv2.circle(frame, (x, y), 5, (0, 0, 255), -1)

        label = f"Frame {key}" if isinstance(key, str) else f"Frame {idx}"
        cv2.putText(frame, label, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
        cv2.imshow("Replay Hand Movement", frame)

        if writer:
            writer.write(frame)

        if cv2.waitKey(int(1000 / FPS)) & 0xFF == ord("q"):
            break

    if writer:
        writer.release()
        print(f"[💾] Đã lưu video mô phỏng vào: {OUTPUT_VIDEO}")

    cv2.destroyAllWindows()
    print("[✓] Đã hoàn tất mô phỏng.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Replay hand movement from JSON file")
    parser.add_argument("--input", type=str, default=DEFAULT_JSON,
                        help="Path to input .json file containing hand keypoints")
    args = parser.parse_args()

    replay_hand_keypoints(args.input, window_size=WINDOW_SIZE)
