import os
import argparse

# ====== CẤU HÌNH ======
DATA_DIR = "data"
VIDEO_PATH = os.path.join(DATA_DIR, "input_video.mp4")
JSON_PATH = os.path.join(DATA_DIR, "hand_keypoints.json")
MODEL_PATH = os.path.join(DATA_DIR, "gesture_classifier.joblib")
KEYPOINT_DIR = os.path.join(DATA_DIR, "keypoints")

# ====== TASK 1: Trích xuất khớp tay ======
def run_task1():
    print("🔍 Task 1: Trích xuất khớp tay từ video...")
    from t1_extract_hand_pose import extract_hand_keypoints
    extract_hand_keypoints(VIDEO_PATH, JSON_PATH)

# ====== TASK 2: Mô phỏng lại chuyển động ======
def run_task2():
    print("🎞️ Task 2: Mô phỏng lại chuyển động từ JSON...")
    from t2_replay_hand_pose import replay_hand_keypoints
    replay_hand_keypoints(JSON_PATH)

# ====== TASK 3: Huấn luyện mô hình phân loại ======
def run_task3():
    print("🤖 Task 3: Huấn luyện mô hình phân loại cử chỉ...")
    from t3_train_gesture_classifier import train_and_evaluate

    if not os.path.exists(KEYPOINT_DIR):
        print(f"[❌] Không tìm thấy thư mục {KEYPOINT_DIR}")
        return

    train_and_evaluate(KEYPOINT_DIR, MODEL_PATH)

# ====== MAIN ENTRY ======
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AI4LI - Điều phối pipeline")
    parser.add_argument("--task", type=int, choices=[1, 2, 3], required=True, help="Chọn task để chạy (1-3)")
    args = parser.parse_args()

    if args.task == 1:
        run_task1()
    elif args.task == 2:
        run_task2()
    elif args.task == 3:
        run_task3()
