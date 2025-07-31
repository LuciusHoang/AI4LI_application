import os
import json
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import joblib

def extract_feature_from_json(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    all_vectors = []
    for frame in data:
        if not frame:
            continue
        hand = frame[0]  # chỉ lấy tay đầu tiên
        vector = []
        for point in hand:
            vector.extend([point["x"], point["y"], point["z"]])
        all_vectors.append(vector)

    if not all_vectors:
        return None

    return np.mean(all_vectors, axis=0)

def load_dataset_from_folder(folder):
    X, y = [], []

    for filename in os.listdir(folder):
        if not filename.endswith(".json"):
            continue

        label = filename.split("_")[0]  # nhãn từ tên file
        path = os.path.join(folder, filename)
        feature = extract_feature_from_json(path)

        if feature is not None:
            X.append(feature)
            y.append(label)
        else:
            print(f"⚠️ Bỏ qua {filename} vì không có keypoints")

    return np.array(X), np.array(y)

def train_and_evaluate(keypoint_dir, model_path):
    LABELS_PATH = "label_encoder.joblib"

    print("📥 Đang tải dữ liệu...")
    X, y = load_dataset_from_folder(keypoint_dir)

    if len(X) == 0:
        print("❌ Không có dữ liệu hợp lệ để huấn luyện.")
        return

    print(f"🧠 Huấn luyện với {len(X)} mẫu — {len(set(y))} nhãn")
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
    )

    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Độ chính xác: {acc:.2%}")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(clf, model_path)
    joblib.dump(le, LABELS_PATH)
    print(f"💾 Đã lưu mô hình vào: {model_path}")
    print(f"💾 Đã lưu encoder vào: {LABELS_PATH}")
