# 📘 Dự án Nhận diện Ngôn ngữ Ký hiệu Tiếng Việt (AI4LI)

Ứng dụng hỗ trợ người khiếm thính nhận diện cử chỉ tay chữ cái tiếng Việt qua video, sử dụng MediaPipe & mô hình học máy.

---

## 📁 Cấu trúc thư mục

```
AI4LI_application/
├── data/
│   ├── raw_video/              # Video gốc tải từ YouTube
│   ├── clips/                  # Clip đã cắt theo nhãn (A.mp4, B.mp4, ...)
│   ├── keypoints/              # File JSON khớp tay từng clip
│   ├── labels.csv              # Thời gian & nhãn cắt clip
│   ├── input_video.mp4         # Video đơn lẻ để test nhanh (Task 1)
│   └── hand_keypoints.json     # Kết quả tọa độ khớp tay (Task 1)
│
├── scripts/
│   ├── download_youtube_batch.py     # Tải video từ YouTube
│   ├── cut_clips_from_labels.py      # Cắt clip từ labels.csv
│   └── extract_keypoints_batch.py    # Trích keypoints từ các clip
│
├── t1_extract_hand_pose.py     # Task 1: Trích khớp tay từ video
├── t2_replay_hand_pose.py      # Task 2: Mô phỏng chuyển động từ JSON
├── t3_train_gesture_classifier.py  # Task 3: Huấn luyện mô hình phân loại
├── main.py                     # Chạy từng task pipeline
└── README.md
```

---

## ❓ Phân biệt `input_video.mp4` và `data/clips/`

| Thành phần           | Mục đích sử dụng                                                                 |
|----------------------|----------------------------------------------------------------------------------|
| `data/input_video.mp4` | Dùng để **test nhanh toàn bộ video**, demo hoặc trích thử khớp tay trong 1 lần |
| `data/clips/`        | Chứa **các đoạn video đã cắt theo từng cử chỉ (A, B, C, ...)**, dùng để huấn luyện |
| `hand_keypoints.json`| Tọa độ trích từ `input_video.mp4`                                               |
| `keypoints/`         | Tọa độ trích từ các clip nhỏ trong `data/clips/`                                |

💡 Nếu bạn đang xây dựng bộ dữ liệu để huấn luyện mô hình, bạn có thể **bỏ qua `input_video.mp4`** và chỉ làm việc với `clips/` + `keypoints/`.

---

## 🚀 Hướng dẫn sử dụng

### ✅ Task 1 – Trích xuất khớp tay từ video
```bash
python main.py --task 1
```

### ✅ Task 2 – Mô phỏng chuyển động từ JSON
```bash
python main.py --task 2
```

### ✅ Task 3 – Huấn luyện mô hình phân loại
```bash
python main.py --task 3
```

---

## 📦 Xử lý dữ liệu (batch)

1. **Tải video từ YouTube**:
```bash
python scripts/download_youtube_batch.py
```

2. **Điền `labels.csv` để chỉ định thời gian cắt clip**
```csv
filename,label,start_time,end_time
video1.mp4,A,10,12
video1.mp4,B,13,15
```

3. **Cắt clip theo nhãn**:
```bash
python scripts/cut_clips_from_labels.py
```

4. **Trích khớp tay từ tất cả clip**:
```bash
python scripts/extract_keypoints_batch.py
```

---

## 📌 Yêu cầu môi trường

- Python >= 3.8
- Thư viện:
```bash
pip install mediapipe opencv-python numpy scikit-learn joblib moviepy pytube
```
