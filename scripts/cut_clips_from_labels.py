import os
import csv
import cv2
import unicodedata
import subprocess

def time_to_seconds(t):
    """
    Chuyển đổi thời gian (ss hoặc mm:ss) về số giây float
    """
    if isinstance(t, str) and ":" in t:
        parts = t.strip().split(":")
        if len(parts) == 2:
            minutes, seconds = map(int, parts)
            return minutes * 60 + seconds
    return float(t)

def normalize_unicode(s):
    """
    Chuẩn hóa chuỗi Unicode tổ hợp về chuẩn NFC
    """
    return unicodedata.normalize("NFC", s)

def get_video_duration(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()
    return frames / fps if fps > 0 else 0

def cut_clip_ffmpeg(in_path, out_path, start, end):
    cmd = [
        "ffmpeg",
        "-y",  # overwrite
        "-ss", str(start),
        "-to", str(end),
        "-i", in_path,
        "-c:v", "libx264",
        "-c:a", "aac",
        "-strict", "experimental",
        out_path
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return result.returncode == 0

def cut_all_from_csv(csv_path="data/labels.csv",
                     raw_dir="data/raw",
                     out_dir="data/clips"):
    os.makedirs(out_dir, exist_ok=True)
    count_map = {}

    with open(csv_path, newline='', encoding='utf-8-sig') as csvfile:
        reader = csv.DictReader(csvfile)

        for row in reader:
            try:
                filename = normalize_unicode(row['filename'].strip())
                label = normalize_unicode(row['label'].strip().upper())
                start = time_to_seconds(row['start_time'])
                end = time_to_seconds(row['end_time'])

                video_path = os.path.join(raw_dir, filename)
                if not os.path.exists(video_path):
                    print(f"⚠️ Không tìm thấy video: {video_path}")
                    continue

                duration = get_video_duration(video_path)
                if end > duration:
                    print(f"⚠️ Bỏ qua: {filename} ({end:.1f}s > {duration:.1f}s)")
                    continue

                count = count_map.get(label, 0) + 1
                count_map[label] = count

                out_filename = f"{label}_{count}.mp4"
                out_path = os.path.join(out_dir, out_filename)

                success = cut_clip_ffmpeg(video_path, out_path, start, end)
                if success:
                    print(f"✂️ Cắt {label}: {start}s → {end}s → {out_filename}")
                else:
                    print(f"❌ Lỗi khi cắt clip: {filename} ({start}s → {end}s)")

            except Exception as e:
                print(f"❌ Lỗi khi xử lý dòng: {row} — {e}")

if __name__ == "__main__":
    cut_all_from_csv()
