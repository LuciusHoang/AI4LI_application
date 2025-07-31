import os
import yt_dlp

# ✅ Danh sách 3 URL đã cập nhật
video_links = [
    "https://www.youtube.com/watch?v=ggQY-g4aQp8",
    "https://www.youtube.com/watch?v=hh1olkNNbYE",
    "https://www.youtube.com/watch?v=VKUVTA9Lla0"
]

SAVE_DIR = "data/raw"
os.makedirs(SAVE_DIR, exist_ok=True)

def download_video(url, save_path):
    try:
        ydl_opts = {
            'outtmpl': os.path.join(save_path, '%(id)s.%(ext)s'),  # lưu theo ID YouTube
            'format': 'mp4',
            'quiet': True,
            'noplaylist': True,
        }
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        print(f"✅ Đã tải: {url}")
    except Exception as e:
        print(f"❌ Lỗi khi tải {url} — {e}")

if __name__ == "__main__":
    print("🚀 Bắt đầu tải video...")

    for url in video_links:
        download_video(url, SAVE_DIR)

    print("✅ Hoàn tất.")
