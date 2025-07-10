from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
import subprocess
import time
import os
import shutil
import threading

# === CONFIGURABLE PATHS ===
WATCH_FOLDER = r"C:\Users\neeraj_sachdeva\Desktop\Final_Demo\Demo\Scripts\videos"
PROCESSED_FOLDER = r"C:\Users\neeraj_sachdeva\Desktop\Final_Demo\Demo\Scripts\processed videos"
EXTRACTOR_SCRIPT = r"C:\Users\neeraj_sachdeva\Desktop\Final_Demo\Demo\Scripts\extractor\master_faiss_vector_index.py"
COMPARISON_SCRIPT = r"C:\Users\neeraj_sachdeva\Desktop\Final_Demo\Demo\Scripts\Comparision Logic\master_comparision_vector.py"

# === THREAD LOCKING AND STATE ===
is_processing = False
lock = threading.Lock()

class VideoHandler(FileSystemEventHandler):
    def on_created(self, event):
        global is_processing

        # Ignore directories and non-video files
        if event.is_directory or not event.src_path.lower().endswith(('.mp4', '.mov', '.avi')):
            return

        with lock:
            if is_processing:
                return
            is_processing = True

        # Wait a moment to allow file writes to complete
        time.sleep(2)

        try:
            video_files = [f for f in os.listdir(WATCH_FOLDER) if f.lower().endswith(('.mp4', '.mov', '.avi'))]
            if not video_files:
                print("[INFO] No new videos found.")
                return

            print(f"[INFO] Detected {len(video_files)} video(s).")
            print("[INFO] Running extractor script...")
            subprocess.run(["python", EXTRACTOR_SCRIPT], check=True)

            print("[INFO] Running comparison script...")
            subprocess.run(["python", COMPARISON_SCRIPT], check=True)

            print("[INFO] Moving videos to processed folder...")
            for file in video_files:
                src = os.path.join(WATCH_FOLDER, file)
                dst = os.path.join(PROCESSED_FOLDER, file)
                if os.path.exists(src):
                    shutil.move(src, dst)
                    print(f"[MOVED] {file} ➜ processed videos")
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] Script failed: {e}")
        finally:
            # Cooldown period before watcher becomes reactive again
            time.sleep(1)
            with lock:
                is_processing = False

if __name__ == "__main__":
    os.makedirs(PROCESSED_FOLDER, exist_ok=True)

    print(f"[WATCHING] Folder: {WATCH_FOLDER}")
    observer = Observer()
    observer.schedule(VideoHandler(), path=WATCH_FOLDER, recursive=False)
    observer.start()

    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        print("\n[STOPPED] Monitoring stopped by user.")
        observer.stop()
    observer.join()
