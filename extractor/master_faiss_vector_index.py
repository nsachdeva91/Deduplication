import os
import re
import uuid
import faiss
import pickle
import pytesseract
import numpy as np
import imagehash
import cv2
from PIL import Image
from pydub import AudioSegment
from python_speech_features import mfcc
from datetime import datetime
from sklearn.preprocessing import normalize
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
VIDEO_DIR = r"C:\Users\neeraj_sachdeva\Desktop\Final_Demo\Demo\Scripts\videos"
INDEX_DIR = "faiss_indices"
SAMPLE_RATE = 16000
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# === FAISS UTILS ===
def create_or_load_index(index_path, dim):
    return faiss.read_index(index_path) if os.path.exists(index_path) else faiss.IndexFlatIP(dim)

def save_index(index, path):
    faiss.write_index(index, path)

def normalize_vec(vec):
    return normalize([vec])[0].astype(np.float32)

# === FEATURE EXTRACTION ===
def extract_frames(path, interval=30):
    cap = cv2.VideoCapture(path)
    results, count = [], 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % interval == 0:
            resized = cv2.resize(frame, (320, int(320 * frame.shape[0] / frame.shape[1])))
            img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
            hash_vec = np.array(imagehash.phash(img).hash, dtype=np.uint8).flatten()
            text = pytesseract.image_to_string(img)
            results.append((hash_vec, text.strip()))
        count += 1
    cap.release()
    return results

def extract_audio(path):
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(SAMPLE_RATE)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32)

    if len(samples) == 0:
        print(f"⚠️ Audio stream is empty: {os.path.basename(path)}")
        return np.zeros(13, dtype=np.float32), 0.0, 0.0

    samples /= np.max(np.abs(samples)) if np.max(np.abs(samples)) > 0 else 1
    total_sec = round(len(samples) / SAMPLE_RATE, 1)
    win = int(SAMPLE_RATE * 0.5)
    active_sec = round(sum(np.std(samples[i:i + win]) > 0.01 for i in range(0, len(samples), win)) * 0.5, 1)

    is_silent = np.std(samples) < 0.01 or active_sec < 0.5
    if is_silent:
        print(f"🔇 Detected as silent audio: {os.path.basename(path)} (ActiveSec={active_sec})")
        return np.zeros(13, dtype=np.float32), total_sec, active_sec

    mfcc_features = mfcc(samples, samplerate=SAMPLE_RATE, numcep=13)
    if mfcc_features.shape[0] == 0:
        print(f"⚠️ No MFCC features extracted: {os.path.basename(path)}")
        return np.zeros(13, dtype=np.float32), total_sec, active_sec

    mfcc_vec = np.mean(mfcc_features, axis=0).astype(np.float32)
    return mfcc_vec, total_sec, active_sec

def phash_vector(frames):
    return np.mean([v for v, _ in frames], axis=0).astype(np.float32)

def text_embedding(frames):
    text = ' '.join([t for _, t in frames if t]).strip().lower()
    clean = re.sub(r"\s+", " ", text)
    return embedder.encode(clean).astype(np.float32)

# === MAIN ===
def main():
    os.makedirs(INDEX_DIR, exist_ok=True)
    audio_index = create_or_load_index(os.path.join(INDEX_DIR, "audio.index"), 13)
    video_index = create_or_load_index(os.path.join(INDEX_DIR, "video.index"), 64)
    text_index  = create_or_load_index(os.path.join(INDEX_DIR, "text.index"), 384)

    metadata = []

    for fname in sorted(os.listdir(VIDEO_DIR)):
        if not fname.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            continue

        print(f"\n🎞️ Processing: {fname}")
        path = os.path.join(VIDEO_DIR, fname)
        uid = str(uuid.uuid4())

        try:
            frames = extract_frames(path)

            try:
                audio_vec, dur_total, dur_active = extract_audio(path)
                silent = np.all(audio_vec == 0)
            except Exception as ae:
                print(f"⚠️ Audio issue in {fname}: {ae}")
                audio_vec = np.zeros(13, dtype=np.float32)
                dur_total, dur_active = 0.0, 0.0
                silent = True

            video_vec = phash_vector(frames)
            text_vec  = text_embedding(frames)

            norm_audio = normalize_vec(audio_vec)
            norm_video = normalize_vec(video_vec)
            norm_text  = normalize_vec(text_vec)

            audio_index.add(np.array([norm_audio]))
            video_index.add(np.array([norm_video]))
            text_index.add(np.array([norm_text]))

            metadata.append({
                "uid": uid,
                "filename": fname,
                "duration": dur_total,
                "active_audio": dur_active,
                "silent_audio": silent,
                "ingested_at": datetime.utcnow().isoformat()
            })

            print(f"✅ Indexed: {fname} | Silent={silent} | AudioActive={dur_active}s")

        except Exception as e:
            print(f"❌ Skipped {fname} → {e}")

    # === SAVE TO DISK ===
    save_index(audio_index, os.path.join(INDEX_DIR, "audio.index"))
    save_index(video_index, os.path.join(INDEX_DIR, "video.index"))
    save_index(text_index,  os.path.join(INDEX_DIR, "text.index"))

    with open(os.path.join(INDEX_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)

    print(f"\n📦 All indices and metadata saved to: {INDEX_DIR}")

if __name__ == "__main__":
    main()
