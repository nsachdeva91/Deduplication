import numpy as np
import csv
import pickle
import faiss
import os
from itertools import combinations
from collections import Counter
import matplotlib.pyplot as plt
from tabulate import tabulate
from colorama import Fore, Style, init

init(autoreset=True)

# === CONFIGURATION ===
INDEX_DIR = "faiss_indices"

THRESHOLDS = {
    "audio": 0.95,
    "video": 0.95,
    "text":  0.60
}

MATCH_LABELS = [
    "Duplicate",
    "Audio+Video Match",
    "Audio+Text Match",
    "Video+Text Match",
    "Audio Only Match",
    "Video Only Match",
    "Text Only Match",
    "No Match"
]

# === LOAD INDEX & METADATA ===
def load_index(name): return faiss.read_index(os.path.join(INDEX_DIR, name))
def load_metadata(path="metadata.pkl"):
    with open(os.path.join(INDEX_DIR, path), "rb") as f:
        return pickle.load(f)

audio_index = load_index("audio.index")
video_index = load_index("video.index")
text_index  = load_index("text.index")
metadata    = load_metadata()
filenames   = [m["filename"] for m in metadata]

# === RECONSTRUCT VECTORS ===
audio_vecs = np.array([audio_index.reconstruct(i) for i in range(audio_index.ntotal)])
video_vecs = np.array([video_index.reconstruct(i) for i in range(video_index.ntotal)])
text_vecs  = np.array([text_index.reconstruct(i)  for i in range(text_index.ntotal)])

# === COSINE SIMILARITY ===
def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))) if np.linalg.norm(a) and np.linalg.norm(b) else 0.0

# === MATCH CATEGORY ASSIGNMENT ===
def get_match_label(a_sim, v_sim, t_sim):
    a, v, t = (
        a_sim >= THRESHOLDS["audio"],
        v_sim >= THRESHOLDS["video"],
        t_sim >= THRESHOLDS["text"]
    )
    if a and v and t: return "Duplicate"
    elif a and v: return "Audio+Video Match"
    elif a and t: return "Audio+Text Match"
    elif v and t: return "Video+Text Match"
    elif a: return "Audio Only Match"
    elif v: return "Video Only Match"
    elif t: return "Text Only Match"
    return "No Match"

# === COMPACT SYMBOLS WITH COLOR ===
def get_tick(value, threshold): 
    return f"{Fore.GREEN}✓{Style.RESET_ALL}" if value >= threshold else f"{Fore.RED}✗{Style.RESET_ALL}"

# === ANALYSIS ===
csv_rows = [["File A", "File B", "AudioSim", "VideoSim", "TextSim", "Match Category"]]
category_counter = Counter()
table_data = []

print("\n📋 Vector Comparison Summary\n")

for i, j in combinations(range(len(filenames)), 2):
    f1, f2 = filenames[i], filenames[j]
    a_sim = cosine_similarity(audio_vecs[i], audio_vecs[j])
    v_sim = cosine_similarity(video_vecs[i], video_vecs[j])
    t_sim = cosine_similarity(text_vecs[i], text_vecs[j])

    label = get_match_label(a_sim, v_sim, t_sim)
    category_counter[label] += 1

    file_a = os.path.basename(f1)
    file_b = os.path.basename(f2)

    audio_tick = get_tick(a_sim, THRESHOLDS["audio"])
    video_tick = get_tick(v_sim, THRESHOLDS["video"])
    text_tick  = get_tick(t_sim, THRESHOLDS["text"])

    label_color = (
        Fore.GREEN if label in ["Audio+Video Match", "Audio+Text Match", "Video+Text Match"]
        else Fore.RED
    )
    if label == "Duplicate":
        label_color = Fore.RED

    row = [file_a, file_b, f"{a_sim:.4f}", f"{v_sim:.4f}", f"{t_sim:.4f}", label]
    csv_rows.append(row)

    table_data.append([
        file_a,
        file_b,
        f"{a_sim:.4f} {audio_tick}",
        f"{v_sim:.4f} {video_tick}",
        f"{t_sim:.4f} {text_tick}",
        f"{label_color}{label}{Style.RESET_ALL}"
    ])

# === FORMATTED TABLE ===
headers = ["File A", "File B", "Audio Sim", "Video Sim", "Text Sim", "Match Category"]

print(tabulate(
    table_data,
    headers=headers,
    tablefmt="fancy_grid",
    stralign="left",
    numalign="center",
    maxcolwidths=[25, 25, 25, 25, 25, 40]
))

# === EXPORT CSV ===
with open("vector_match_analysis.csv", "w", newline="", encoding="utf-8") as f:
    csv.writer(f).writerows(csv_rows)

print("\n📄 CSV saved: vector_match_analysis.csv")

# === VISUALIZATION ===
labels = MATCH_LABELS
counts = [category_counter.get(label, 0) for label in labels]
colors = [
    "#F44336", "#4CAF50", "#4CAF50", "#4CAF50", 
    "#F44336", "#F44336", "#F44336", "#F44336"
]

plt.figure(figsize=(12, 6))
bars = plt.bar(labels, counts, color=colors)
plt.title("📊 Pairwise Match Distribution Across Modalities")
plt.xlabel("Match Category")
plt.ylabel("Number of Pairs")
plt.xticks(rotation=30)

for bar, count in zip(bars, counts):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), str(count),
             ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()
