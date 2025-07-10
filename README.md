**Video Deduplication Pipeline**

An automated system for ingesting videos, extracting multi-modal features (audio, visual, and text), and comparing them using vector similarity across modalities. Designed for fast, scalable video analysis and detection of duplicates or near-matches.

🧰 **Features**

🔍 Watches a folder for incoming video files
🖼️ Extracts perceptual hashes from video frames
🧠 Embeds visible text from frames using a transformer model
🔊 Processes audio with MFCC feature extraction and silence detection
🎯 Compares videos pairwise across audio, video, and text modalities
📊 Generates CSV reports and visual match distribution charts
📁 Moves processed files automatically to a backup directory

📦 **Requirements**

Before running the code, install dependencies listed in **requirements.txt**

<img width="340" alt="{3DACCCCC-32CB-400C-ACE2-938B7DFCEB22}" src="https://github.com/user-attachments/assets/ed6c9e1d-8661-42b9-806a-8592b1ec9c0e" />

Install all Python dependencies:

<img width="233" alt="{2097BB58-6E30-42D6-B76A-C01EE5425343}" src="https://github.com/user-attachments/assets/7c9aec8b-edda-4feb-bed6-7350a6c67cec" />

Make sure Tesseract OCR is installed

📁 **Folder Structure**

<img width="371" alt="{AFB7D972-5436-4EFD-85A6-DB63BA233CB8}" src="https://github.com/user-attachments/assets/83f93371-6811-49f9-a9a9-f87d4c23ae11" />


🚀 **How to Run**

Start the watcher script:

<img width="198" alt="{D3C8518D-98B8-485A-9559-EAD1197340BD}" src="https://github.com/user-attachments/assets/1f254215-ceba-4999-a3cc-7e186dbf0d27" />

This will continuously monitor **videos/** for new files.
Automatic processing flow:
Extractor script runs on all videos present
Comparison script evaluates video similarities
Generates **vector_match_analysis.csv** and chart
All videos are moved to processed **videos/**

📊 **Output Artifacts**

**vector_match_analysis.csv**: Full similarity report across video pairs
**Chart visualization**: Bar plot of match categories
**faiss_indices/**: Contains 3 FAISS indices and metadata.pkl for processed files
